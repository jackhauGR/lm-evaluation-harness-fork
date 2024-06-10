import asyncio  # for running API calls concurrently
import atexit
import copy
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os
from pathlib import Path
import re  # for matching endpoint from request URL
import signal
import tempfile
import time  # for sleeping after rate limit is hit
from collections import defaultdict
from requests import get as get_request
import google.generativeai as genai

# Openai parallel processor imports
from dataclasses import (
    dataclass,
    field,
)

# for storing API inputs, outputs, and metadata
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple

import aiohttp  # for making API calls concurrently
import tiktoken  # for counting tokens
from tokenizers import Tokenizer
from tqdm import tqdm

import lm_eval.models.utils
from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import retry_on_specific_exceptions
from lm_eval.utils import eval_logger


async def process_api_requests_from_file(
    requests_filepath: str,
    save_filepath: str,
    request_url: str,
    api_key: str,
    max_requests_per_minute: float,
    max_tokens_per_minute: float,
    token_encoding_name: str,
    max_attempts: int,
    logging_level: int,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = request_url
    request_header = {'content-type': 'application/json',}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 0, 1, 2, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug("Initialization complete.")

    # initialize file reading
    with open(requests_filepath) as file:
        # `requests` will provide requests one at a time
        requests = file.__iter__()
        logging.debug("File opened. Entering main loop")
        async with aiohttp.ClientSession() as session:  # Initialize ClientSession here
            while True:
                # get next request (if one is not already waiting for capacity)
                if next_request is None:
                    if not queue_of_requests_to_retry.empty():
                        next_request = queue_of_requests_to_retry.get_nowait()
                        logging.debug(
                            f"Retrying request {next_request.task_id}: {next_request}"
                        )
                    elif file_not_finished:
                        try:
                            # get new request
                            request_json = json.loads(next(requests))
                            next_request = APIRequest(
                                task_id=next(task_id_generator),
                                request_json=request_json,
                                token_consumption=num_tokens_consumed_from_request(
                                    request_json, api_endpoint, token_encoding_name
                                ),
                                attempts_left=max_attempts,
                                metadata=request_json.pop("metadata", None),
                            )
                            status_tracker.num_tasks_started += 1
                            status_tracker.num_tasks_in_progress += 1
                            logging.debug(
                                f"Reading request {next_request.task_id}: {next_request}"
                            )
                        except StopIteration:
                            # if file runs out, set flag to stop reading it
                            logging.debug("Read file exhausted")
                            file_not_finished = False

                # update available capacity
                current_time = time.time()
                seconds_since_update = current_time - last_update_time
                available_request_capacity = min(
                    available_request_capacity
                    + max_requests_per_minute * seconds_since_update / 60.0,
                    max_requests_per_minute,
                )
                available_token_capacity = min(
                    available_token_capacity
                    + max_tokens_per_minute * seconds_since_update / 60.0,
                    max_tokens_per_minute,
                )
                last_update_time = current_time

                # if enough capacity available, call API
                if next_request:
                    next_request_tokens = next_request.token_consumption
                    if (
                        available_request_capacity >= 1
                        and available_token_capacity >= next_request_tokens
                    ):
                        # update counters
                        available_request_capacity -= 1
                        available_token_capacity -= next_request_tokens
                        next_request.attempts_left -= 1

                        # call API
                        asyncio.create_task(
                            next_request.call_api(
                                session=session,
                                request_url=request_url,
                                request_header=request_header,
                                retry_queue=queue_of_requests_to_retry,
                                save_filepath=save_filepath,
                                status_tracker=status_tracker,
                            )
                        )
                        next_request = None  # reset next_request to empty

                # if all tasks are finished, break
                if status_tracker.num_tasks_in_progress == 0:
                    break

                # main loop sleeps briefly so concurrent tasks can run
                await asyncio.sleep(seconds_to_sleep_each_loop)

                # if a rate limit error was hit recently, pause to cool down
                seconds_since_rate_limit_error = (
                    time.time() - status_tracker.time_of_last_rate_limit_error
                )
                if (
                    seconds_since_rate_limit_error
                    < seconds_to_pause_after_rate_limit_error
                ):
                    remaining_seconds_to_pause = (
                        seconds_to_pause_after_rate_limit_error
                        - seconds_since_rate_limit_error
                    )
                    await asyncio.sleep(remaining_seconds_to_pause)
                    # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
                    logging.warn(
                        f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
                    )

        # after finishing, log final status
        logging.info(
            f"""Parallel processing complete. Results saved to {save_filepath}"""
        )
        if status_tracker.num_tasks_failed > 0:
            logging.warning(
                f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
            )
        if status_tracker.num_rate_limit_errors > 0:
            logging.warning(
                f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
            )


@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


def task_id_generator_function():
    """Generate integers 0, 1, 2, and so on."""
    task_id = 0
    while True:
        yield task_id
        task_id += 1


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        session: aiohttp.ClientSession,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: str,
        status_tracker: StatusTracker,
    ):

        """Calls the Anthropic API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None

        try:

            async with session.post(
                url=request_url, headers=request_header, data=json.dumps(self.request_json)
            ) as response:
                response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "rate limit" in response["error"].get("message", "") or response["error"].get("code") ==  403:
                    status_tracker.time_of_last_rate_limit_error = time.time()
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e
        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


def append_to_jsonl(data, filename: str) -> None:
    """Append a json payload to the end of a jsonl file."""
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")

def process_chat_request(messages, model, idx, max_out_len, temperature, **kwargs):

    request = {
        "contents": [{'role': 'user', 'parts': [{'text': messages}]}],
        "model": model,
        "metadata": {"idx": idx},
        'safetySettings': [
            {
                'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                'threshold': 'BLOCK_NONE'
            },
            {
                'category': 'HARM_CATEGORY_HATE_SPEECH',
                'threshold': 'BLOCK_NONE'
            },
            {
                'category': 'HARM_CATEGORY_HARASSMENT',
                'threshold': 'BLOCK_NONE'
            },
            {
                'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                'threshold': 'BLOCK_NONE'
            },
        ],
        'generationConfig': {
                'candidate_count': 1,
                'temperature': temperature,
                'maxOutputTokens': max_out_len
            }
        
    }

    # request.update(kwargs)

    return json.dumps(request)

def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    model_name: str,
):
    """Count the number of tokens in the request. Only supports completion and embedding requests."""

    model_str = "models/" + model_name

    model = genai.GenerativeModel(model_str)
    model_info = genai.get_model(model_str)

    input_tokens_limit = model_info.input_token_limit
    output_tokens_limit = model_info.output_token_limit

    max_tokens = request_json.get("generationConfig", {}).get("maxOutputTokens", output_tokens_limit)
    n = request_json.get("generationConfig", {}).get("candidate_count", 1)

    completion_tokens = n * max_tokens

    prompt = request_json["contents"][0]["parts"][0]["text"]

    prompt_tokens = model.count_tokens(prompt).total_tokens

    return prompt_tokens + completion_tokens


@register_model("google-chat")
class GoogleChatLM(LM):
    def __init__(
            self,
            model: str,
            batch_size: int = 1,
            max_tokens: int = 256,
            temperature: float = 0,  # defaults to 0
            token_counter_loc: str = "TOKEN_COUNTER.json",
            **kwargs,  # top_p, top_k, etc.
        ) -> None:
            """Anthropic API wrapper.

            :param model: str
                Google model e.g. 'gemini-1.0-pro', 'gemini-1.5-pro'
            :param max_tokens: int
                Maximum number of tokens to sample from the model
            :param temperature: float
                Sampling temperature
            :param kwargs: Any
                Additional model_args to pass to the API client
            """
            super().__init__()

            self.model = model
            self.temperature = temperature
            self.max_tokens = genai.get_model("models/"+model).output_token_limit

            self.kwargs = kwargs
            self.token_counter = Path(token_counter_loc)

            self.api_key = os.environ["GOOGLE_API_KEY"]

            self.url = f'https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={self.api_key}'


    @property
    def max_length(self) -> int:
        return 218_000

    @property
    def max_gen_toks(self) -> int:
        return 4_000

    @property
    def batch_size(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override _loglikelihood_tokens
        raise NotImplementedError()
    
    def generate_until(self, requests) -> List[str]:
        
        temp_dir = tempfile.gettempdir()
        requests_file_path = os.path.join(
            temp_dir, "lm_eval_harness_requests.jsonl"
        )
        responses_file_path = os.path.join(
            temp_dir, "lm_eval_harness_responses.jsonl"
        )

        
        def clean_up_requests(signum=None, frame=None):
            if os.path.exists(requests_file_path):
                os.remove(requests_file_path)
                print("Cached requests temp file deleted.")
            if os.path.exists(responses_file_path):
                os.remove(responses_file_path)
                print("Cached responses temp file deleted.")
            if signum is not None:
                exit(0)  # Only exit if called by a signal

        atexit.register(clean_up_requests)
        signal.signal(signal.SIGINT, clean_up_requests)
        signal.signal(signal.SIGTERM, clean_up_requests)


        pbar = tqdm(total=len(requests), disable=(self.rank != 0))
        with open(requests_file_path, "w") as file:
            for idx, request in enumerate(requests):
    
                message = request.args[0]

                gen_kwargs = request.args[1]

                until = None
                if isinstance(kwargs := copy.deepcopy(gen_kwargs), dict):
                    if "do_sample" in kwargs.keys():
                        kwargs.pop("do_sample")
                    if "until" in kwargs.keys():
                        until = kwargs.pop("until")
                        if isinstance(until, str):
                            until = [until]
                        elif not isinstance(until, list):
                            raise ValueError(
                                f"Expected repr(kwargs['until']) to be of type Union[str, list] but got {until}"
                            )
                        # kwargs["stop_sequences"] = until[:5]

                    if "temperature" in kwargs.keys():
                        kwargs.pop("temperature")

                    if "max_requests_per_minute" in kwargs.keys():
                        max_requests_per_minute = kwargs.pop(
                            "max_requests_per_minute"
                        )
                    else:
                        max_requests_per_minute = 360  # Google Production Key limit
                    if "max_attempts" in kwargs.keys():
                        max_attempts = kwargs.pop("max_attempts")
                    else:
                        max_attempts = 10
                    kwargs["max_tokens"] = kwargs.pop(
                        "max_gen_toks", self.max_gen_toks
                    )
                else:
                    raise ValueError(
                        f"Expected repr(kwargs) to be of type repr(dict) but got {kwargs}"
                    )

                request_to_cache = process_chat_request(
                    messages=message,
                    model=self.model,
                    idx=idx,
                    temperature=self.temperature,
                    max_out_len=int(self.max_tokens),
                    **kwargs,
                )
                file.write(request_to_cache + "\n")
            
            print(f"Requests written to {requests_file_path}.")

        print(f"Max requests per minute: {max_requests_per_minute}")
        print(
            "use --gen_kwargs max_requests_per_minute=N to override"
        )

        asyncio.run(
            process_api_requests_from_file(
                requests_filepath=str(requests_file_path),
                save_filepath=str(responses_file_path),
                request_url=str(self.url),
                api_key=str(self.api_key),
                max_requests_per_minute=max_requests_per_minute,
                max_tokens_per_minute=float(120000),
                token_encoding_name=self.model,
                max_attempts=int(max_attempts),
                logging_level=int(30)
            )
        )

        with open(responses_file_path, "r") as responses_temp_file:
            lines = responses_temp_file.readlines()


        input_toks = 0
        output_toks = 0

        results = []
        for line in lines:
            response_object = json.loads(line)

            context = response_object[0]["contents"][0].get("parts", [{}])[0].get("text")
            response = response_object[1]["candidates"][0].get("content", {"parts": [{"text": ""}]})["parts"][0]["text"]
            idx = response_object[2]["idx"]

            input_toks += int(response_object[1]["usageMetadata"].get("promptTokenCount", 0))
            output_toks += int(response_object[1]["usageMetadata"].get("candidatesTokenCount", 0))

            results.append((idx, response))

            self.cache_hook.add_partial(
                "generate_until", (context, {"until": until}), response
            )
            pbar.update(1)

        results.sort(key=lambda x: x[0])
        pbar.close()
        clean_up_requests()
        final_responses = [s for idx, s in results]


        cwd = os.getcwd()

        if self.token_counter.exists:
            with open(self.token_counter, "r+") as f:
                token_counter = json.load(f)
                total_input_toks = int(token_counter["input_toks"]) + input_toks
                total_output_toks = int(token_counter["output_toks"]) + output_toks
                print(total_input_toks, total_output_toks)
                f.seek(0)
                f.truncate()
                f.write(json.dumps({"input_toks": total_input_toks, "output_toks": total_output_toks}))


        return final_responses
    

    def loglikelihood(self, requests):
        raise NotImplementedError("No support for logits.")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("No support for logits.")