# Copyright (c) Praneeth Vadlapati

import os
import time
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from IPython.display import display, Markdown
import openai

from response_cacher import get_cache_key, get_cached_response, save_cached_response


load_dotenv()


def display_md(md_text):
    display(Markdown(md_text))


def print_progress(chr="."):
    if chr == 0 and type(chr) == int:
        return
    if type(chr) == bool:
        chr = "." if chr else ","
    print(chr, end="", flush=True)


def print_error(err=None, chr="!"):
    # print(err)
    print_progress(chr)


model = os.getenv("OPENAI_MODEL")
if model:
    model = model.strip()
    print(f"Model: {model}")
else:
    raise Exception("OPENAI_MODEL is not set in the environment variables")
model_name_short = model.split("/")[-1].lower()  # OpenAI/GPT-4 -> gpt-4

client = openai.OpenAI()


def get_response(
    messages: Union[str, List[Dict[str, str]]],
    attempt: Optional[int] = None,
    max_retries: int = 3,
    ignore_cache: bool = False,
    **kwargs: Any,
) -> Optional[str]:
    """Get response from OpenAI API with automatic caching."""
    if not messages:
        return None

    # Generate cache key
    kwargs["model"] = model_name_short  # Ensure model is part of the cache key
    cache_key = get_cache_key(messages, **kwargs)
    if attempt:  # attempt 0 doesn't need to be mentioned
        cache_key += f"_attempt{attempt}"
    if not ignore_cache and not os.getenv("IGNORE_CACHE"):
        cached_response = get_cached_response(cache_key)
        if cached_response is not None:
            return cached_response

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    for _ in range(max_retries):
        response = None
        try:
            response = client.chat.completions.create(  # type: ignore
                messages=messages,  # type: ignore
                reasoning_effort="low",
                **kwargs,
                # "model" is passed in kwargs and should not be passed again due to an error
            )
            response = response.choices[0].message.content.strip()
            if not response:
                raise Exception("Empty response from the bot")
            save_cached_response(cache_key, response)
            return response
        except openai.RateLimitError as e:
            e = str(e)
            total_wait_time = None
            if "Please retry after" in e:  # Please retry after X sec
                total_wait_time = (
                    e.split("Please retry after")[1].split("sec")[0].strip()
                )
                total_wait_time = int(total_wait_time) + 1
            elif "Please try again in" in e:
                rate_limit_time = (
                    e.split("Please try again in")[1].split(".")[0].strip()
                )
                rate_limit_time_min = 0
                rate_limit_time_sec = 0
                if "m" in rate_limit_time:
                    rate_limit_time_min = rate_limit_time.split("m")[0]
                    rate_limit_time = rate_limit_time.split("m")[1]
                if "s" in rate_limit_time:
                    rate_limit_time_sec = rate_limit_time.split("s")[0]
                total_wait_time = (
                    (int(rate_limit_time_min) * 60) + int(rate_limit_time_sec) + 1
                )
            else:
                print(e)
                total_wait_time = 20
            print_progress(f" RL Wait{total_wait_time}s ")
            time.sleep(int(total_wait_time))
        except Exception as e:
            e = str(e)
            print(e)
            if "503" in e:  # Service Unavailable
                print_progress("Unavailable Wait ")
                time.sleep(15)
            elif e == "Connection error.":
                print_progress("Server not online ")
            else:
                print_progress(f"Error Retrying ")
    raise Exception(f"No response from the bot after {max_retries} retries")


if __name__ == "__main__":
    # Test the get_response function
    test_message = "Hello, how are you?"
    print("Testing get_response function...")
    start = time.time()
    response = get_response(test_message)
    end = time.time()
    print("Response:", response)
    print(f"Time taken: {end - start:.2f} seconds")

    start2 = time.time()
    response2 = get_response(test_message)
    end2 = time.time()
    print("Response from cache:", response2)
    print(f"Time taken for cached response: {end2 - start2:.2f} seconds")