import copy
import logging
from time import sleep
from typing import Any

import backoff
import regex

from ._find import Find
from ._gen import Gen
from ._select import Select
from .backend import PathFinder
from .model import Model
from .templates import LLAMA_CHAT_TEMPLATE
from .trie import MarisaTrie, Trie


def can_be_int(s):
    try:
        int(s)  # Try converting `s` to int
        return True
    except ValueError:
        return False  # Return False if a ValueError is raised


class ModelAPI(PathFinder):
    token_in = 0
    token_out = 0

    def __init__(self, model_name, seed, api_assistant=True) -> None:
        super().__init__(model_name)

        self.temperature = 0.0
        self.top_p = 1.0
        self.max_tokens = 1000
        self.seed = seed

        self.prefix_text = ""
        self.api_assistant = api_assistant

    def _current_prompt(self):
        if isinstance(self.chat, list):
            prompt_render = str(self.chat)
        else:
            prompt_render = self.chat
        return prompt_render

    def _consume_assistant_text(self, value):
        self.prefix_text += value
        match = regex.match(
            r"(.*?)" + regex.escape(value) + r"(.*?)",
            self.text_to_consume,
            regex.DOTALL,
        )
        if match:
            self.text_to_consume = self.text_to_consume[len(match.group()) :]
            self.prefix_text = ""
        else:
            self.text_to_consume = ""

    def _get_gen(self, value: Gen):
        self.temperature = value.temperature
        self.top_p = value.top_p
        self.max_tokens = value.max_tokens
        if value.stop_regex is None:
            r = r"(.*?)"
        else:
            r = rf"(.*?)({value.stop_regex})"

        return self.run(self, r, value.name, True, value.save_stop_text)

    def _get_find(self, value: Find):
        self.temperature = value.temperature
        self.top_p = value.top_p
        return self.run_find(self, value.regex, value.name)

    def _get_select(self, value: Select):
        if all(can_be_int(x) for x in value.options):
            r = r"(\d+)"
        else:
            r = r"("
            r += r"|".join([regex.escape(o) for o in value.options])
            r += r")"
        return self.run(self, r, value.name, False, False)

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        raise NotImplementedError

    def run_find(self, lm, r, name):
        if lm.text_to_consume == "":
            tmp_chat = (
                lm.chat[:-1]
                if lm.chat[-1]["role"] == "assistant" and lm.chat[-1]["content"] == ""
                else lm.chat
            )
            if self.api_assistant:
                lm.text_to_consume = self.request_api(
                    tmp_chat, lm.temperature, lm.top_p, lm.max_tokens
                )
            else:
                tmp_chat = (
                    tmp_chat[:-1] if tmp_chat[-1]["role"] == "assistant" else tmp_chat
                )
                lm.text_to_consume = self.request_api(
                    tmp_chat, lm.temperature, lm.top_p, lm.max_tokens
                )
                match = regex.match(
                    regex.escape(lm.prefix_text) + r"(.*?)",
                    lm.text_to_consume,
                    regex.DOTALL,
                )
                if match:
                    lm.text_to_consume = lm.text_to_consume[len(match.group()) :]
                    lm.prefix_text = ""

        original_res = lm.text_to_consume
        match = regex.search(r, lm.text_to_consume)
        if match:
            res = match.group(0)
            lm._variables[name] = res
            return res, original_res
        else:
            raise Exception(f"Regex {r} not found in {lm.text_to_consume}")

    def run(self, lm, r, name, is_gen, save_stop_text):
        if lm.text_to_consume == "":
            tmp_chat = (
                lm.chat[:-1]
                if lm.chat[-1]["role"] == "assistant" and lm.chat[-1]["content"] == ""
                else lm.chat
            )
            if self.api_assistant:
                lm.text_to_consume = self.request_api(
                    tmp_chat, lm.temperature, lm.top_p, lm.max_tokens
                )
            else:
                tmp_chat = (
                    tmp_chat[:-1] if tmp_chat[-1]["role"] == "assistant" else tmp_chat
                )
                lm.text_to_consume = self.request_api(
                    tmp_chat, lm.temperature, lm.top_p, lm.max_tokens
                )
                match = regex.match(
                    regex.escape(lm.prefix_text) + r"(.*?)",
                    lm.text_to_consume,
                    regex.DOTALL,
                )
                if match:
                    lm.text_to_consume = lm.text_to_consume[len(match.group()) :]
                    lm.prefix_text = ""

            # remove any prefix, if any
            p = lm.chat[-1]["content"].strip()
            if lm.text_to_consume.startswith(p):
                lm.text_to_consume = lm.text_to_consume[len(p) :]

        if regex.search(r, lm.text_to_consume):
            match = regex.match(r + r"(.*?)", lm.text_to_consume, regex.DOTALL)
            if match:
                # complete match
                match_res = match.group()
                if save_stop_text:
                    res = match.group()
                    lm.text_to_consume = lm.text_to_consume[len(match_res) :]
                else:
                    res = match.group(1)
                    lm.text_to_consume = lm.text_to_consume[len(match.group(1)) :]
            else:
                match = regex.findall(r, lm.text_to_consume, regex.DOTALL)[0]
                lm.text_to_consume = ""  # reset since this was a search of the response
                res = match
        elif is_gen:
            # not stop token
            res = lm.text_to_consume
            lm.text_to_consume = ""
        else:
            raise Exception(f"Cant find {r} in {lm.text_to_consume}")
        return res


class OpenAIAPI(ModelAPI):
    def __init__(self, model_name, seed):
        super().__init__(model_name, seed)
        from openai import OpenAI

        self.client = OpenAI()

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        import openai

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return self.client.chat.completions.create(**kwargs)

        out = completions_with_backoff(
            model=self.model_name,
            messages=chat,
            temperature=tmeperature,
            top_p=top_p,
            seed=self.seed,
            max_tokens=max_tokens,
        )
        logging.info(f"OpenAI system_fingerprint: {out.system_fingerprint}")
        return out.choices[0].message.content


class OpenRouter(ModelAPI):
    def __init__(self, model_name, seed):
        super().__init__(model_name, seed)
        from os import getenv

        from openai import OpenAI

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=getenv("OPENROUTER_API_KEY"),
        )

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        import openai

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return self.client.chat.completions.create(**kwargs)

        out = completions_with_backoff(
            model=self.model_name,
            messages=chat,
            temperature=tmeperature,
            top_p=top_p,
            seed=self.seed,
            max_tokens=max_tokens,
        )
        logging.info(f"OpenAI system_fingerprint: {out.system_fingerprint}")
        return out.choices[0].message.content


import json
import os


def append_token_usage(token_in, token_out, model, file_name):

    # Create a dictionary to append
    file_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "api_usage"
    )
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, f"{file_name}.json")

    if "gpt-4-turbo-2024-04-09" in model:
        cost_in = token_in * 10 / 1e6
        cost_out = token_out * 30 / 1e6
    else:
        raise ValueError(f"Model {model} not supported")

    new_entry = {
        "token_in": token_in,
        "token_out": token_out,
        "cost_in": cost_in,
        "cost_out": cost_out,
        "total_cost": cost_in + cost_out,
    }

    # Check if file exists and is not empty
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        # Open the file in read/write mode
        with open(file_path, "r+") as file:
            file.seek(0, os.SEEK_END)  # Move the cursor to the end of file
            file.seek(
                file.tell() - 1, os.SEEK_SET
            )  # Move the cursor back by one character from the end
            file.write(
                ",\n" + json.dumps(new_entry) + "\n]"
            )  # Convert dict to JSON string and append
    else:
        # If the file does not exist or is empty, write a new list with one element
        with open(file_path, "w") as file:
            json.dump([new_entry], file, indent=4)


import uuid


class AzureOpenAIAPI(ModelAPI):
    def __init__(self, model_name, seed):
        super().__init__(model_name, seed)
        from openai import AzureOpenAI

        self.client = AzureOpenAI()
        self.random_name = str(uuid.uuid4())

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        import openai

        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return self.client.chat.completions.create(**kwargs)

        out = completions_with_backoff(
            model=self.model_name,
            messages=chat,
            temperature=tmeperature,
            top_p=top_p,
            seed=self.seed,
            max_tokens=max_tokens,
        )
        logging.info(f"OpenAI system_fingerprint: {out.system_fingerprint}")

        self.token_in = out.usage.prompt_tokens
        self.token_out = out.usage.completion_tokens

        append_token_usage(
            self.token_in, self.token_out, self.model_name, self.random_name
        )

        return out.choices[0].message.content


import os


class MistralAPI(ModelAPI):
    def __init__(self, model_name, seed):
        super().__init__(model_name, seed, api_assistant=False)
        from httpx import Client as HTTPClient
        from httpx import HTTPTransport
        from mistralai.client import MistralClient

        api_key = os.environ["MISTRAL_API_KEY"]
        self.client = MistralClient(api_key=api_key)

        http_proxies = [
            proxy
            for varname, proxy in os.environ.items()
            if varname.lower() == "http_proxy"
        ]
        https_proxies = [
            proxy
            for varname, proxy in os.environ.items()
            if varname.lower() == "https_proxy"
        ]
        all_proxies = [
            proxy
            for varname, proxy in os.environ.items()
            if varname.lower() == "all_proxy"
        ]
        proxies = {
            "http://": http_proxies[0] if len(http_proxies) > 0 else None,
            "https://": https_proxies[0] if len(https_proxies) > 0 else None,
            "all://": all_proxies[0] if len(all_proxies) > 0 else None,
        }

        self.client._client = HTTPClient(
            proxies=proxies,
            follow_redirects=True,
            timeout=self.client._timeout,
            transport=HTTPTransport(retries=self.client._max_retries),
        )

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        from mistralai.exceptions import MistralException

        @backoff.on_exception(backoff.expo, MistralException)
        def completions_with_backoff(**kwargs):
            return self.client.chat(**kwargs)

        from mistralai.models.chat_completion import ChatMessage

        if chat[-1]["role"] == "assistant":
            raise Exception(
                "Assistant should not be the last role in the chat for Mistral."
            )

        chat_mistral = [
            ChatMessage(role=entry["role"], content=entry["content"]) for entry in chat
        ]
        out = completions_with_backoff(
            model=self.model_name,
            messages=chat_mistral,
            temperature=tmeperature,
            top_p=top_p,
            random_seed=self.seed,
            max_tokens=max_tokens,
        )
        return out.choices[0].message.content


class AnthropicAPI(ModelAPI):
    def __init__(self, model_name, seed):
        super().__init__(model_name, seed, api_assistant=False)
        from anthropic import Anthropic

        self.client = Anthropic()

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        from anthropic._exceptions import APIStatusError

        @backoff.on_exception(backoff.expo, APIStatusError)
        def completions_with_backoff(**kwargs):
            return self.client.messages.create(**kwargs)

        if chat[-1]["role"] == "assistant":
            raise Exception(
                "Assistant should not be the last role in the chat for Anthropic."
            )

        out = completions_with_backoff(
            model=self.model_name,
            messages=chat,
            temperature=tmeperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        return out.content[0].text
