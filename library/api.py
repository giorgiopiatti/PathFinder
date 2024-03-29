import copy
import logging
from time import sleep
from typing import Any

import backoff
import numpy as np
import openai
import pygtrie
import regex
import torch
from transformers import (
    AutoConfig,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
    MaxLengthCriteria,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

from ._find import Find
from ._gen import Gen
from ._select import Select
from .model import Model
from .templates import LLAMA_CHAT_TEMPLATE
from .trie import MarisaTrie, Trie


class DummyConfig:
    name_or_path: str

    def __init__(self, name_or_path: str) -> None:
        self.name_or_path = name_or_path


class DummyModel:
    config: DummyConfig

    def __init__(self, config: DummyConfig) -> None:
        self.config = config


def can_be_int(s):
    try:
        int(s)  # Try converting `s` to int
        return True
    except ValueError:
        return False  # Return False if a ValueError is raised


class ModelAPI:
    token_in = 0
    token_out = 0

    def __init__(self, model_name, seed) -> None:
        self.model_name = model_name
        self.model = DummyModel(DummyConfig(model_name))

        self._variables = {}
        self._variables_log_probs = {}

        self.chat = []
        self.text_to_consume = ""
        self.temperature = 0.7
        self.top_p = 1.0
        self.seed = seed

    def _current_prompt(self):
        if isinstance(self.chat, list):
            prompt_render = str(self.chat)
        else:
            prompt_render = self.chat
        return prompt_render

    def _format_chat_entry_as_html(self, entry):
        # Bold the role tag
        role_tag = f'<strong>{entry["role"].upper()}</strong>'
        return f'<div>{role_tag}: {entry["content"]}</div>'

    def html(self):
        if isinstance(self.chat, list):
            # Process each chat entry and format it as HTML
            html_entries = [
                self._format_chat_entry_as_html(entry) for entry in self.chat
            ]
            prompt_render = "".join(html_entries)
        else:
            # Format a single chat entry as HTML
            prompt_render = self._format_chat_entry_as_html(self.chat)
        return prompt_render

    def __str__(self) -> str:
        pass

    def copy(self):
        """Create a shallow copy of the model object."""

        # start with a shallow copy
        new_lm = copy.copy(self)

        # then copy a few things we need deeper copies of
        new_lm._variables = self._variables.copy()
        new_lm._variables_log_probs = self._variables_log_probs.copy()

        if isinstance(self.chat, list):
            new_lm.chat = self.chat.copy()
        else:
            new_lm.chat = self.chat

        return new_lm

    def __add__(self, value):
        # we hav string, gen, select
        lm = self.copy()

        if len(lm.chat) == 0 and Model.empty_block and Model.open_block is None:
            # We are not in a chat block, so we simply add the string
            lm.chat = ""
        elif Model.open_block is not None and Model.open_block.init_tag:
            Model.open_block.init_tag = False
            lm.chat.append(
                {
                    "role": Model.open_block.role,
                    "content": "",
                }
            )

        if isinstance(value, str):
            if isinstance(lm.chat, list):
                lm.chat[-1]["content"] += value
            else:
                lm.chat += value

            if lm.chat[-1]["role"] == "assistant":
                match = regex.match(
                    regex.escape(value) + r"(.*?)", lm.text_to_consume, regex.DOTALL
                )
                if match:
                    lm.text_to_consume = lm.text_to_consume[len(match.group()) :]
                else:
                    lm.text_to_consume = ""
        else:
            if isinstance(lm.chat, list):
                if lm.chat[-1]["role"] != "assistant":
                    raise Exception(
                        f"{value} can be used only in assistant block, not in"
                        f" {lm.chat[-1]['role']} block!"
                    )
            # any string regex then stop token
            if isinstance(value, Gen):
                lm.temperature = value.temperature
                lm.top_p = value.top_p
                lm.max_tokens = value.max_tokens
                if value.stop_regex is None:
                    r = r"(.*?)"
                else:
                    r = rf"(.*?)({value.stop_regex})"

                self.run(lm, r, value.name, True, value.save_stop_text)
            elif isinstance(value, Find):
                lm.temperature = value.temperature
                lm.top_p = value.top_p
                self.run_find(lm, value.regex, value.name)
            elif isinstance(value, Select):
                # detect if value.options is a list of string of numbers, if so use a digit regex
                if all(can_be_int(x) for x in value.options):
                    r = r"(\d+)"
                else:
                    r = r"("
                    r += r"|".join([regex.escape(o) for o in value.options])
                    r += r")"
                self.run(lm, r, value.name, False, False)

        return lm

    def request_api(self, chat, tmeperature, top_p, max_tokens):
        raise NotImplementedError

    def run_find(self, lm, r, name):
        if lm.text_to_consume == "":
            tmp_chat = (
                lm.chat[:-1]
                if lm.chat[-1]["role"] == "assistant" and lm.chat[-1]["content"] == ""
                else lm.chat
            )
            lm.text_to_consume = self.request_api(
                tmp_chat, lm.temperature, lm.top_p, lm.max_tokens
            )

        lm._variables[f"PATHFINDER_ORIGINAL_{name}"] = lm.text_to_consume
        lm.chat[-1]["content"] += lm.text_to_consume

        match = regex.search(r, lm.text_to_consume)
        if match:
            res = match.group(0)
            lm._variables[name] = res
        else:
            raise Exception(f"Regex {r} not found in {lm.text_to_consume}")

    def run(self, lm, r, name, is_gen, save_stop_text):
        if lm.text_to_consume == "":
            tmp_chat = (
                lm.chat[:-1]
                if lm.chat[-1]["role"] == "assistant" and lm.chat[-1]["content"] == ""
                else lm.chat
            )
            lm.text_to_consume = self.request_api(
                tmp_chat, lm.temperature, lm.top_p, lm.max_tokens
            )
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
                    lm.chat[-1]["content"] += match.group()
                    lm._variables[name] = match.group()
                    lm.text_to_consume = lm.text_to_consume[len(match_res) :]
                else:
                    lm.chat[-1]["content"] += match.group(1)
                    lm._variables[name] = match.group(1)
                    lm.text_to_consume = lm.text_to_consume[len(match.group(1)) :]
            else:
                match = regex.findall(r, lm.text_to_consume, regex.DOTALL)[0]
                lm.text_to_consume = ""  # reset since this was a search of the response
                lm._variables[name] = match
                lm.chat[-1]["content"] += match
        elif is_gen:
            # not stop token
            lm.chat[-1]["content"] += lm.text_to_consume
            lm._variables[name] = lm.text_to_consume
            lm.text_to_consume = ""
        else:
            raise Exception(f"Cant find {r} in {lm.text_to_consume}")

    def __getitem__(self, key):
        if key in self._variables:
            return self._variables[key]

    def set(self, key, value):
        lm = self.copy()
        lm._variables[key] = value
        return lm


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


import os


class MistralAPI(ModelAPI):
    def __init__(self, model_name, seed):
        super().__init__(model_name, seed)
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
        super().__init__(model_name, seed)
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
