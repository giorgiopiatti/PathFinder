import copy
from time import sleep
from typing import Any

import backoff
import numpy as np
import openai
import pygtrie
import regex
import torch
from openai import OpenAI
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

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model = DummyModel(DummyConfig(model_name))

        self._variables = {}
        self._variables_log_probs = {}

        self.chat = []
        self.text_to_consume = ""
        self.temperature = 0.7
        self.top_p = 1.0

        self.client = OpenAI()

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

    def run_find(self, lm, r, name):
        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return self.client.chat.completions.create(**kwargs)

        if lm.text_to_consume == "":
            tmp_chat = (
                lm.chat[:-1]
                if lm.chat[-1]["role"] == "assistant" and lm.chat[-1]["content"] == ""
                else lm.chat
            )
            out = completions_with_backoff(
                model=self.model_name,
                messages=tmp_chat,
                temperature=lm.temperature,
                top_p=lm.top_p,
            )
            lm.text_to_consume = out.choices[0].message.content

        lm._variables[f"PATHFINDER_ORIGINAL_{name}"] = lm.text_to_consume
        lm.chat[-1]["content"] += lm.text_to_consume

        match = regex.search(r, lm.text_to_consume)
        if match:
            res = match.group(0)
            lm._variables[name] = res
        else:
            raise Exception(f"Regex {r} not found in {lm.text_to_consume}")

    def run(self, lm, r, name, is_gen, save_stop_text):
        @backoff.on_exception(backoff.expo, openai.RateLimitError)
        def completions_with_backoff(**kwargs):
            return self.client.chat.completions.create(**kwargs)

        if lm.text_to_consume == "":

            tmp_chat = (
                lm.chat[:-1]
                if lm.chat[-1]["role"] == "assistant" and lm.chat[-1]["content"] == ""
                else lm.chat
            )
            out = completions_with_backoff(
                model=self.model_name,
                messages=tmp_chat,
                temperature=lm.temperature,
                top_p=lm.top_p,
            )
            lm.text_to_consume = out.choices[0].message.content
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
