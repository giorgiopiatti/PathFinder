import copy
from time import sleep
from typing import Any

import guidance
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

from ._gen import Gen
from ._select import Select
from .model import Model
from .templates import LLAMA_CHAT_TEMPLATE
from .trie import MarisaTrie, Trie

client = OpenAI()


import backoff


@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


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
        self.regex = r""
        self.conditions = []
        self.temperature = 0.7
        self.top_p = 1.0

        self.pending_generation = False

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

        if isinstance(value, str) and not self.pending_generation:
            if isinstance(lm.chat, list):
                lm.chat[-1]["content"] += value
            else:
                lm.chat += value
        else:
            if not self.pending_generation:
                lm.regex = r""
            lm.pending_generation = True

            if isinstance(lm.chat, list):
                if lm.chat[-1]["role"] != "assistant":
                    raise Exception(
                        f"{value} can be used only in assistant block, not in"
                        f" {lm.chat[-1]['role']} block!"
                    )
            # any string regex then stop token
            if isinstance(value, Gen):
                self.temperature = value.temperature
                self.top_p = value.top_p
                if value.stop_regex is None:
                    lm.regex += r"(.*?)"
                    self.conditions.append((value.name, None))
                else:
                    lm.regex += rf"(.*?)({value.stop_regex})(.*?)"
                    self.conditions.append((value.name, value.stop_regex))

            elif isinstance(value, Select):
                # detect if value.options is a list of string of numbers, if so use a digit regex
                if all(can_be_int(x) for x in value.options):
                    lm.regex += r"(\d+)"
                else:
                    lm.regex += r"("
                    lm.regex += r"|".join(value.options)
                    lm.regex += r")"
                self.conditions.append((value.name, value.options))

        return lm

    def run(self):
        if self.pending_generation:
            out = completions_with_backoff(
                model=self.model_name,
                messages=self.chat,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            res = out.choices[0].message.content
            self.chat[-1]["content"] += res
            # given regex extract the variables
            if self.regex != "":
                match = regex.findall(self.regex, res, regex.DOTALL)[0]
                # save here

                skip_next = 0
                save_index = 0
                for i, m in enumerate(match):
                    if skip_next > 0:
                        skip_next -= 1
                        continue  # skipping stop token
                    self._variables[self.conditions[save_index][0]] = m
                    if type(self.conditions[save_index][1]) is str:
                        skip_next = 2  # skip stop token and next part " " or "\n" or similar things
                    save_index += 1
                    if i == len(self.conditions) - 1:
                        break  # skipping last part
                # what to do if do not match?
            self.pending_generation = False

    def __getitem__(self, key):
        if self.pending_generation:
            self.run()

        if key in self._variables:
            return self._variables[key]

    def set(self, key, value):
        lm = self.copy()
        lm._variables[key] = value
        return lm
