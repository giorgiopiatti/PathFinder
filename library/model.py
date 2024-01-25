import copy
from typing import Any

import torch
from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
    generate_enforced,
)
from transformers import (
    AutoConfig,
    GenerationConfig,
    MaxLengthCriteria,
    PreTrainedModel,
    PreTrainedTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    pipeline,
)

from ._gen import Gen
from ._select import Select
from .templates import LLAMA_CHAT_TEMPLATE


class StopAtSequenceCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer, stop_sequence: str):
        self.tokenizer = tokenizer
        self.stop_token_ids = tokenizer.encode(stop_sequence, add_special_tokens=False)
        self.stop_token_ids = [
            token for token in self.stop_token_ids if tokenizer.decode(token) != ""
        ]  # remove tokens that decode to empty string

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        # Get the length of the stop sequence
        len_stop_sequence = len(self.stop_token_ids)

        # Check if the end of input_ids matches the stop sequence
        if input_ids.shape[1] >= len_stop_sequence:
            return all(
                input_ids[0, -len_stop_sequence:]
                == torch.tensor(self.stop_token_ids, device=input_ids.device)
            )
        else:
            return False


class Model:
    open_block = None
    empty_block = True

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        self.model = model
        self.template = None
        self.tokenizer = tokenizer
        # these are the state variables stored with the model
        self._variables = {}
        self._variables_log_probs = {}

        self.chat = []

    def _current_prompt(self):
        if isinstance(self.chat, list):
            prompt_render = self.tokenizer.apply_chat_template(
                self.chat,
                tokenize=False,
                add_generation_prompt=self.chat[-1]["role"] != "assistant",
                chat_template=self.template,
            )
        else:
            prompt_render = self.chat
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
        else:

            if isinstance(lm.chat, list):
                if lm.chat[-1]["role"] != "assistant":
                    raise Exception(
                        f"{value} can be used only in assistant block, not in"
                        f" {lm.chat[-1]['role']} block!"
                    )
                prompt_render = self.tokenizer.apply_chat_template(
                    lm.chat,
                    tokenize=False,
                    add_generation_prompt=lm.chat[-1]["role"] != "assistant",
                    chat_template=self.template,
                )
            else:
                prompt_render = lm.chat

            # Tokenize
            input_ids = self.tokenizer(
                prompt_render, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.model.device)

            # Run specific generation
            if isinstance(value, Gen):
                model_config = AutoConfig.from_pretrained(self.model.name_or_path)

                generation_config = GenerationConfig(
                    max_new_tokens=value.max_tokens,
                    **(
                        {
                            "temperature": value.temperature,
                            "do_sample": True,
                            "top_p": value.top_p,
                        }
                        if value.temperature != 0.0
                        else {}
                    ),
                )
                model_config.update(generation_config.to_dict())

                output = self.model.generate(
                    inputs=input_ids,
                    generation_config=generation_config,
                    stopping_criteria=StoppingCriteriaList(
                        [
                            StopAtSequenceCriteria(self.tokenizer, pattern)
                            for pattern in value.stop_patterns
                        ],
                    ),
                )

                res = self.tokenizer.decode(output[0], skip_special_tokens=True)[
                    len(prompt_render) :
                ]
                # remove end pattern if it exists and save_stop_text is True
                if not value.save_stop_text:
                    for pattern in value.stop_patterns:
                        if res.endswith(pattern):
                            res = res[: -len(pattern)]
                            break
            elif isinstance(value, Select):
                model_config = AutoConfig.from_pretrained(self.model.name_or_path)
                generation_config = GenerationConfig(max_length=4096)
                model_config.update(generation_config.to_dict())

                # hf_pipeline = pipeline(
                #     "text-generation",
                #     model=self.model,
                #     tokenizer=self.tokenizer,
                #     generation_config=model_config,
                # )
                regex = ""
                for i, o in enumerate(value.options):
                    if i == 0:
                        regex += o
                    else:
                        regex += "|" + o

                parser = RegexParser(regex)
                prefix_function = build_transformers_prefix_allowed_tokens_fn(
                    self.tokenizer, parser
                )

                output = self.model.generate(
                    inputs=input_ids,
                    generation_config=generation_config,
                    prefix_allowed_tokens_fn=prefix_function,
                )
                res = self.tokenizer.decode(output[0], skip_special_tokens=True)[
                    len(prompt_render) :
                ]
            else:
                raise Exception("Invalid state")
            # Save the result
            lm._variables[value.name] = res
            if isinstance(lm.chat, list):
                lm.chat[-1]["content"] += res
            else:
                lm.chat += res

        return lm

    def __getitem__(self, key):
        if key in self._variables:
            return self._variables[key]
