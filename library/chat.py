from transformers import PreTrainedModel, PreTrainedTokenizer

from .model import Model
from .templates import LLAMA_CHAT_TEMPLATE, MIXTRAL_TEMPLATE


class LlamaChat(Model):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(model, tokenizer)
        self.template = LLAMA_CHAT_TEMPLATE


class Mixtral(Model):
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(model, tokenizer)
        self.template = MIXTRAL_TEMPLATE
