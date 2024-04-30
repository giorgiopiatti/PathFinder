import os

from transformers import PreTrainedModel, PreTrainedTokenizer

from .model import Model
from .templates import LLAMA_CHAT_TEMPLATE, MIXTRAL_INSTRUCT_TEMPLATE


class LlamaChat:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/llama-2-chat.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class Llama3Chat:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/llama-3-chat.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class MixtralInstruct:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/mistral-instruct.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class Vicuna:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/vicuna.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class ChatML:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/chatml.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class Phi3:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/phi-3.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class DeepSeek:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/deep_seek.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class MetaMath:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/alpaca.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class MistralInstruct:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/mistral-instruct.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template


class Cohere:
    def __init__(self):
        template_path = os.path.join(
            os.path.dirname(__file__), "./templates_jinja/cohere.jinja"
        )
        chat_template = open(template_path).read()
        chat_template = chat_template.replace("    ", "").replace("\n", "")
        self.template = chat_template
