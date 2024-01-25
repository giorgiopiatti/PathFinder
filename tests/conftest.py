import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from pathfinder.library import LlamaChat, PathFinderModel


def pytest_addoption(parser):
    parser.addoption(
        "--LLM",
        type=str,
        default="TheBloke/Llama-2-7B-GPTQ",
    )
    parser.addoption("--temperature", type=float, default=0.0)


@pytest.fixture(scope="session")
def arg_LLM(request):
    return request.config.getoption("--LLM")


@pytest.fixture(scope="session")
def arg_temperature(request):
    return request.config.getoption("--temperature")


@pytest.fixture(scope="session")
def base_llm(arg_LLM):
    model = AutoModelForCausalLM.from_pretrained(
        arg_LLM,
        device_map="auto",
    )
    return model


@pytest.fixture(scope="session")
def base_tokenizer(arg_LLM):
    tokenizer = AutoTokenizer.from_pretrained(arg_LLM, use_fast=True)
    return tokenizer


@pytest.fixture()
def llm(base_llm, base_tokenizer):
    return LlamaChat(base_llm, base_tokenizer)
