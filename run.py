import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer

from library import LlamaChat, assistant, gen, select, user

arg_LLM = "TheBloke/Llama-2-7B-GPTQ"
model = AutoModelForCausalLM.from_pretrained(
    arg_LLM,
    device_map="auto",
)


tokenizer = AutoTokenizer.from_pretrained(arg_LLM, use_fast=True)

base = LlamaChat(model, tokenizer)

lm = base
with user():
    lm += "Write a list"
with assistant():
    lm += "1)"
    lm += gen(max_tokens=100, stop_regex=r"4\)", name="list")
lm += "Write a list"
lm += "1)"
lm += gen(max_tokens=100, stop_regex=["4)"], name="list")
assert not "4)" in str(lm["list"])

lm = base
with user():
    lm += "Do you like apples or oranges?"
with assistant():
    lm += "I like "
    lm += select(["apples", "oranges"], name="fruit")
assert lm["fruit"] == "apples" or lm["fruit"] == "oranges"
