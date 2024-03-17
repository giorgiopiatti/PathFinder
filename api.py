import pytest

from library import LlamaChat, ModelAPI, assistant, gen, select, user

base = ModelAPI("gpt-3.5-turbo")

lm = base
with user():
    lm += "Write a list"
with assistant():
    lm += "1)"
    lm += gen(max_tokens=100, stop_regex=r"4\)", name="list")
lm.run()
assert not "4)" in str(lm["list"])

lm = base
with user():
    lm += "Do you like apples or oranges?"
with assistant():
    lm += "I like "
    lm += select(["apples", "oranges"], name="fruit")
assert lm["fruit"] == "apples" or lm["fruit"] == "oranges"
