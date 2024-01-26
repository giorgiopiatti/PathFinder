from pathfinder import PathFinderModel, assistant, gen, user


def test_gen_basic(llm):
    lm = llm
    lm += "Generate a poem: "
    lm += gen(max_tokens=100)


def test_gen_stop_pattern(llm):
    lm = llm
    with user():
        lm += "Write a list"
    with assistant():
        lm += "1)"
        lm += gen(max_tokens=100, stop_regex=r"4\)", name="list")
    assert not "4)" in str(lm["list"])


def test_gen_stop_pattern_save(llm):
    lm = llm
    with user():
        lm += "Write a list"
    with assistant():
        lm += "1)"
        lm += gen(max_tokens=100, stop_regex=r"4\)", name="list", save_stop_text=True)
    assert "4)" in str(lm["list"])


def test_gen_tags(llm):
    lm = llm
    with user():
        lm += "Write a list"
    with assistant():
        lm += "1)"
        lm += gen(max_tokens=50, name="list")
