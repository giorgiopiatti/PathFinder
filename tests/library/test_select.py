from pathfinder import PathFinderModel, assistant, gen, select, user


def test_select_tags(llm):
    lm = llm
    with user():
        lm += "Do you like apples or oranges?"
    with assistant():
        lm += "I like "
        lm += select(["apples", "oranges"], name="fruit")
    assert lm["fruit"] == "apples" or lm["fruit"] == "oranges"
