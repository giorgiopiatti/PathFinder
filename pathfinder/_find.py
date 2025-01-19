class Find:
    def __init__(
        self,
        name,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        regex,
        stop_regex,
    ) -> None:
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.regex = regex
        self.stop_regex = stop_regex

    def __repr__(self) -> str:
        return (
            f"gen({self.name}, {self.max_tokens}, {self.temperature}, {self.top_p},"
            f" {self.repetition_penalty}, {self.regex}, {self.stop_regex})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def find(
    name: str = "find",
    max_tokens=100,
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.0,
    regex=r".*",
    stop_regex=None,
):
    return Find(
        name, max_tokens, temperature, top_p, repetition_penalty, regex, stop_regex
    )
