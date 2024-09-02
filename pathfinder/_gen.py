class Gen:
    def __init__(
        self,
        name,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        stop_regex,
        save_stop_text,
    ) -> None:
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.stop_regex = stop_regex
        self.save_stop_text = save_stop_text

    def __repr__(self) -> str:
        return (
            f"gen({self.name}, {self.max_tokens}, {self.temperature}, {self.top_p},"
            f" {self.repetition_penalty}, {self.stop_regex}, {self.save_stop_text})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def gen(
    name: str = "gen",
    max_tokens=100,
    temperature=0.0,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_regex=[],
    save_stop_text=False,
):
    return Gen(
        name,
        max_tokens,
        temperature,
        top_p,
        repetition_penalty,
        stop_regex,
        save_stop_text,
    )
