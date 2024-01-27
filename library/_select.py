class Select:
    def __init__(
        self,
        options,
        name,
    ):
        self.options = options
        self.name = name

    def __repr__(self) -> str:
        return f"select({self.options}, {self.name})"

    def __str__(self) -> str:
        return self.__repr__()


def select(
    options=None,
    name="select",
):
    return Select(
        options,
        name,
    )
