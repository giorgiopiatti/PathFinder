class Select:
    def __init__(
        self,
        options,
        regex_select,
        name,
    ):
        assert regex_select is not None or options is not None
        self.options = options
        self.regex_select = regex_select
        self.name = name

    def __repr__(self) -> str:
        return f"select({self.options}, {self.name})"

    def __str__(self) -> str:
        return self.__repr__()


def select(
    options=None,
    regex_select=None,
    name="select",
):
    return Select(
        options,
        regex_select,
        name,
    )
