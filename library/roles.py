from .model import Model


class ContextBlock:
    def __init__(
        self,
        role,
    ):
        self.role = role
        self.init_tag = True

    def __enter__(self):
        if Model.open_block is not None:
            raise Exception("Cannot open a block inside another block")
        Model.open_block = self
        Model.empty_block = True

    def __exit__(self, exc_type, exc_value, traceback):
        Model.open_block = None


def system():
    return ContextBlock("system")


def user():
    return ContextBlock("user")


def assistant():
    return ContextBlock("assistant")
