from .backend import PathFinder


class ContextBlock:
    def __init__(
        self,
        role,
    ):
        self.role = role
        self.init_tag = True

    def __enter__(self):
        if PathFinder.open_block is not None:
            raise Exception("Cannot open a block inside another block")
        PathFinder.open_block = self
        PathFinder.empty_block = True

    def __exit__(self, exc_type, exc_value, traceback):
        PathFinder.open_block = None


def system():
    return ContextBlock("system")


def user():
    return ContextBlock("user")


def assistant():
    return ContextBlock("assistant")
