from .base_generator import BaseGenerator, _GENERATORS
from .api_generator import APIGenerator


def get_generator(name: str) -> BaseGenerator:
    """
    Return constructor for specified generator
    """
    name = "".join(name.lower().split("_"))
    if name in _GENERATORS:
        return _GENERATORS[name]
    else:
        raise Exception("Error: Trying to access a generator that has not been registered")
