import sys
from typing import Dict, Any, Tuple, Optional
from abc import abstractmethod
from easydict import EasyDict


# specifies a dictionary of engines
_GENERATORS: Dict[str, Any] = {}  # registry


def register_generator(name):
    """Decorator used to register a generator
    Args:
        name: Name of the engine type to register
    """

    def register_class(cls, name):
        _GENERATORS[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls


@register_generator
class BaseGenerator:
    def __init__(self, config: EasyDict):
        self.config = config

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass
