from abc import ABC, abstractmethod

from mulcrr.utils import read_json

class Tool(ABC):
    def __init__(self, config_path: str, *args, **kwargs) -> None:
        self.config = read_json(config_path)
    
    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError("reset method not implemented")
