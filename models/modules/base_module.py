from abc import abstractmethod, ABC

from torch import nn


class BaseModule(nn.Module, ABC):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size

    @property
    @abstractmethod
    def output_size(self):
        pass
