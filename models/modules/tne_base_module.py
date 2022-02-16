from abc import abstractmethod, ABC

from torch import nn


class TNEBaseModule(nn.Module, ABC):
    def __init__(self, input_size: int):
        super().__init__()
        self.input_size = input_size

    @abstractmethod
    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        pass

    @property
    @abstractmethod
    def output_size(self):
        pass
