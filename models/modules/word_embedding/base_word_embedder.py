from abc import abstractmethod

import torch
from torch import nn


class BaseWordEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, ids: torch.tensor, mask: torch.tensor) -> torch.tensor:
        pass
