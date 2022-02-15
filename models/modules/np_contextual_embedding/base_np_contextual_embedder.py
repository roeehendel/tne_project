from abc import abstractmethod

import torch
from torch import nn


class BaseNPContextualEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, np_embeddings: torch.tensor) -> torch.tensor:
        pass
