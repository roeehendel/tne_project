from abc import abstractmethod

import torch
from torch import nn

from models.modules.base_module import BaseModule


class BaseAnchorComplementEmbedder(BaseModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, np_embeddings: torch.tensor) -> torch.tensor:
        pass
