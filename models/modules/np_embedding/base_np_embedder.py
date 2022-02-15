from abc import abstractmethod

import torch

from models.modules.base_module import BaseModule


class BaseNPEmbedder(BaseModule):
    @abstractmethod
    def forward(self, word_embeddings: torch.tensor, nps: torch.tensor) -> torch.tensor:
        pass
