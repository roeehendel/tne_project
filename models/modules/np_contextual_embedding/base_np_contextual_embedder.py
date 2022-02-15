from abc import abstractmethod

import torch

from models.modules.base_module import BaseModule


class BaseNPContextualEmbedder(BaseModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, np_embeddings: torch.tensor, token_embedding: torch.tensor, num_nps) -> torch.tensor:
        pass
