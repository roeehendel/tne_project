from abc import abstractmethod

import torch

from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder


class EncoderNPContextualEmbedder(BaseNPContextualEmbedder):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, np_embeddings: torch.tensor) -> torch.tensor:
        pass
