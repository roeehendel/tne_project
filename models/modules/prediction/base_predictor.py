from abc import abstractmethod

import torch

from models.modules.base_module import BaseModule


class BasePredictor(BaseModule):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, anchor_complement_embeddings: torch.tensor) -> torch.tensor:
        pass
