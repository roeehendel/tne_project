from abc import abstractmethod

import torch

from models.modules.base_module import BaseModule


class BaseWordEmbedder(BaseModule):
    @property
    @abstractmethod
    def tokenizer(self):
        pass

    @abstractmethod
    def forward(self, ids: torch.tensor, mask: torch.tensor) -> torch.tensor:
        pass
