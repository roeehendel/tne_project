from abc import ABC

import torch
from torch import nn

from models.modules.tne_base_module import TNEBaseModule


class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs


class BaseAnchorComplementEmbedder(TNEBaseModule, ABC):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self._hidden_size = input_size

        self._anchor_encoder = nn.Linear(self.input_size, self._hidden_size)
        self._complement_encoder = nn.Linear(self.input_size, self._hidden_size)

    def _get_anchor_or_complement_encoder(self):
        return nn.Sequential(
            nn.Linear(self.input_size, self._hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            ResNet(nn.Linear(self._hidden_size, self._hidden_size))
        )

    def _get_anchor_complement_embeddings_pairs(self, intermediate_outputs):
        np_contextual_embeddings = intermediate_outputs['np_contextual_embedder']['embeddings']

        batch_size, max_nps, _ = np_contextual_embeddings.shape

        anchor_embeddings = self._anchor_encoder(np_contextual_embeddings)
        complement_embeddings = self._complement_encoder(np_contextual_embeddings)

        anchor_complement_embeddings_pairs = torch.cat(
            [anchor_embeddings[:, None, :, :, None].repeat(1, max_nps, 1, 1, 1),
             complement_embeddings[:, :, None, :, None].repeat(1, 1, max_nps, 1, 1)],
            dim=-1
        ).view(batch_size, max_nps ** 2, self._hidden_size, 2)

        return anchor_complement_embeddings_pairs
