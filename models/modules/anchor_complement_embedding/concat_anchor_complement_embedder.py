import torch
from torch import nn

from models.modules.anchor_complement_embedding.base_anchor_complement_embedder import BaseAnchorComplementEmbedder


class ConcatAnchorComplementEmbedder(BaseAnchorComplementEmbedder):
    def __init__(self, np_contextual_embeddings_size: int, hidden_size: int):
        super().__init__()
        self._np_embeddings_size = np_contextual_embeddings_size
        self._hidden_size = hidden_size

        self._anchor_encoder = self._anchor_complement_encoder()
        self._complement_encoder = self._anchor_complement_encoder()

    @property
    def output_size(self):
        return self._hidden_size * 2

    def forward(self, np_embeddings: torch.tensor) -> torch.tensor:
        batch_size, num_nps, _ = np_embeddings.shape

        anchor_embeddings = self._anchor_encoder(np_embeddings)
        complement_embeddings = self._complement_encoder(np_embeddings)

        anchor_complement_embeddings = torch.cat(
            [anchor_embeddings[:, None, :, :, None].repeat(1, num_nps, 1, 1, 1),
             complement_embeddings[:, :, None, :, None].repeat(1, 1, num_nps, 1, 1)],
            dim=-1
        ).reshape(batch_size, num_nps ** 2, -1)

        return anchor_complement_embeddings

    def _anchor_complement_encoder(self):
        return nn.Linear(self._np_embeddings_size, self._hidden_size)
