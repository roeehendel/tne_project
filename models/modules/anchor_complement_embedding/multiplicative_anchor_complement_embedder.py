import torch
from torch import nn

from models.modules.anchor_complement_embedding.base_anchor_complement_embedder import BaseAnchorComplementEmbedder


class MultiplicativeAnchorComplementEmbedder(BaseAnchorComplementEmbedder):
    def __init__(self, input_size: int, hidden_size: int = 512):
        super().__init__(input_size)
        self._hidden_size = hidden_size

        self._anchor_encoder = nn.Linear(self.input_size, self._hidden_size)
        self._complement_encoder = nn.Linear(self.input_size, self._hidden_size)
        self._projection = nn.Linear(3 * self._hidden_size, self._hidden_size)

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        anchor_complement_embeddings_pairs = self._get_anchor_complement_embeddings_pairs(intermediate_outputs)
        batch_size, num_anchor_complement_pairs, _, _ = anchor_complement_embeddings_pairs.shape

        # anchor_complement_distances = torch.norm(anchor_complement_embeddings_concat[:, :, :, 0] -
        #                                          anchor_complement_embeddings_concat[:, :, :, 1], dim=-1)
        elementwise_product = anchor_complement_embeddings_pairs.prod(axis=-1)
        concat_and_product = torch.cat([anchor_complement_embeddings_pairs, elementwise_product.unsqueeze(-1)],
                                       dim=-1)

        concat_and_product_flat = concat_and_product.view(batch_size, num_anchor_complement_pairs, -1)

        # all_features = torch.cat([concat_and_product_flat, anchor_complement_distances.unsqueeze(-1)], dim=-1)
        # anchor_complement_embeddings = self._projection(all_features)

        anchor_complement_embeddings = self._projection(concat_and_product_flat)

        return dict(embeddings=anchor_complement_embeddings)
