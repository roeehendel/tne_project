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
        np_contextual_embeddings = intermediate_outputs['np_contextual_embedder']['embeddings']

        batch_size, max_nps, _ = np_contextual_embeddings.shape

        anchor_embeddings = self._anchor_encoder(np_contextual_embeddings)
        complement_embeddings = self._complement_encoder(np_contextual_embeddings)

        anchor_complement_embeddings_concat = torch.cat(
            [anchor_embeddings[:, None, :, :, None].repeat(1, max_nps, 1, 1, 1),
             complement_embeddings[:, :, None, :, None].repeat(1, 1, max_nps, 1, 1)],
            dim=-1
        ).view(batch_size, max_nps ** 2, self._hidden_size, 2)

        elementwise_product = anchor_complement_embeddings_concat.prod(axis=-1)
        concat_and_product = torch.cat([anchor_complement_embeddings_concat, elementwise_product.unsqueeze(-1)],
                                       dim=-1)

        concat_and_product_flat = concat_and_product.view(batch_size, max_nps ** 2, -1)
        anchor_complement_embeddings = self._projection(concat_and_product_flat)

        return dict(embeddings=anchor_complement_embeddings)
