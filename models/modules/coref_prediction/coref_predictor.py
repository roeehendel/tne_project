import torch
from torch import nn

from models.modules.tne_base_module import TNEBaseModule


class CorefPredictor(TNEBaseModule):
    def __init__(self, input_size: int, hidden_size: int = 256):
        super().__init__(input_size)
        self._hidden_size = hidden_size

        self._projection = nn.Linear(self.input_size, self._hidden_size)
        self._predictor = nn.Linear(self._hidden_size * 3, 1)

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        np_embeddings = intermediate_outputs['np_embedder']['embeddings']

        batch_size, max_nps, _ = np_embeddings.shape

        np_embeddings_projected = self._projection(np_embeddings)

        np_embeddings_pairs = torch.cat(
            [np_embeddings_projected[:, None, :, :, None].repeat(1, max_nps, 1, 1, 1),
             np_embeddings_projected[:, :, None, :, None].repeat(1, 1, max_nps, 1, 1)],
            dim=-1
        ).view(batch_size, max_nps, max_nps, self._hidden_size, 2)

        elementwise_product = np_embeddings_pairs.prod(axis=-1)
        concat_and_product = torch.cat([np_embeddings_pairs, elementwise_product.unsqueeze(-1)], dim=-1)
        concat_and_product_flat = concat_and_product.view(batch_size, max_nps, max_nps, -1)

        coref_scores = self._predictor(concat_and_product_flat).squeeze()
        coref_scores = coref_scores + coref_scores.transpose(1, 2)

        return dict(scores=coref_scores)

    @property
    def output_size(self):
        return 0
