from torch import nn

from models.modules.anchor_complement_embedding.base_anchor_complement_embedder import BaseAnchorComplementEmbedder


class ConcatAnchorComplementEmbedder(BaseAnchorComplementEmbedder):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self._hidden_size = input_size

        self._anchor_encoder = nn.Linear(self.input_size, self._hidden_size)
        self._complement_encoder = nn.Linear(self.input_size, self._hidden_size)
        self._projection = nn.Linear(2 * self._hidden_size, self._hidden_size)

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        anchor_complement_embeddings_pairs = self._get_anchor_complement_embeddings_pairs(intermediate_outputs)
        batch_size, num_anchor_complement_pairs, _, _ = anchor_complement_embeddings_pairs.shape
        anchor_complement_embeddings_concat = anchor_complement_embeddings_pairs.view(batch_size,
                                                                                      num_anchor_complement_pairs, -1)
        anchor_complement_embeddings = self._projection(anchor_complement_embeddings_concat)
        return dict(embeddings=anchor_complement_embeddings)
