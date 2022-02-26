import torch

from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder


class CorefNPContextualEmbedder(BaseNPContextualEmbedder):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self._gate = torch.nn.Linear(2 * input_size, input_size)

    @property
    def output_size(self):
        return self.input_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        np_embeddings = intermediate_outputs['np_embedder']['embeddings']
        coref_scores = intermediate_outputs['coref_predictor']['logits']

        coref_attention = torch.softmax(coref_scores.transpose(1, 2), dim=-1)
        coref_np_embeddings = torch.bmm(coref_attention, np_embeddings)

        np_embeddings_concat = torch.cat([np_embeddings, coref_np_embeddings], dim=-1)
        gate = torch.sigmoid(self._gate(np_embeddings_concat))

        final_np_embedding = gate * np_embeddings + (1 - gate) * coref_np_embeddings

        return dict(embeddings=final_np_embedding)
