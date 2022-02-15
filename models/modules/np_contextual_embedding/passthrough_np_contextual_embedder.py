import torch

from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder


class PassthroughNPContextualEmbedder(BaseNPContextualEmbedder):
    @property
    def output_size(self):
        return self.input_size

    def forward(self, np_embeddings: torch.tensor, token_embedding: torch.tensor, num_nps) -> torch.tensor:
        return np_embeddings
