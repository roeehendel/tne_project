import torch

from models.modules.np_embedding.base_np_embedder import BaseNPEmbedder


class ConcatNPEmbedder(BaseNPEmbedder):
    @property
    def output_size(self):
        return self.input_size * 2

    def forward(self, word_embeddings: torch.tensor, nps: torch.tensor) -> torch.tensor:
        batch_size, num_nps, _ = nps.shape
        batch_size, num_tokens, embedding_dim = word_embeddings.shape

        nps_flat = nps.reshape(batch_size, -1)
        np_start_end_embeddings = word_embeddings.gather(1, nps_flat[:, :, None].repeat(1, 1, embedding_dim))
        np_embeddings = np_start_end_embeddings.reshape(batch_size, num_nps, -1)

        return np_embeddings
