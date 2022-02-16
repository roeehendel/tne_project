import torch

from models.modules.np_embedding.base_np_embedder import BaseNPEmbedder


class ConcatNPEmbedder(BaseNPEmbedder):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.projection = torch.nn.Linear(input_size * 2, input_size)

    @property
    def output_size(self):
        return self.input_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        nps = inputs['nps']
        word_embeddings = intermediate_outputs['word_embedder']['embeddings']

        batch_size, num_nps, _ = nps.shape
        batch_size, num_tokens, embedding_dim = word_embeddings.shape

        nps_flat = nps.reshape(batch_size, -1)
        np_start_end_embeddings = word_embeddings.gather(1, nps_flat[:, :, None].repeat(1, 1, embedding_dim))
        np_embeddings_concat = np_start_end_embeddings.reshape(batch_size, num_nps, -1)

        np_embeddings_projected = self.projection(np_embeddings_concat.reshape(-1, np_embeddings_concat.shape[-1]))
        np_embeddings = np_embeddings_projected.reshape(batch_size, num_nps, -1)

        return dict(embeddings=np_embeddings)
