import torch
from torch import nn

from models.modules.np_embedding.base_np_embedder import BaseNPEmbedder


class AttentionConcatNPEmbedder(BaseNPEmbedder):
    def __init__(self, input_size: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__(input_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(input_size, nhead, dim_feedforward, dropout,
                                                         batch_first=True)
        self._transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self._projection = torch.nn.Linear(input_size * 2, input_size)

        self._np_start_encoding = nn.Parameter(torch.randn(input_size))
        self._np_end_encoding = nn.Parameter(torch.randn(input_size))

    @property
    def output_size(self):
        return self.input_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        nps = inputs['nps']
        word_embeddings = intermediate_outputs['word_embedder']['embeddings']
        mask = inputs['mask']

        batch_size, num_nps, _ = nps.shape
        batch_size, num_tokens, embedding_dim = word_embeddings.shape

        for i in range(batch_size):
            word_embeddings[i, nps[i, :, 0], :] += self._np_start_encoding
            word_embeddings[i, nps[i, :, 1], :] += self._np_end_encoding

        word_embeddings = self._transformer(word_embeddings, src_key_padding_mask=mask)

        nps_flat = nps.reshape(batch_size, -1)
        np_start_end_embeddings = word_embeddings.gather(1, nps_flat[:, :, None].repeat(1, 1, embedding_dim))
        np_embeddings_concat = np_start_end_embeddings.reshape(batch_size, num_nps, -1)

        np_embeddings_projected = self._projection(np_embeddings_concat.reshape(-1, np_embeddings_concat.shape[-1]))
        np_embeddings = np_embeddings_projected.reshape(batch_size, num_nps, -1)

        return dict(embeddings=np_embeddings)
