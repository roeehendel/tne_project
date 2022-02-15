from abc import abstractmethod

import torch

from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder


class AttentionNPContextualEmbedder(BaseNPContextualEmbedder):
    def __init__(self, cross_attention: bool, np_embedding_size: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.cross_attention = cross_attention
        if cross_attention:
            decoder_layer = torch.nn.TransformerDecoderLayer(np_embedding_size, nhead, dim_feedforward, dropout)
            self.transformer = torch.nn.TransformerDecoder(decoder_layer, num_layers)
        else:
            encoder_layer = torch.nn.TransformerEncoderLayer(np_embedding_size, nhead, dim_feedforward, dropout)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)

    @abstractmethod
    def forward(self, np_embeddings: torch.tensor, token_embedding: torch.tensor, num_nps) -> torch.tensor:
        max_nps = np_embeddings.shape[1]
        mask = torch.arange(max_nps)[None, :] >= num_nps[:, None]
        if self.cross_attention:
            return self.transformer(np_embeddings, token_embedding, mask)
        return self.transformer(np_embeddings, mask)
