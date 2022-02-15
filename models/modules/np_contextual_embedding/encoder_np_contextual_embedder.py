from abc import abstractmethod

import torch

from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder


class EncoderNPContextualEmbedder(BaseNPContextualEmbedder):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)

    @abstractmethod
    def forward(self, np_embeddings: torch.tensor, num_nps) -> torch.tensor:
        max_nps = np_embeddings.shape[1]
        mask = torch.arange(max_nps)[None, :] >= num_nps[:, None]
        return self.transformer_encoder(np_embeddings, mask)
