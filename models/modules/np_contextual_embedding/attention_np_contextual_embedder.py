import torch

from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder


class TransformerNPContextualEmbedder(BaseNPContextualEmbedder):
    def __init__(self, input_size: int, cross_attention: bool, nhead: int, num_layers: int,
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__(input_size)
        self.cross_attention = cross_attention
        if cross_attention:
            decoder_layer = torch.nn.TransformerDecoderLayer(self.input_size, nhead, dim_feedforward, dropout,
                                                             batch_first=True)
            self.transformer = torch.nn.TransformerDecoder(decoder_layer, num_layers)
        else:
            encoder_layer = torch.nn.TransformerEncoderLayer(self.input_size, nhead, dim_feedforward, dropout,
                                                             batch_first=True)
            self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)

    @property
    def output_size(self):
        return self.input_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        num_nps = inputs['num_nps']
        np_embeddings = intermediate_outputs['np_embedder']['embeddings']
        token_embeddings = intermediate_outputs['word_embedder']['embeddings']

        device = np_embeddings.device
        max_nps = np_embeddings.shape[1]
        mask = torch.arange(max_nps)[None, :].to(device) >= num_nps[:, None]
        if self.cross_attention:
            np_embeddings = self.transformer(np_embeddings, token_embeddings, tgt_key_padding_mask=mask)
        else:
            np_embeddings = self.transformer(np_embeddings, src_key_padding_mask=mask)

        return dict(embeddings=np_embeddings)
