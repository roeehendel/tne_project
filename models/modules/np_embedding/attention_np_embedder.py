import torch

from models.modules.np_embedding.base_np_embedder import BaseNPEmbedder


class AttentionNPEmbedder(BaseNPEmbedder):
    def __init__(self, input_size: int):
        super().__init__(input_size)
        self.attn = torch.nn.Linear(in_features=input_size, out_features=1)
        # self.dropout = torch.nn.Dropout(config.dropout_rate)

    @property
    def output_size(self):
        return self.input_size

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        nps = inputs['nps']
        word_embeddings = intermediate_outputs['word_embedder']['embeddings']
        num_nps = inputs['num_nps']

        attn_scores = self._attn_scores(word_embeddings, nps, num_nps)
        np_embeddings = torch.bmm(attn_scores, word_embeddings)

        # words = self.dropout(words)
        return dict(embeddings=np_embeddings)

    def _attn_scores(self, word_embeddings: torch.Tensor, nps: torch.Tensor, num_nps: torch.Tensor):
        batch_size, max_nps, _ = nps.shape
        batch_size, num_tokens, embedding_dim = word_embeddings.shape
        device = word_embeddings.device

        mask = (torch.arange(max_nps)[None, :].to(device) >= num_nps[:, None]) * -1

        start_idx = nps[:, :, 0]
        end_idx = nps[:, :, 1] + mask # in this way we will ignore idx after num_nps

        attn_mask = torch.arange(0, num_tokens, device=device).expand((batch_size, max_nps, num_tokens))
        attn_mask = ((attn_mask >= start_idx.unsqueeze(2)) * (attn_mask <= end_idx.unsqueeze(2))
                     * end_idx.unsqueeze(2) != -1)

        attn_mask = torch.log(attn_mask.to(torch.float))

        attn_scores = self.attn(word_embeddings).transpose(1, 2)  # [batch ,1, num_tokens]
        attn_scores = attn_scores.expand((batch_size, max_nps, num_tokens))
        attn_scores = attn_mask + attn_scores
        del attn_mask
        return torch.softmax(attn_scores, dim=2)
