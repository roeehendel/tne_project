from abc import abstractmethod

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForMaskedLM


class RobertaWordEmbedder(nn.Module):
    def __init__(self, pretrained_model_name: str):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.embedder = AutoModelForMaskedLM.from_pretrained(pretrained_model_name, output_hidden_states=True)

    @abstractmethod
    def forward(self, ids: torch.tensor, mask: torch.tensor) -> torch.tensor:
        encoder_results = self.embedder(ids, attention_mask=mask)
        embeddings = encoder_results.hidden_states[-1]
        return embeddings
