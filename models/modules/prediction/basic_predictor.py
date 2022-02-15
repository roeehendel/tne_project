import torch
import torch.nn.functional as F

from models.modules.prediction.base_predictor import BasePredictor


class BasicPredictor(BasePredictor):
    def __init__(self, anchor_complement_embedding_size: int, hidden_size: int, num_prepositions: int):
        super().__init__()
        self._linear1 = torch.nn.Linear(anchor_complement_embedding_size, hidden_size)
        self._linear2 = torch.nn.Linear(hidden_size, num_prepositions)

    def forward(self, anchor_complement_embeddings: torch.tensor) -> torch.tensor:
        hidden = self._linear1(anchor_complement_embeddings)
        logits = self._linear2(F.relu(hidden))
        return logits
