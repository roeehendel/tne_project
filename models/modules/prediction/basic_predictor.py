import torch
import torch.nn.functional as F

from models.modules.prediction.base_predictor import BasePredictor


class BasicPredictor(BasePredictor):
    @property
    def output_size(self):
        return self._num_prepositions

    def __init__(self, input_size: int, hidden_size: int, num_prepositions: int):
        super().__init__(input_size)
        self._hidden_size = hidden_size
        self._num_prepositions = num_prepositions

        self._linear1 = torch.nn.Linear(input_size, hidden_size)
        self._linear2 = torch.nn.Linear(hidden_size, num_prepositions)

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        anchor_complement_embeddings = intermediate_outputs["anchor_complement_embedder"]["embeddings"]

        hidden = self._linear1(anchor_complement_embeddings)
        logits = self._linear2(F.relu(hidden))
        return dict(logits=logits)
