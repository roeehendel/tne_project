import torch
import torch.nn.functional as F

from models.modules.prediction.base_predictor import BasePredictor


class TransformerPredictor(BasePredictor):
    @property
    def output_size(self):
        return self._num_prepositions

    def __init__(self, input_size: int, hidden_size: int, num_prepositions: int, nhead: int, num_layers: int,
                 dim_feedforward: int = 32, dropout: float = 0.1):
        super().__init__(input_size)
        self._hidden_size = hidden_size
        self._num_prepositions = num_prepositions

        self._projection = torch.nn.Linear(input_size, hidden_size)
        encoder_layer = torch.nn.TransformerEncoderLayer(hidden_size, nhead, dim_feedforward, dropout,
                                                         batch_first=True)
        self._transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self._predictor = torch.nn.Linear(hidden_size, num_prepositions)

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        anchor_complement_embeddings = intermediate_outputs["anchor_complement_embedder"]["embeddings"]

        hidden = self._projection(anchor_complement_embeddings)
        transformed = self._transformer(hidden)
        logits = self._predictor(F.relu(transformed))
        return dict(logits=logits)
