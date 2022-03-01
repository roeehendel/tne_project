import torch
import torch.nn.functional as F

from data_loading.tne_class_weights import TNE_CLASS_WEIGHTS
from data_loading.tne_dataset import NUM_PREPOSITIONS
from models.modules.prediction.base_predictor import BasePredictor


class BasicPredictor(BasePredictor):
    @property
    def output_size(self):
        return self._num_prepositions

    def __init__(self, input_size: int, initialize_bias: bool,
                 hidden_size: int = 128, num_prepositions: int = NUM_PREPOSITIONS):
        super().__init__(input_size)
        self._hidden_size = hidden_size
        self._num_prepositions = num_prepositions

        self._linear1 = torch.nn.Linear(input_size, hidden_size)
        self._dropout = torch.nn.Dropout(p=0.3)
        self._linear2 = torch.nn.Linear(hidden_size, num_prepositions)

        if initialize_bias:
            self._linear2.bias.data.copy_(torch.log(torch.tensor(TNE_CLASS_WEIGHTS)))

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        anchor_complement_embeddings = intermediate_outputs["anchor_complement_embedder"]["embeddings"]

        hidden = self._linear1(anchor_complement_embeddings)
        dropout = self._dropout(hidden)
        logits = self._linear2(F.relu(dropout))
        return dict(logits=logits)
