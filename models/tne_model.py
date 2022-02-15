import torch

from models.base_tne_model import BaseTNEModel
from models.modules.anchor_complement_embedding.base_anchor_complement_embedder import BaseAnchorComplementEmbedder
from models.modules.anchor_complement_embedding.concat_anchor_complement_embedder import ConcatAnchorComplementEmbedder
from models.modules.np_contextual_embedding.attention_np_contextual_embedder import AttentionNPContextualEmbedder
from models.modules.np_contextual_embedding.base_np_contextual_embedder import BaseNPContextualEmbedder
from models.modules.np_contextual_embedding.passthrough_np_contextual_embedder import PassthroughNPContextualEmbedder
from models.modules.np_embedding.base_np_embedder import BaseNPEmbedder
from models.modules.np_embedding.concat_np_embedder import ConcatNPEmbedder
from models.modules.prediction.basic_predictor import BasicPredictor
from models.modules.word_embedding.base_word_embedder import BaseWordEmbedder
from models.modules.word_embedding.roberta_word_embedder import RobertaWordEmbedder


class TNEModel(BaseTNEModel):
    def __init__(self, ignore_index: int,
                 learning_rate: float, loss_weight_power: float,

                 word_embedder_type: str, word_embedder_params: dict,
                 np_embedder_type: str, np_embedder_params: dict,
                 np_contextual_embedder_type: str, np_contextual_embedder_params: dict,
                 anchor_complement_embedder_type: str, anchor_complement_embedder_params: dict,
                 predictor_type: str, predictor_params: dict

                 ):
        super().__init__(ignore_index, learning_rate, loss_weight_power)

        # Layers
        self.word_embedder = self._get_word_embedder(
            word_embedder_type,
            word_embedder_params
        )
        self.np_embedder = self._get_np_embedder(
            np_embedder_type, np_embedder_params,
            input_size=self.word_embedder.output_size
        )
        self.np_contextual_embedder = self._get_np_contextual_embedder(
            np_contextual_embedder_type, np_contextual_embedder_params,
            input_size=self.np_embedder.output_size
        )
        self.anchor_complement_embedder = self._get_anchor_complement_embedder(
            anchor_complement_embedder_type, anchor_complement_embedder_params,
            input_size=self.np_contextual_embedder.output_size
        )
        self.predictor = self._get_predictor(
            predictor_type, predictor_params,
            input_size=self.anchor_complement_embedder.output_size
        )

        # Hyper parameters
        self.save_hyperparameters(
            'word_embedder_type',
            'word_embedder_params',
            'np_embedder_type',
            'np_embedder_params',
            'anchor_complement_embedder_type',
            'anchor_complement_embedder_params',
            'np_contextual_embedder_type',
            'np_contextual_embedder_params',
            'predictor_type',
            'predictor_params'
        )

    def _get_word_embedder(self, word_embedder_type: str, word_embedder_params: dict,
                           input_size: int) -> BaseWordEmbedder:
        word_embedder_types = {
            'roberta': RobertaWordEmbedder
        }
        return word_embedder_types[word_embedder_type](**word_embedder_params, input_size=input_size)

    def _get_np_embedder(self, np_embedder_type: str, np_embedder_params: dict, input_size: int) -> BaseNPEmbedder:
        np_embedder_types = {
            'concat': ConcatNPEmbedder
        }
        return np_embedder_types[np_embedder_type](**np_embedder_params, input_size=input_size)

    def _get_np_contextual_embedder(self, np_contextual_embedder_type: str,
                                    np_contextual_embedder_params: dict,
                                    input_size: int) -> BaseNPContextualEmbedder:
        np_contextual_embedder_types = {
            'passthrough': PassthroughNPContextualEmbedder,
            'attention': AttentionNPContextualEmbedder,
        }
        return np_contextual_embedder_types[np_contextual_embedder_type](**np_contextual_embedder_params,
                                                                         input_size=input_size)

    def _get_anchor_complement_embedder(self, anchor_complement_embedder_type: str,
                                        anchor_complement_embedder_params: dict,
                                        input_size: int) -> BaseAnchorComplementEmbedder:
        anchor_complement_embedder_types = {
            'concat': ConcatAnchorComplementEmbedder
        }
        return anchor_complement_embedder_types[anchor_complement_embedder_type](**anchor_complement_embedder_params,
                                                                                 input_size=input_size)

    def _get_predictor(self, predictor_type: str, predictor_params: dict, input_size: int) -> torch.nn.Module:
        predictor_types = {
            'basic': BasicPredictor
        }
        return predictor_types[predictor_type](**predictor_params, input_size=input_size)

    @property
    def tokenizer(self):
        return self.word_embedder.tokenizer

    def forward(self, ids, mask, nps, num_nps):
        word_embeddings = self.word_embedder(ids, mask)
        np_embeddings = self.np_embedder(word_embeddings, nps)
        contextual_np_embeddings = self.np_contextual_embedder(np_embeddings, num_nps)
        anchor_complement_embeddings = self.anchor_complement_embedder(contextual_np_embeddings)
        prediction = self.predictor(anchor_complement_embeddings)
        return prediction
