from collections import OrderedDict

from torch.nn import ModuleDict

from models.base_tne_model import BaseTNEModel
from models.modules.anchor_complement_embedding.concat_anchor_complement_embedder import ConcatAnchorComplementEmbedder
from models.modules.anchor_complement_embedding.multiplicative_anchor_complement_embedder import \
    MultiplicativeAnchorComplementEmbedder
from models.modules.coref_prediction.coref_predictor import CorefPredictor
from models.modules.np_contextual_embedding.attention_np_contextual_embedder import TransformerNPContextualEmbedder
from models.modules.np_contextual_embedding.coref_np_contextual_embedder import CorefNPContextualEmbedder
from models.modules.np_contextual_embedding.passthrough_np_contextual_embedder import PassthroughNPContextualEmbedder
from models.modules.np_embedding.attention_np_embedder import AttentionNPEmbedder
from models.modules.np_embedding.concat_np_embedder import ConcatNPEmbedder
from models.modules.prediction.attention_predictor import TransformerPredictor
from models.modules.prediction.basic_predictor import BasicPredictor
from models.modules.word_embedding.roberta_word_embedder import RobertaWordEmbedder
from models.modules.word_embedding.spanbert_word_embedder import SpanBertWordEmbedder


class TNEModel(BaseTNEModel):
    _MODULES = OrderedDict(
        word_embedder={
            "roberta": RobertaWordEmbedder,
            "spanbert": SpanBertWordEmbedder,
        },
        np_embedder={
            "concat": ConcatNPEmbedder,
            "attention": AttentionNPEmbedder
        },
        coref_predictor={
            "basic": CorefPredictor,
        },
        np_contextual_embedder={
            "passthrough": PassthroughNPContextualEmbedder,
            "transformer": TransformerNPContextualEmbedder,
            "coref": CorefNPContextualEmbedder,
        },
        anchor_complement_embedder={
            "concat": ConcatAnchorComplementEmbedder,
            "multiplicative": MultiplicativeAnchorComplementEmbedder,
        },
        predictor={
            "basic": BasicPredictor,
            "transformer": TransformerPredictor,
        },
    )

    _MODULE_INPUTS = OrderedDict(
        word_embedder=None,
        np_embedder='word_embedder',
        coref_predictor='np_embedder',
        np_contextual_embedder='np_embedder',
        anchor_complement_embedder='np_contextual_embedder',
        predictor='anchor_complement_embedder'
    )

    def __init__(self, ignore_index: int,
                 learning_rate: float, loss_weight_power: float,
                 architecture_config: dict):
        super().__init__(ignore_index, learning_rate, loss_weight_power)

        # Architecture modules
        self.forward_modules = ModuleDict()
        for module_name, module_classes in TNEModel._MODULES.items():
            module_config = architecture_config[module_name]
            module_class = module_classes[module_config['type']]

            input_module_name = TNEModel._MODULE_INPUTS[module_name]
            input_module = self.forward_modules[input_module_name] if input_module_name else None
            input_size = input_module.output_size if input_module is not None else -1

            module = module_class(**module_config['params'], input_size=input_size)
            self.forward_modules[module_name] = module

        # Hyper parameters
        self.save_hyperparameters('architecture_config')

    @property
    def tokenizer(self):
        return self.forward_modules['word_embedder'].tokenizer

    def forward(self, inputs):
        intermediate_outputs = dict()
        for module_name in self.forward_modules.keys():
            intermediate_outputs[module_name] = self.forward_modules[module_name](inputs, intermediate_outputs)

        outputs = {'tne_logits': intermediate_outputs['predictor']['logits']}

        if self._use_coref_loss:
            outputs['coref_logits'] = intermediate_outputs['coref_predictor']['logits']

        return outputs
