from collections import OrderedDict

from torch.nn import ModuleDict

from models.base_tne_model import BaseTNEModel
from models.modules.anchor_complement_embedding.concat_anchor_complement_embedder import ConcatAnchorComplementEmbedder
from models.modules.np_contextual_embedding.attention_np_contextual_embedder import AttentionNPContextualEmbedder
from models.modules.np_contextual_embedding.passthrough_np_contextual_embedder import PassthroughNPContextualEmbedder
from models.modules.np_embedding.concat_np_embedder import ConcatNPEmbedder
from models.modules.prediction.attention_predictor import AttentionPredictor
from models.modules.prediction.basic_predictor import BasicPredictor
from models.modules.word_embedding.roberta_word_embedder import RobertaWordEmbedder


class TNEModel(BaseTNEModel):
    _MODULES = OrderedDict(
        word_embedder={
            "roberta": RobertaWordEmbedder,
        },
        np_embedder={
            "concat": ConcatNPEmbedder,
        },
        np_contextual_embedder={
            "passthrough": PassthroughNPContextualEmbedder,
            "attention": AttentionNPContextualEmbedder,
        },
        anchor_complement_embedder={
            "concat": ConcatAnchorComplementEmbedder,
        },
        predictor={
            "basic": BasicPredictor,
            "attention": AttentionPredictor,
        },
    )

    def __init__(self, ignore_index: int,
                 learning_rate: float, loss_weight_power: float,
                 architecture_config: dict):
        super().__init__(ignore_index, learning_rate, loss_weight_power)

        # Architecture modules
        self.forward_modules = ModuleDict()
        last_module = None
        for module_name, module_classes in TNEModel._MODULES.items():
            module_config = architecture_config[module_name]
            module_class = module_classes[module_config['type']]
            input_size = last_module.output_size if last_module is not None else -1
            module = module_class(**module_config['params'], input_size=input_size)
            self.forward_modules[module_name] = module
            last_module = module

        # Hyper parameters
        self.save_hyperparameters('architecture_config')

    @property
    def tokenizer(self):
        return self.forward_modules['word_embedder'].tokenizer

    def forward(self, inputs):
        intermediate_outputs = dict()
        for module_name in TNEModel._MODULES.keys():
            intermediate_outputs[module_name] = self.forward_modules[module_name](inputs, intermediate_outputs)
        return intermediate_outputs['predictor']['logits']
