from collections import OrderedDict
from dataclasses import dataclass

from torch import nn
from torch.nn import ModuleDict

from models.modules.get_modules_classes import get_tne_modules_classes


@dataclass
class TNEArchitectureConfiguration:
    word_embedder: dict
    np_embedder: dict
    coref_predictor: dict
    np_contextual_embedder: dict
    anchor_complement_embedder: dict
    predictor: dict


class TNEModel(nn.Module):
    _MODULE_INPUTS = OrderedDict(
        word_embedder=None,
        np_embedder='word_embedder',
        coref_predictor='np_embedder',
        np_contextual_embedder='np_embedder',
        anchor_complement_embedder='np_contextual_embedder',
        predictor='anchor_complement_embedder'
    )

    def __init__(self, architecture_configuration: dict):
        super().__init__()

        # Architecture modules
        self._architecture_configuration = architecture_configuration
        self._forward_modules = ModuleDict()
        modules_classes = get_tne_modules_classes()
        for module_name, module_config in self._architecture_configuration.items():
            module_class = modules_classes[module_config['type']]

            input_module_name = TNEModel._MODULE_INPUTS[module_name]
            input_module = self._forward_modules[input_module_name] if input_module_name else None
            input_size = input_module.output_size if input_module is not None else -1

            module = module_class(**module_config['params'], input_size=input_size)
            self._forward_modules[module_name] = module

    @property
    def tokenizer(self):
        return self._forward_modules['word_embedder'].tokenizer

    def forward(self, inputs):
        intermediate_outputs = dict()
        for module_name in self._forward_modules.keys():
            intermediate_outputs[module_name] = self._forward_modules[module_name](inputs, intermediate_outputs)

        outputs = {'tne_logits': intermediate_outputs['predictor']['logits']}

        if intermediate_outputs['coref_predictor'] is not None:
            outputs['coref_logits'] = intermediate_outputs['coref_predictor']['logits']

        return outputs
