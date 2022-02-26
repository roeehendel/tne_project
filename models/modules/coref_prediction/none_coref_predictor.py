from models.modules.tne_base_module import TNEBaseModule


class NoneCorefPredictor(TNEBaseModule):
    def __init__(self, input_size: int):
        super().__init__(input_size)

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        pass

    @property
    def output_size(self):
        return 0
