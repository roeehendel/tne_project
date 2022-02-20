from transformers import AutoTokenizer, AutoModelForMaskedLM

from models.modules.word_embedding.base_word_embedder import BaseWordEmbedder


class SpanBertWordEmbedder(BaseWordEmbedder):
    def __init__(self, input_size: int,
                 pretrained_model_name: str, freeze_embeddings: bool, num_layers_to_freeze: int,
                 num_layers_to_reinitialize: int):
        super().__init__(input_size)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self._embedder = AutoModelForMaskedLM.from_pretrained(pretrained_model_name, output_hidden_states=True)
        
        self._freeze_embeddings = freeze_embeddings
        self._num_layers_to_freeze = num_layers_to_freeze
        self._num_layers_to_reinitialize = num_layers_to_reinitialize

        self._freeze_layers()
        self._reinitialize_layers()

    def _reinitialize_layers(self):
        for i in range(self._num_layers_to_reinitialize):
            self._embedder.base_model.encoder.layer[-(1 + i)].apply(self._embedder._init_weights)

    def _freeze_layers(self):
        modules_to_freeze = self._embedder.base_model.encoder.layer[:self._num_layers_to_freeze]
        if self._freeze_embeddings:
            modules_to_freeze.append(self._embedder.base_model.embeddings)
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    @property
    def output_size(self):
        return self._embedder.config.hidden_size

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(self, inputs: dict, intermediate_outputs: dict) -> dict:
        ids = inputs['ids']
        mask = inputs['mask']

        encoder_results = self._embedder(ids, attention_mask=mask)
        embeddings = encoder_results.hidden_states[-1]
        return dict(embeddings=embeddings)
