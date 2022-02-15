import torch
from torch import nn

from models.base_tne_model import BaseTNEModel


class TNEModel(BaseTNEModel):
    def __init__(self, ignore_index: int,
                 learning_rate: float, loss_weight_power: float,
                 word_embedder_type: str, word_embedder_kwargs: dict,
                 pretrained_model_name: str,
                 anchor_complement_embedding_size: int = 512, predictor_hidden: int = 256,
                 num_layers_to_freeze: int = 0, freeze_embeddings: bool = True, num_layers_to_reinitialize: int = 0):
        super().__init__(ignore_index, learning_rate, loss_weight_power)

        # Hyper parameters
        hyper_parameters = ['word_embedder_type']
        self.save_hyperparameters(
            'anchor_complement_embedding_size',
            'predictor_hidden',
            'num_layers_to_freeze',
            'freeze_embeddings',
            'num_layers_to_reinitialize',
            'pretrained_model_name'
        )

        # Layers
        self.word_embedder = self._get_word_embedder()
        self._tokenizer = self.word_embedder.tokenizer
        self.anchor_encoder = self._anchor_complement_encoder()
        self.complement_encoder = self._anchor_complement_encoder()
        self.predictor = self._predictor()

        # Layer freezing
        modules_to_freeze = self.embedder.base_model.encoder.layer[:self.hparams.num_layers_to_freeze]
        if self.hparams.freeze_embeddings:
            modules_to_freeze.append(self.embedder.base_model.embeddings)
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        # Layer reinitialization
        for i in range(self.hparams.num_layers_to_reinitialize):
            self.embedder.base_model.encoder.layer[-(1 + i)].apply(self.embedder._init_weights)

    def _get_word_embedder(self):
        pass

    @property
    def tokenizer(self):
        return self._tokenizer

    def forward(self, ids, mask, nps):
        encoder_results = self.embedder(ids, attention_mask=mask)
        embeddings = encoder_results.hidden_states[-1]

        batch_size, num_nps, _ = nps.shape
        batch_size, num_tokens, embedding_dim = embeddings.shape

        nps_flat = nps.reshape(batch_size, -1)
        np_start_end_embeddings = embeddings.gather(1, nps_flat[:, :, None].repeat(1, 1, embedding_dim))

        np_embeddings = np_start_end_embeddings.reshape(batch_size, num_nps, -1)

        anchor_embeddings = self.anchor_encoder(np_embeddings)
        complement_embeddings = self.complement_encoder(np_embeddings)

        anchor_complement_pair_embeddings = torch.cat(
            [anchor_embeddings[:, None, :, :, None].repeat(1, num_nps, 1, 1, 1),
             complement_embeddings[:, :, None, :, None].repeat(1, 1, num_nps, 1, 1)],
            dim=-1
        ).reshape(batch_size, num_nps ** 2, -1)

        prediction = self.predictor(anchor_complement_pair_embeddings)

        return prediction

    def _anchor_complement_encoder(self):
        embedder_hidden_size = self.embedder.config.hidden_size
        return nn.Linear(2 * embedder_hidden_size, self.hparams.anchor_complement_embedding_size)

    def _predictor(self):
        return nn.Sequential(
            torch.nn.Linear(2 * self.hparams.anchor_complement_embedding_size, self.hparams.predictor_hidden),
            nn.ReLU(),
            torch.nn.Linear(self.hparams.predictor_hidden, self.hparams.num_prepositions)
        )
