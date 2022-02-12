import dataclasses
import math
from typing import Callable, Dict

import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from pytorch_lightning import LightningModule
from transformers import BertTokenizerFast

from data_loading.tne_dataset import PREPOSITION_LIST


@dataclasses.dataclass
class MetricConfig:
    metric_functions: Dict[str, torchmetrics.Metric]
    averaging_method: str
    preprocessor: Callable
    num_classes: int = len(PREPOSITION_LIST)
    multiclass: bool = True


class BaseTNEModel(LightningModule):
    def __init__(self, ignore_index: int,
                 learning_rate: float = 1e-04,
                 anchor_complement_embedding_size: int = 768, predictor_hidden: int = 512,
                 num_prepositions: int = len(PREPOSITION_LIST), freeze_embedder: bool = False):
        super().__init__()
        # Hyper Parameters
        self.ignore_index = ignore_index
        self.save_hyperparameters(
            "learning_rate",
            "freeze_embedder",
            "anchor_complement_embedding_size",
            "predictor_hidden",
            "num_prepositions"
        )

        # Tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Layers
        self.embedder = transformers.BertModel.from_pretrained('bert-base-uncased')
        if freeze_embedder:
            for param in self.embedder.parameters():
                param.requires_grad = False

        embedder_hidden_size = self.embedder.config.hidden_size
        self.anchor_encoder = torch.nn.Linear(2 * embedder_hidden_size, self.hparams.anchor_complement_embedding_size)
        self.complement_encoder = torch.nn.Linear(2 * embedder_hidden_size,
                                                  self.hparams.anchor_complement_embedding_size)

        self.predictor_1 = torch.nn.Linear(2 * self.hparams.anchor_complement_embedding_size,
                                           self.hparams.predictor_hidden)
        self.predictor_2 = torch.nn.Linear(self.hparams.predictor_hidden, self.hparams.num_prepositions)

        # Loss
        self.loss_weight = torch.tensor([1] + [10 for _ in range(self.hparams.num_prepositions - 1)],
                                        device=self.device, dtype=torch.float)

        # Metrics
        self.data_splits = ['train', 'dev']

        metric_functions = {
            'precision': torchmetrics.Precision,
            'recall': torchmetrics.Recall,
            'f1': torchmetrics.F1Score,
        }

        self.metric_configs = {
            'prepositions': MetricConfig(
                metric_functions=metric_functions,
                preprocessor=lambda predictions, targets: (predictions, targets),
                averaging_method='micro'
            ),
            'prepositions_linked_targets': MetricConfig(
                metric_functions=metric_functions,
                preprocessor=lambda predictions, targets: (predictions[targets != 0], targets[targets != 0]),
                averaging_method='micro'
            ),
            'prepositions_linked_predictions': MetricConfig(
                metric_functions=metric_functions,
                preprocessor=lambda predictions, targets: (predictions[predictions != 0], targets[predictions != 0]),
                averaging_method='micro'
            ),
            'prepositions_linked_both': MetricConfig(
                metric_functions=metric_functions,
                preprocessor=lambda predictions, targets: (
                    predictions[(predictions != 0) & (targets != 0)], targets[(predictions != 0) & (targets != 0)]),
                averaging_method='micro'
            ),
            'links': MetricConfig(
                metric_functions=metric_functions,
                preprocessor=lambda predictions, targets: (predictions != 0, targets != 0),
                averaging_method='none',
                num_classes=1,
                multiclass=False
            )
        }

        self.metrics = {
            data_split: {
                metric_name: {
                    metric_function_name: metric_function(average=metric_config.averaging_method,
                                                          num_classes=metric_config.num_classes,
                                                          multiclass=metric_config.multiclass)
                    for metric_function_name, metric_function in metric_config.metric_functions.items()
                } for metric_name, metric_config in self.metric_configs.items()
            }
            for data_split in self.data_splits
        }

        # Register the metrics
        for data_split in self.metrics:
            for metric_name in self.metrics[data_split]:
                for metric_function_name in self.metrics[data_split][metric_name]:
                    metric = self.metrics[data_split][metric_name][metric_function_name]
                    self.__setattr__(f'{data_split}/{metric_name}_{metric_function_name}', metric)

    def forward(self, ids, mask, token_type_ids, nps):
        encoder_results = self.embedder(ids, attention_mask=mask, token_type_ids=token_type_ids)
        embeddings = encoder_results['last_hidden_state']

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

        predictor_hidden = torch.relu(self.predictor_1(anchor_complement_pair_embeddings))
        prediction = self.predictor_2(predictor_hidden)

        return prediction

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        loss, target, predictions = self._train_val_step(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self._calc_metrics(target, predictions, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        loss, target, predictions = self._train_val_step(batch)

        self.log('dev_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self._calc_metrics(target, predictions, 'dev')

        return loss

    def test_step(self, batch, batch_idx):
        num_nps = batch['num_nps']
        logits = self._predict_logits(batch)
        predictions = logits.argmax(dim=-1)

        batch_size, max_num_nps_squared = predictions.shape
        max_num_nps = int(math.sqrt(max_num_nps_squared))
        predictions = predictions.reshape(-1, max_num_nps, max_num_nps)
        predictions[:, torch.arange(max_num_nps), torch.arange(max_num_nps)] = -1
        predictions_per_step = [predictions[i][:num_nps[i], :num_nps[i]].flatten().tolist() for i in range(batch_size)]

        return {
            'predictions': predictions_per_step
        }

    def test_epoch_end(self, all_predictions):
        predictions = [example_prediction
                       for batch_predictions in all_predictions
                       for example_prediction in batch_predictions['predictions']]
        self.test_results = predictions

    def _train_val_step(self, batch):
        logits = self._predict_logits(batch)
        targets = self._unpack_targets(batch)

        logits_flat = self._flatten_logits(logits)
        predictions = logits_flat.argmax(dim=-1)
        targets = targets.flatten()

        self.loss_weight = self.loss_weight.to(self.device)
        loss = F.cross_entropy(logits_flat, targets, ignore_index=self.ignore_index, weight=self.loss_weight)

        return loss, targets, predictions

    def _predict_logits(self, batch):
        model_inputs = self._unpack_model_inputs(batch)
        prediction = self.forward(*model_inputs)
        return prediction

    def _flatten_logits(self, logits):
        num_classes = logits.shape[-1]
        logits_flat = logits.reshape(-1, num_classes)
        return logits_flat

    def _unpack_model_inputs(self, batch):
        device = self.device

        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
        nps = batch['nps'].to(device, dtype=torch.long)

        return ids, mask, token_type_ids, nps

    def _unpack_targets(self, batch):
        device = self.device
        targets = batch['targets'].to(device, dtype=torch.long)
        return targets

    def _calc_metrics(self, targets, predictions, data_split):
        targets_masked = targets[targets != self.ignore_index]
        predictions_masked = predictions[targets != self.ignore_index]

        for metric_name, metric_config in self.metric_configs.items():
            metric_preprocessor = metric_config.preprocessor
            for metric_function_name in metric_config.metric_functions.keys():
                metric = self.metrics[data_split][metric_name][metric_function_name]
                targets_preprocessed, predictions_preprocessed = metric_preprocessor(predictions_masked, targets_masked)
                if len(targets_preprocessed) != 0:
                    if metric_config.num_classes == 1:
                        targets_preprocessed = targets_preprocessed.bool()
                        predictions_preprocessed = predictions_preprocessed.bool()
                    metric(targets_preprocessed.to(self.device), predictions_preprocessed.to(self.device))
                    self.log(f'{data_split}_{metric_name}_{metric_function_name}', metric, on_step=True, on_epoch=True)
