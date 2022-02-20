import dataclasses
import math
from abc import abstractmethod
from typing import Callable, Dict

import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning import LightningModule
from torchmetrics import Metric

from data_loading.tne_class_weights import TNE_CLASS_WEIGHTS
from data_loading.tne_dataset import NUM_PREPOSITIONS
from metrics.custom_f1 import CustomF1


@dataclasses.dataclass
class MetricConfig:
    metric_functions: Dict[str, Metric]
    preprocessor: Callable
    metric_functions_kwargs: Dict = dataclasses.field(default_factory=dict)


class BaseTNEModel(LightningModule):
    def __init__(self, ignore_index: int, learning_rate: float, loss_weight_power,
                 num_prepositions: int = NUM_PREPOSITIONS):
        super().__init__()
        # Hyper Parameters
        self.ignore_index = ignore_index
        self.save_hyperparameters(
            "learning_rate",
            "loss_weight_power",
            "num_prepositions"
        )

        # Loss
        self.loss_weight = self._loss_weight()

        # Metrics
        self.data_splits = ['train', 'dev']

        self.metric_configs = {
            'links': MetricConfig(
                metric_functions={'precision': torchmetrics.Precision, 'recall': torchmetrics.Recall,
                                  'f1': torchmetrics.F1Score},
                preprocessor=lambda predictions, targets: (predictions != 0, targets != 0),
                metric_functions_kwargs={
                    'average': 'none',
                    'num_classes': 1,
                    'multiclass': False
                }
            ),
            'prepositions': MetricConfig(
                metric_functions={'custom_f1': CustomF1},
                preprocessor=lambda predictions, targets: (predictions, targets),
            ),
        }

        self.metrics = {
            data_split: {
                metric_name: {
                    metric_function_name: metric_function(**metric_config.metric_functions_kwargs)
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
                    self.__setattr__(self._metric_full_name(data_split, metric_name, metric_function_name), metric)

    @property
    @abstractmethod
    def tokenizer(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        tne_loss, coref_loss, loss, tne_targets, predictions = self._train_val_step(batch)

        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/tne_loss', tne_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/coref_loss', coref_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self._calc_metrics(tne_targets, predictions, 'train')

        return loss

    def validation_step(self, batch, batch_idx):
        tne_loss, coref_loss, loss, tne_targets, tne_predictions = self._train_val_step(batch)

        self.log('dev/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('dev/tne_loss', tne_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('dev/coref_loss', coref_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self._calc_metrics(tne_targets, tne_predictions, 'dev')

        return loss

    def test_step(self, batch, batch_idx):
        num_nps = batch['num_nps']
        tne_logits, coref_logits = self._predict_logits(batch)
        predictions = tne_logits.argmax(dim=-1)

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

    def _loss_weight(self):
        class_weights = torch.tensor(TNE_CLASS_WEIGHTS, device=self.device, dtype=torch.float)
        loss_weight = (1 / class_weights)
        loss_weight = loss_weight ** self.hparams.loss_weight_power
        loss_weight = loss_weight / loss_weight.sum()
        return loss_weight

    def _train_val_step(self, batch):
        tne_logits, coref_logits = self._predict_logits(batch)
        tne_targets, coref_targets = self._unpack_targets(batch)

        logits_flat = self._flatten_logits(tne_logits)
        tne_predictions = logits_flat.argmax(dim=-1)
        tne_targets = tne_targets.flatten()

        self.loss_weight = self.loss_weight.to(self.device)
        tne_loss = F.cross_entropy(logits_flat, tne_targets, ignore_index=self.ignore_index, weight=self.loss_weight)

        mask = coref_targets != self.ignore_index
        coref_loss = F.binary_cross_entropy_with_logits(coref_logits[mask], coref_targets[mask].float())

        use_coref_loss = True

        loss = tne_loss
        if use_coref_loss:
            loss += coref_loss

        return tne_loss, coref_loss, loss, tne_targets, tne_predictions

    def _predict_logits(self, batch):
        model_inputs = self._unpack_model_inputs(batch)
        model_outputs = self.forward(model_inputs)
        return model_outputs['tne_logits'], model_outputs['coref_logits']

    def _flatten_logits(self, logits):
        num_classes = logits.shape[-1]
        logits_flat = logits.reshape(-1, num_classes)
        return logits_flat

    def _unpack_model_inputs(self, batch):
        device = self.device

        ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        nps = batch['nps'].to(device, dtype=torch.long)
        num_nps = batch['num_nps'].to(device, dtype=torch.long)

        return dict(ids=ids, mask=mask, nps=nps, num_nps=num_nps)

    def _unpack_targets(self, batch):
        device = self.device
        tne_targets = batch['targets'].to(device, dtype=torch.long)
        coref_targets = batch['coref_targets'].to(device, dtype=torch.long)
        return tne_targets, coref_targets

    def _calc_metrics(self, targets, predictions, data_split):
        targets_masked = targets[targets != self.ignore_index]
        predictions_masked = predictions[targets != self.ignore_index]

        for metric_name, metric_config in self.metric_configs.items():
            metric_preprocessor = metric_config.preprocessor
            for metric_function_name in metric_config.metric_functions.keys():
                metric = self.metrics[data_split][metric_name][metric_function_name]
                targets_preprocessed, predictions_preprocessed = metric_preprocessor(predictions_masked, targets_masked)
                if len(targets_preprocessed) != 0:
                    metric(targets_preprocessed.to(self.device), predictions_preprocessed.to(self.device))
                    self.log(self._metric_full_name(data_split, metric_name, metric_function_name),
                             metric, on_step=True, on_epoch=True)

    @staticmethod
    def _metric_full_name(data_split, metric_name, metric_function_name):
        return f'{data_split}/{metric_name}/{metric_function_name}'
