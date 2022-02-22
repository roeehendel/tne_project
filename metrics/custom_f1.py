import torch
from torchmetrics import Metric


class CustomF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("predictions_relations", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_relations", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, multi_label_target: torch.Tensor):
        # number of relations that are different from 0 - in our prediction
        relation_preds = preds != 0

        # number of relations that are different from 0 - in the true labels
        relation_true = multi_label_target.sum(dim=1) != 0

        num_preds = preds.shape[0]

        # Relationships we correctly predicted from those other than 0
        true_positives = multi_label_target[torch.arange(num_preds), preds].bool() & relation_preds

        self.true_positive += true_positives.sum().item()
        self.predictions_relations += relation_preds.sum().item()
        self.true_relations += relation_true.sum().item()

    def compute(self):
        precision = self._get_metric(self.true_positive, self.predictions_relations)
        recall = self._get_metric(self.true_positive, self.true_relations)
        if (precision == 0) or (recall == 0):
            return torch.tensor(0)
        return 2 * ((precision * recall) / (precision + recall))

    def _get_metric(self, numerator, denominator):
        return numerator / denominator if denominator else 0
