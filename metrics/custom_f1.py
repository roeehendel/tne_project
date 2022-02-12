from torchmetrics import Metric
import torch

class CustomF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # those are the relevant states. Scalars with a size of 0
        self.add_state("true_positive", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("predictions_relations", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("true_relations", default=torch.tensor(0), dist_reduce_fx="sum")


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # number of relations that are different from 0 - in our prediction
        relation_preds = preds != 0

        # number of relations that are different from 0 - in the true labels
        relation_true = target != 0

        # Relationships we correctly predicted from those other than 0
        true_positives = (preds == target) & relation_preds

        self.true_positive += torch.sum(true_positives).item()
        self.predictions_relations += relation_preds.sum().item()
        self.true_relations += relation_true.sum().item()

    def get_metric(self, numerator ,denominator):
        return numerator / denominator if denominator else 0

    def compute(self):
        precision = self.get_metric(self.true_positive, self.predictions_relations)
        recall = self.get_metric(self.true_positive, self.true_relations)
        return 0 if (not precision or not recall) else 2 * ((precision * recall) / (precision + recall))

