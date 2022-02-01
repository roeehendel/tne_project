import torch
import torch.nn.functional as F
import transformers
from pytorch_lightning import LightningModule


class BaseTNEModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        encoder_results = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        pooler_output = encoder_results['pooler_output']
        prediction = self.l2(pooler_output).squeeze()
        return prediction

    def training_step(self, batch, batch_idx):
        data = batch
        device = self.device

        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        logits = self(ids, mask, token_type_ids)

        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-04)
