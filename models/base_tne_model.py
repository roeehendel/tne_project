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

    def unpack_data(self, data):
        device = self.device

        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        return (ids, mask, token_type_ids), targets

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-04)

    def general_step(self, batch):
        x, y = self.unpack_data(batch)
        logits = self(*x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch)
        self.log('dev_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
