from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from transformers import BertTokenizerFast

from data_loading.tne_data_module import TNEDataModule
from models.base_tne_model import BaseTNEModel

EPOCHS = 200

wandb_logger = WandbLogger(project="TNE")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
tne_dm = TNEDataModule(tokenizer)
model = BaseTNEModel()
trainer = Trainer(
    max_epochs=EPOCHS,
    gpus=1,
    logger=wandb_logger,
    log_every_n_steps=4
)
trainer.fit(model, tne_dm)
