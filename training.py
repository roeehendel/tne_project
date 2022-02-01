from pytorch_lightning import Trainer
from transformers import BertTokenizer

from data_loading.tne_data_module import TNEDataModule
from models.base_tne_model import BaseTNEModel

EPOCHS = 200

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tne_dm = TNEDataModule(tokenizer)
model = BaseTNEModel()
trainer = Trainer(
    max_epochs=EPOCHS,
    gpus=1,
)
trainer.fit(model, tne_dm)
