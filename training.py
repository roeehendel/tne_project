import os

from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config import CHECKPOINTS_ROOT_DIR, IGNORE_INDEX
from data_loading.tne_data_module import TNEDataModule
from models.base_tne_model import BaseTNEModel
from utils.initialization import initialize


class SaveTopKModelsCallback(Callback):
    def __init__(self, checkpoint_callback: ModelCheckpoint):
        self.checkpoint_callback = checkpoint_callback

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch > 0:
            self.checkpoint_callback.to_yaml()


def train(epochs: int, data_loaders_num_workers: int, run_name: str):
    initialize(0)

    run_dir_path = os.path.join(CHECKPOINTS_ROOT_DIR, run_name)

    wandb_logger = WandbLogger(project="TNE")

    model = BaseTNEModel(ignore_index=IGNORE_INDEX, freeze_embedder=False)
    tne_data_module = TNEDataModule(model.tokenizer, ignore_index=IGNORE_INDEX, num_workers=data_loaders_num_workers)

    dev_prepositions_f1_checkpoint_callback = ModelCheckpoint(
        monitor="dev_prepositions_f1",
        dirpath=run_dir_path,
        filename='tne-model-{epoch:02d}-{dev_prepositions_f1:.2f}',
        save_top_k=3,
        mode="max",
        save_last=True,
        auto_insert_metric_name=True
    )
    save_top_k_callback = SaveTopKModelsCallback(dev_prepositions_f1_checkpoint_callback)

    trainer = Trainer(
        max_epochs=epochs,
        gpus=1,
        logger=wandb_logger,
        log_every_n_steps=4,
        callbacks=[dev_prepositions_f1_checkpoint_callback, save_top_k_callback],
    )
    trainer.fit(model, tne_data_module)


if __name__ == '__main__':
    EPOCHS = 100
    DATA_LOADERS_NUM_WORKERS = 8
    RUN_NAME = 'tne_model_run_4'

    train(EPOCHS, DATA_LOADERS_NUM_WORKERS, RUN_NAME)
