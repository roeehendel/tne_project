import os

import wandb
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


def train():
    initialize(0)
    hyperparameter_defaults = dict(
        learning_rate=1e-4,
        dropout_p=0.2,
        loss_weight_power=0.5
    )

    wandb.init(project='TNE', config=hyperparameter_defaults)

    sweep_iteration_dir_path = os.path.join(CHECKPOINTS_ROOT_DIR, wandb.run.name)

    wandb_logger = WandbLogger()

    monitor_metric = "dev/links/f1"

    model = BaseTNEModel(
        ignore_index=IGNORE_INDEX,
        learning_rate=wandb.config.learning_rate,
        dropout_p=wandb.config.dropout_p,
        loss_weight_power=wandb.config.loss_weight_power
    )
    tne_data_module = TNEDataModule(model.tokenizer, ignore_index=IGNORE_INDEX, num_workers=0)

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=sweep_iteration_dir_path,
        filename='tne-model-{epoch:02d}-{dev_prepositions_f1:.2f}',
        save_top_k=2,
        mode="max",
        save_last=True,
        auto_insert_metric_name=True
    )
    save_top_k_callback = SaveTopKModelsCallback(checkpoint_callback)

    # early_stop_callback = EarlyStopping(
    #     monitor=monitor_metric,
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode="max"
    # )

    trainer = Trainer(
        max_epochs=100,
        gpus=1,
        logger=wandb_logger,
        log_every_n_steps=50,
        callbacks=[checkpoint_callback, save_top_k_callback],
        profiler='simple'
    )
    trainer.fit(model, tne_data_module)


if __name__ == '__main__':
    train()
