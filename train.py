import os

import wandb
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from config import CHECKPOINTS_ROOT_DIR, IGNORE_INDEX
from data_loading.tne_data_module import TNEDataModule
from models.tne_model import TNEModel
from utils.initialization import initialize


class SaveTopKModelsCallback(Callback):
    def __init__(self, checkpoint_callback: ModelCheckpoint, checkpoints_directory: str):
        self.checkpoint_callback = checkpoint_callback
        self.checkpoints_directory = checkpoints_directory

    def on_validation_epoch_end(self, trainer, pl_module):
        if os.path.isdir(self.checkpoints_directory):
            self.checkpoint_callback.to_yaml()


def train():
    initialize(0)
    hyperparameter_defaults = dict(
        max_epochs=100,
        learning_rate=3e-4,
        batch_size=32,
        loss_weight_power=0.32,
        num_layers_to_freeze=8,
        num_layers_to_reinitialize=1,
        # pretrained_model_name = 'distilroberta-base'
        pretrained_model_name='roberta-base',
        # pretrained_model_name='roberta-large',
    )

    wandb.init(project='TNE', config=hyperparameter_defaults)

    sweep_iteration_dir_path = os.path.join(CHECKPOINTS_ROOT_DIR, wandb.run.name)

    wandb_logger = WandbLogger()

    model = TNEModel(
        ignore_index=IGNORE_INDEX,
        learning_rate=wandb.config.learning_rate,
        pretrained_model_name=wandb.config.pretrained_model_name,
        loss_weight_power=wandb.config.loss_weight_power,
        num_layers_to_freeze=wandb.config.num_layers_to_freeze,
        num_layers_to_reinitialize=wandb.config.num_layers_to_reinitialize
    )
    tne_data_module = TNEDataModule(model.tokenizer, batch_size=wandb.config.batch_size,
                                    ignore_index=IGNORE_INDEX, num_workers=4)

    monitor_metric = "dev/links/f1"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=sweep_iteration_dir_path,
        filename='tne-model-{epoch:02d}-{dev/links/f1:.2f}',
        save_top_k=2,
        mode="max",
        save_last=False,
        # auto_insert_metric_name=True,
        every_n_epochs=1
    )
    save_top_k_callback = SaveTopKModelsCallback(checkpoint_callback, checkpoints_directory=sweep_iteration_dir_path)

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="max"
    )

    callbacks = [
        checkpoint_callback, save_top_k_callback,
        # early_stop_callback
    ]

    trainer = Trainer(
        max_epochs=wandb.config.max_epochs,
        gpus=1,
        logger=wandb_logger,
        log_every_n_steps=50,
        callbacks=callbacks,
        precision=16, amp_backend="native"
        # profiler='simple'
    )
    trainer.fit(model, tne_data_module)


if __name__ == '__main__':
    train()
