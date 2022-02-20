import os

import torch
import wandb
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from config import IGNORE_INDEX, CHECKPOINTS_ROOT_DIR
from data_loading.tne_data_module import TNEDataModule
from models.default_model_architecture import DEFAULT_ARCHITECTURE_CONFIG
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
        learning_rate=1e-4,  # 1e-5
        batch_size=32,
        loss_weight_power=0,
        model_architecture=DEFAULT_ARCHITECTURE_CONFIG
    )

    wandb.init(project='TNE', config=hyperparameter_defaults)

    run_dir_path = os.path.join(CHECKPOINTS_ROOT_DIR, wandb.run.name)

    wandb_logger = WandbLogger()

    model = TNEModel(
        ignore_index=IGNORE_INDEX,
        learning_rate=wandb.config.learning_rate,
        loss_weight_power=wandb.config.loss_weight_power,
        architecture_config=wandb.config.model_architecture
    )
    tne_data_module = TNEDataModule(model.tokenizer, batch_size=wandb.config.batch_size,
                                    ignore_index=IGNORE_INDEX, num_workers=0)

    monitor_metric = "dev/links/f1"
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=run_dir_path,
        filename='tne-model-{epoch:02d}-{dev/links/f1:.2f}',
        save_top_k=2,
        mode="max",
        save_last=False,
        every_n_epochs=1
    )
    save_top_k_callback = SaveTopKModelsCallback(checkpoint_callback, checkpoints_directory=run_dir_path)

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

    trainer_kwargs = {
        'max_epochs': wandb.config.max_epochs,
        'gpus': torch.cuda.device_count(),
        'logger': wandb_logger,
        'log_every_n_steps': 50,
        'callbacks': callbacks,
        # 'accumulate_grad_batches': 8
    }

    if torch.cuda.device_count() > 0:
        trainer_kwargs.update({
            'precision': 16,
            'amp_backend': "native"
        })

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(model, tne_data_module)


if __name__ == '__main__':
    train()
