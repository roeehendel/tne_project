import os
from dataclasses import asdict
from typing import Optional

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from callbacks.save_top_k_models_callback import SaveTopKModelsCallback
from config import IGNORE_INDEX, CHECKPOINTS_ROOT_DIR
from data_loading.tne_data_module import TNEDataModule
from lightning.tne_lightning_module import TNELightningModule
from models.architecture_configurations import ARCHITECTURE_CONFIGURATIONS
from utils.initialization import initialize


def train(hyperparameters: dict, group_name: Optional[str] = None, experiment_name: Optional[str] = None):
    initialize(0)

    wandb_init_kwargs = {'project': 'TNE', 'config': hyperparameters}
    if group_name is not None:
        wandb_init_kwargs['group'] = group_name
    if experiment_name is not None:
        wandb_init_kwargs['name'] = experiment_name

    wandb.init(**wandb_init_kwargs)

    run_dir_path = os.path.join(CHECKPOINTS_ROOT_DIR, wandb.run.name)

    wandb_logger = WandbLogger()
    lightning_module = TNELightningModule(
        architecture_config=wandb.config.architecture_configuration,
        ignore_index=IGNORE_INDEX,
        learning_rate=wandb.config.learning_rate,
        loss_weight_power=wandb.config.loss_weight_power,
        batch_size=wandb.config.batch_size,
        max_epochs=wandb.config.max_epochs
    )
    tne_data_module = TNEDataModule(lightning_module.tokenizer, batch_size=wandb.config.batch_size,
                                    ignore_index=IGNORE_INDEX, num_workers=0)

    monitor_metric = TNELightningModule.metric_full_name('dev', 'prepositions', 'custom_f1_epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=run_dir_path,
        filename='tne-model-{epoch:02d}-{' + monitor_metric + ':.2f}',
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

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = [
        lr_monitor,
        # checkpoint_callback, save_top_k_callback,
        # early_stop_callback
    ]

    trainer_kwargs = {
        'max_epochs': wandb.config.max_epochs,
        'logger': wandb_logger,
        'log_every_n_steps': 50,
        'callbacks': callbacks,
        # 'accumulate_grad_batches': 8
    }

    if torch.cuda.device_count() > 0:
        if torch.cuda.device_count() > 1:
            # trainer_kwargs['strategy'] = 'ddp'
            # trainer_kwargs['gpus'] =  [1, 2, 3]
            trainer_kwargs['gpus'] = [torch.cuda.device_count() - 1]
        else:
            trainer_kwargs['gpus'] = 1

        trainer_kwargs.update({
            'precision': 16,
            'amp_backend': "native"
        })

    trainer = Trainer(**trainer_kwargs)
    trainer.fit(lightning_module, tne_data_module)


if __name__ == '__main__':
    # architecture_configuration = list(ARCHITECTURE_CONFIGURATIONS.values())[-1]
    experiment_name = 'basic-spanbert-base'
    architecture_configuration = ARCHITECTURE_CONFIGURATIONS[experiment_name]

    hyperparameters = dict(
        max_epochs=20,
        learning_rate=5e-5,  # 1e-5
        batch_size=16,
        loss_weight_power=0.0,
        architecture_configuration=asdict(architecture_configuration)
    )

    train(hyperparameters, group_name='manual', experiment_name=experiment_name)
