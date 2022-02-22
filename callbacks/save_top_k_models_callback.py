import os

from pytorch_lightning import Callback
from pytorch_lightning.callbacks import ModelCheckpoint


class SaveTopKModelsCallback(Callback):
    def __init__(self, checkpoint_callback: ModelCheckpoint, checkpoints_directory: str):
        self.checkpoint_callback = checkpoint_callback
        self.checkpoints_directory = checkpoints_directory

    def on_validation_epoch_end(self, trainer, pl_module):
        if os.path.isdir(self.checkpoints_directory):
            self.checkpoint_callback.to_yaml()