from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from data_loading.tne_data_paths import TRAIN_DATASET, DEV_DATASET, TEST_DATASET
from data_loading.tne_dataset import TNEDataset


class TNEDataModule(LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, batch_size: int,
                 num_workers: int = 0, max_len: int = 512, max_nps: int = 60, ignore_index: int = -100):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len
        self.max_nps = max_nps
        self.ignore_index = ignore_index
        self.num_workers = num_workers
        self.train = None
        self.dev = None
        self.test = None

    def prepare_data(self):
        pass
        # called only on 1 GPU
        # download_dataset()
        # tokenize()
        # build_vocab()

    def load_datasets(self):
        datasets = [
            (TRAIN_DATASET, True),
            (DEV_DATASET, True),
            (TEST_DATASET, False)
        ]
        return (TNEDataset(path, self.tokenizer, self.max_len, self.max_nps, self.ignore_index, has_targets)
                for path, has_targets in datasets)

    def setup(self, stage: Optional[str] = None):
        self.train, self.dev, self.test = self.load_datasets()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)