from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from data_loading.tne_dataset import TNEDataset

TRAIN_DATASET = 'data/train.jsonl.gz'
DEV_DATASET = 'data/dev.jsonl.gz'
TEST_DATASET = 'data/test_unlabeled.jsonl.gz'


class TNEDataModule(LightningDataModule):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_len: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
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
        return (TNEDataset(path, self.tokenizer, self.max_len) for path in [TRAIN_DATASET, DEV_DATASET, TEST_DATASET])

    def setup(self, stage: Optional[str] = None):
        self.train, self.dev, self.test = self.load_datasets()

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev, batch_size=8, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=8, num_workers=8)
