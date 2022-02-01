import gzip
import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TNEDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_len: int):
        with gzip.open(file_path, 'rb') as f:
            lines = f.readlines()

        self.data = [json.loads(line) for line in lines]

        # TODO: remove this to use all data!!!
        self.data = self.data[:256]

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: dict = self.data[idx]

        inputs = self.tokenizer.encode_plus(
            item['text'],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        target = len(item['nps']) > 35

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }
