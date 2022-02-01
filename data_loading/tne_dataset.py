import gzip
import json

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TNEDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_len = max_length

        with gzip.open(file_path, 'rb') as f:
            lines = f.readlines()

        self.data = [json.loads(line) for line in lines]

        # TODO: remove this to use all data!!!
        self.data = self.data[:16]

        texts = [item['text'] for item in self.data]
        self.encoded_inputs = self.tokenizer(texts, max_length=max_length, padding='max_length')
        # TODO: use the char_to_token method of the tokenizer to get the token ids of the nps start and end

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: dict = self.data[idx]

        ids = self.encoded_inputs['input_ids'][idx]
        mask = self.encoded_inputs['attention_mask'][idx]
        token_type_ids = self.encoded_inputs["token_type_ids"][idx]

        target = len(item['nps']) > 35

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }
