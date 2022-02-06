import gzip
import json

import numpy as np
import torch
from tokenizers import Encoding
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast

preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']


def create_target(item: dict, ignore_index: int = -100):
    len_nps = len(item['nps'])
    np_relations = item['np_relations']

    # creating target vec of k^2
    target = np.zeros(len_nps ** 2)
    # go over all the np relations and update target
    for np_relation in np_relations:
        preposition_index = preposition_list.index(np_relation['preposition'])
        anchor_index = int(np_relation['anchor'][2:])
        complement_index = int(np_relation['complement'][2:])
        target[anchor_index * len_nps + complement_index] = preposition_index
    # update target where anchor and complement are same to -100 to ignore them
    indexes_anchor_complement_are_same = (len_nps + 1) * np.arange(len_nps)
    target[indexes_anchor_complement_are_same] = -ignore_index
    return target


def create_nps(item: dict, encoding: Encoding):
    raw_nps = [item['nps'][f'np{i}'] for i in range(len(item['nps']))]
    nps_characters = [(np['first_char'], np['last_char'] - 1) for np in raw_nps]
    tokens_idx = [tuple(map(encoding.char_to_token, characters)) for characters in nps_characters]
    tokens_idx = [(idx_start - 1, idx_end - 1) for idx_start, idx_end in tokens_idx]
    return tokens_idx


class TNEDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerFast, max_length: int):
        self.tokenizer = tokenizer
        self.max_len = max_length

        with gzip.open(file_path, 'rb') as f:
            lines = f.readlines()

        self.data = [json.loads(line) for line in lines]

        # TODO: remove this to use all data!!!
        self.data = self.data[:16]

        texts = [item['text'] for item in self.data]
        self.encodings = self.tokenizer(texts, max_length=max_length, padding='max_length')

        self.targets = [create_target(item) for item in self.data]
        self.nps = [create_nps(item, self.encodings[i]) for i, item in enumerate(self.data)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: dict = self.data[idx]

        ids = self.encodings['input_ids'][idx]
        mask = self.encodings['attention_mask'][idx]
        token_type_ids = self.encodings["token_type_ids"][idx]
        nps = self.nps[idx]
        target = self.targets[idx]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'nps': nps,
            'targets': torch.tensor(target, dtype=torch.long)
        }
