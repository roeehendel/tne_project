import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding

from data_loading.tne_data_decompression import decompress_tne_dataset

PREPOSITION_LIST = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']
NUM_PREPOSITIONS = len(PREPOSITION_LIST)


def create_target(item: dict, max_nps: int, ignore_index: int):
    len_nps = len(item['nps'])
    np_relations = item['np_relations']

    target = np.ones((max_nps, max_nps)) * ignore_index
    target[:len_nps, :len_nps] = 0

    # go over all the np relations and update target
    for np_relation in np_relations:
        preposition_index = PREPOSITION_LIST.index(np_relation['preposition'])
        anchor_index = int(np_relation['anchor'][2:])
        complement_index = int(np_relation['complement'][2:])
        target[anchor_index, complement_index] = preposition_index
    # update target where anchor and complement are same to -100 to ignore them
    target[np.arange(len_nps), np.arange(len_nps)] = ignore_index
    return target


def create_nps(item: dict, max_nps: int, encoding: BatchEncoding):
    raw_nps = [item['nps'][f'np{i}'] for i in range(len(item['nps']))]
    nps_characters = [(np['first_char'], np['last_char'] - 1) for np in raw_nps]
    tokens_idx = [list(map(encoding.char_to_token, characters)) for characters in nps_characters]
    tokens_idx = [list(map(lambda x: x if x is not None else 1, idx)) for idx in tokens_idx]  # handle tagging error
    tokens_idx = [[idx_start - 1, idx_end - 1] for idx_start, idx_end in tokens_idx]
    tokens_idx = tokens_idx + [[0, 0] for _ in range(max_nps - len(tokens_idx))]
    return tokens_idx


def create_corefs_labels(item: dict, max_nps: int, ignore_index: int):
    len_nps = len(item['nps'])

    # get all corefs indexes
    corefs = [item['coref'][i]['members'] for i in range(len(item['coref']))]
    corefs_indices = [[int(member[2:]) for member in coref] for coref in corefs]

    corefs_target = np.ones((max_nps, max_nps)) * ignore_index
    corefs_target[:len_nps, :len_nps] = 0
    for coref_indices in corefs_indices:
        for member_index in coref_indices:
            corefs_target[member_index][coref_indices] = 1

    return corefs_target


def create_multi_label_target(item: dict, max_nps: int, ignore_index: int):
    len_nps = len(item['nps'])
    np_relations = item['np_relations']
    multi_label_target = np.ones((max_nps, max_nps, NUM_PREPOSITIONS)) * ignore_index
    multi_label_target[:len_nps, :len_nps, :] = 0

    # go over all the np relations and update multi_label_target
    for np_relation in np_relations:
        preposition_index = PREPOSITION_LIST.index(np_relation['preposition'])
        anchor_index = int(np_relation['anchor'][2:])
        complement_index = int(np_relation['complement'][2:])
        multi_label_target[anchor_index, complement_index, preposition_index] = 1
    # update target where anchor and complement are same to -100 to ignore them
    multi_label_target[np.arange(len_nps), np.arange(len_nps), :] = ignore_index
    return multi_label_target


class TNEDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizerFast,
                 max_length: int, max_nps: int, ignore_index: int, has_targets: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_nps = max_nps

        decompress_tne_dataset()

        with open(file_path, 'rb') as f:
            lines = f.readlines()

        self.data = [json.loads(line) for line in lines]

        # TODO: remove this to use all data!!!
        # self.data = self.data[:16]

        self.has_targets = has_targets
        if self.has_targets:
            self.tne_targets = [create_target(item, self.max_nps, ignore_index) for item in self.data]
            self.coref_targets = [create_corefs_labels(item, self.max_nps, ignore_index) for item in self.data]
            self.tne_multilabel_targets = [create_multi_label_target(item, self.max_nps, ignore_index) for item in
                                           self.data]
        else:
            self.tne_targets = None
            self.coref_targets = None
            self.tne_multilabel_targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item: dict = self.data[idx]

        text = item['text']
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length')
        ids = encoding['input_ids']
        mask = encoding['attention_mask']
        nps = create_nps(item, self.max_nps, encoding)
        num_nps = len(item['nps'])

        item = {'inputs': {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'nps': torch.tensor(nps, dtype=torch.long),
            'num_nps': torch.tensor(num_nps, dtype=torch.long)
        }}

        if self.has_targets:
            tne_targets = self.tne_targets[idx]
            tne_multilabel_targets = self.tne_multilabel_targets[idx]
            coref_targets = self.coref_targets[idx]

            item['targets'] = {
                'tne': torch.tensor(tne_targets, dtype=torch.long),
                'tne_multilabel': torch.tensor(tne_multilabel_targets, dtype=torch.long),
                'coref': torch.tensor(coref_targets, dtype=torch.long),
            }
        return item
