import gzip
import json

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']

"""
:param
len_nps - amount of nps in the example
len_nps - list of dicts. relevant keys in each dict contains: 'anchor', 'complement', 'preposition' 
          (there is 'complement_coref_cluster_id')
"""


def create_target(len_nps, np_relations, ignore_index=-100):
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

        target = create_target(len(item['nps']), item['np_relations'])

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
        }
