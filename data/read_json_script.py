import json
import matplotlib.pyplot as plt
import numpy as np

preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']


def open_file_json(json_path):
    # Opening JSON file
    input_file = json_path
    with open(input_file, 'r', encoding='latin-1') as f:
        lines = f.readlines()
    f.close()
    return lines

"""
SCRIPT FOR CALC NUMBER OF TUPLES THAT HAVE MORE THAN 1 RELATION.
WE FOUND THAT THERE ARE ONLY TUPLES WITH 0/1/2 RELATIONS
"""
def get_numer_of_tuples_with_more_than_one_relation(json_path):
    lines = open_file_json(json_path)
    total_nps = 0
    greater_than_one_nps = 0
    for i in range(len(lines)):  # go over all the examples
        json_object = json.loads(lines[i])
        anchoer_complement_tuples_dict = dict()
        np_relations = json_object['np_relations']  # go over all the np relations
        for np_relation in np_relations:
            anchor = np_relation['anchor']
            complement = np_relation['complement']
            if (anchor, complement) not in anchoer_complement_tuples_dict:
                anchoer_complement_tuples_dict[(anchor, complement)] = 1
            else:
                anchoer_complement_tuples_dict[(anchor, complement)] += 1
        total_nps += len(anchoer_complement_tuples_dict)
        values = anchoer_complement_tuples_dict.values()
        greater_than_one_nps += len([val for val in values if val > 1])
    print(greater_than_one_nps)
    print(total_nps)
    return greater_than_one_nps, total_nps
"""
SCRIPT FOR CALC WEIGHTS IN EACH CLASS
"""


def get_classes_weights(json_path):
    lines = open_file_json(json_path)
    amount_in_each_class = np.zeros(len(preposition_list), dtype=int)

    for i in range(len(lines)):  # go over all the examples
        json_object = json.loads(lines[i])
        anchor_complement_with_relation = []
        num_nps = len(json_object['nps'])
        max_non_related_nps = num_nps * (num_nps - 1)
        np_relations = json_object['np_relations']  # go over all the np relations
        for np_relation in np_relations:
            anchor = np_relation['anchor']
            complement = np_relation['complement']
            if (anchor, complement) not in anchor_complement_with_relation:
                anchor_complement_with_relation.append((anchor, complement))
            index = preposition_list.index(np_relation['preposition'])
            amount_in_each_class[index] += 1
        amount_in_each_class[0] += (max_non_related_nps - len(anchor_complement_with_relation))

    sum_examples = np.sum(amount_in_each_class)
    weights_class = np.true_divide(amount_in_each_class, sum_examples)
    # print(weights_class)
    return weights_class

"""
SCRIPT FOR CALC MAX LEN OF NP
"""


def get_max_length_np_from_json(json_path: str):
    lines = open_file_json(json_path)
    max_len = 0
    for i in range(len(lines)):  # go over all the examples
        json_object = json.loads(lines[i])
        nps = json_object['nps']
        for key_np, dict_np in nps.items():
            current_len = dict_np['last_token'] - dict_np['first_token'] + 1
            max_len = max(max_len, current_len)
    return max_len

"""
SCRIPT FOR CALC MAX LEN OF NP
"""

def create_histogram_length_nps_from_json(json_path: str):
    lines = open_file_json(json_path)
    nps_len = []
    for i in range(len(lines)):  # go over all the examples
        json_object = json.loads(lines[i])
        nps = json_object['nps']
        for key_np, dict_np in nps.items():
            current_len = dict_np['last_token'] - dict_np['first_token'] + 1
            nps_len.append(current_len)
    # plt.hist(nps_len, 28)
    # plt.show()
    return (np.array(nps_len) > 10).mean()

# max_dev_train = max(get_max_length_np_from_json('train.jsonl'), get_max_length_np_from_json('dev.jsonl'))
# print(create_histogram_length_nps_from_json('train.jsonl'))
# print(create_histogram_length_nps_from_json('dev.jsonl'))

"""
SCRIPT FOR CREATE COREFS TARGETS
"""
def create_corefs_labels(json_path : str, max_nps: int, ignore_index: int):
    lines = open_file_json(json_path)
    for i in range(len(lines)):
        item = json.loads(lines[i])

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

# create_corefs_labels('dev.jsonl', 50, -100)
# get_numer_of_tuples_with_more_than_one_relation('train.jsonl')

