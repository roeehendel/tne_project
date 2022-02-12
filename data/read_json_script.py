import json

import numpy as np

preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']

# Opening JSON file
input_file = 'train.jsonl'
with open(input_file, 'r', encoding='latin-1') as f:
    lines = f.readlines()

f.close()
"""
SCRIPT FOR CALC NUMBER OF TUPLES THAT HAVE MORE THAN 1 RELATION.
WE FOUND THAT THERE ARE ONLY TUPLES WITH 0/1/2 RELATIONS
"""
# total_nps = 0
# greater_than_one_nps = 0
#
# for i in range(len(lines)):  # go over all the examples
#     json_object = json.loads(lines[i])
#     anchoer_complement_tuples_dict = dict()
#     np_relations = json_object['np_relations']  # go over all the np relations
#     for np_relation in np_relations:
#         anchor = np_relation['anchor']
#         complement = np_relation['complement']
#         if (anchor, complement) not in anchoer_complement_tuples_dict:
#             anchoer_complement_tuples_dict[(anchor, complement)] = 1
#         else:
#             anchoer_complement_tuples_dict[(anchor, complement)] += 1
#     total_nps += len(anchoer_complement_tuples_dict)
#     values = anchoer_complement_tuples_dict.values()
#     greater_than_one_nps += len([val for val in values if val > 1])
#
#
# print(greater_than_one_nps)
# print(total_nps)
"""
SCRIPT FOR CALC WEIGHTS IN EACH CLASS
"""
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
print(weights_class)
