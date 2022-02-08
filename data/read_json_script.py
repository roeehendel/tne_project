import json

preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                    'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                    'inside', 'outside', 'into', 'around']

# Opening JSON file
input_file = 'train.jsonl'
with open(input_file, 'r', encoding='latin-1') as f:
    lines = f.readlines()

f.close()
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