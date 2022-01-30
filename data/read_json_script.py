import json


preposition_list = ['no-relation', 'of', 'against', 'in', 'by', 'on', 'about', 'with', 'after', 'member(s) of',
                          'to', 'from', 'for', 'among', 'under', 'at', 'between', 'during', 'near', 'over', 'before',
                          'inside', 'outside', 'into', 'around']

# Opening JSON file
input_file = 'train.jsonl'
with open(input_file, 'r', encoding='latin-1') as f:
    lines = f.readlines()

f.close()

relations_amount_per_tuple = []

for i in range(len(lines)):  # go over all the examples
    json_object = json.loads(lines[i])
    np_relations = json_object['np_relations']  # go over all the np relations
    for np_relation in np_relations:
        preposition = np_relation['preposition']
        if preposition not in preposition_list:
            relations_amount_per_tuple.append(len(preposition))
        else:  # one word
            relations_amount_per_tuple.append(1)

larger_elements = [element for element in relations_amount_per_tuple if element > 1]
print(f'all tuples: {len(relations_amount_per_tuple)}, more that 1 relation: {len(larger_elements)}')