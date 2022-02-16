import json

import numpy as np
from tqdm import tqdm

from data_loading.tne_data_paths import TRAIN_DATASET
from data_loading.tne_dataset import PREPOSITION_LIST, NUM_PREPOSITIONS

TNE_CLASS_WEIGHTS = np.array([8.58832905e-01, 3.46315044e-02, 2.97061955e-03, 3.21925512e-02,
                              7.32956338e-03, 5.02182032e-03, 4.80661299e-03, 6.51461969e-03,
                              6.60032340e-04, 8.22622898e-03, 2.47118687e-03, 1.18657925e-02,
                              1.00081836e-02, 2.08381368e-04, 4.69853530e-04, 6.29012147e-03,
                              2.46872194e-03, 7.01556925e-04, 3.00550779e-03, 5.49679332e-04,
                              2.05347608e-04, 1.10921838e-04, 2.31703395e-04, 6.04855836e-05,
                              1.66098342e-04])


def calculate_tne_class_weights():
    file_path = TRAIN_DATASET
    with open(file_path, 'r') as f:
        lines = f.readlines()

    data = [json.loads(line) for line in lines]

    amount_in_each_class = np.zeros(NUM_PREPOSITIONS, dtype=int)

    for example in tqdm(data):
        anchor_complement_with_relation = []
        num_nps = len(example['nps'])
        max_non_related_nps = num_nps * (num_nps - 1)
        np_relations = example['np_relations']  # go over all the np relations
        for np_relation in np_relations:
            anchor = np_relation['anchor']
            complement = np_relation['complement']
            if (anchor, complement) not in anchor_complement_with_relation:
                anchor_complement_with_relation.append((anchor, complement))
            index = PREPOSITION_LIST.index(np_relation['preposition'])
            amount_in_each_class[index] += 1
        amount_in_each_class[0] += (max_non_related_nps - len(anchor_complement_with_relation))

    sum_examples = np.sum(amount_in_each_class)
    class_weights = np.true_divide(amount_in_each_class, sum_examples)

    return class_weights
