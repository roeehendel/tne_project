import pickle

import numpy as np
import wandb
from tqdm import tqdm


def get_metric_history(run, metric: str):
    return [row[metric] for row in run.scan_history(keys=[metric])]


LINKS_F1_METRIC = 'dev/links/f1_epoch'
PREPOSITIONS_F1_METRIC = 'dev/prepositions/custom_f1_epoch'


def load_metrics():
    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs("roeehendel/TNE", filters={'group': 'experiments_5'})

    runs_results = dict()
    for run in tqdm(runs):
        runs_results[run.name] = dict(
            links_f1=get_metric_history(run, LINKS_F1_METRIC),
            prepositions_f1=get_metric_history(run, PREPOSITIONS_F1_METRIC)
        )

    run_metrics = [(run_name,
                    np.mean(run_results['prepositions_f1'][-10:]) * 100,
                    np.mean(run_results['links_f1'][-10:]) * 100)
                   for run_name, run_results in runs_results.items()]
    run_metrics = sorted(run_metrics, key=lambda x: x[1], reverse=True)

    # save results to pickle

    with open('results.pkl', 'wb') as f:
        pickle.dump(run_metrics, f)


load_metrics()

# load results from pickle
with open('results.pkl', 'rb') as f:
    run_metrics = pickle.load(f)

# print the metrics in latex table format
for run_name, prepositions_f1, links_f1 in run_metrics:
    run_name.replace('-base', '')
    run_name.replace('advnaced-', 'A &')
    run_name.replace('roberta-', 'rob &')
    run_name.replace('spanbert-', 'spa &')
    print(f'{run_name} & {prepositions_f1:.2f} & {links_f1:.2f} \\\\')
