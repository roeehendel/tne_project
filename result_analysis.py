import wandb


def get_metric_history(run, metric: str):
    return [row[metric] for row in run.scan_history(keys=[metric])]


LINKS_F1_METRIC = 'dev/links/f1_epoch'
PREPOSITIONS_F1_METRIC = 'dev/prepositions/custom_f1_epoch'

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("roeehendel/TNE", filters={'group': 'experiments_1'})

runs_results = dict()
for run in runs:
    runs_results[run.name] = dict(
        links_f1=get_metric_history(run, LINKS_F1_METRIC),
        prepositions_f1=get_metric_history(run, PREPOSITIONS_F1_METRIC)
    )

print(runs_results)

# runs_df = pd.DataFrame({
#     "summary": summary_list,
#     "config": config_list,
#     "name": name_list
#     })
