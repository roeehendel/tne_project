import sys
from dataclasses import asdict

from models.architecture_configurations import ARCHITECTURE_CONFIGURATIONS
from train import train


def run_experiment(group_name, experiment_name):
    hyperparameters = dict(
        max_epochs=30,
        learning_rate=1e-4,  # 1e-5
        batch_size=16,
        loss_weight_power=0.25,
        architecture_configuration=asdict(ARCHITECTURE_CONFIGURATIONS[experiment_name])
    )

    train(hyperparameters=hyperparameters, group_name=group_name, experiment_name=experiment_name)


if __name__ == '__main__':
    group_name = 'experiments_5'
    experiment_number = sys.argv[1]
    experiment_name = list(ARCHITECTURE_CONFIGURATIONS.keys())[int(experiment_number)]

    run_experiment(group_name, experiment_name)
