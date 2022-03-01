#!/bin/bash -l

#SBATCH --job-name=tne_project_experiments
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --partition=studentbatch
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

echo $1
python run_experiment.py $1