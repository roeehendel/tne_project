#!/bin/bash -l

#SBATCH --job-name=tne_project_experiments
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --partition=studentbatch
#SBATCH --time=1
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4

python train.py