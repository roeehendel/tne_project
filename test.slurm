#!/bin/bash -l

#SBATCH --job-name=tne_project_experiments
#SBATCH --output=test.out
#SBATCH --error=test.err
#SBATCH --partition=studentbatch
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1

nvidia-smi
python t.py