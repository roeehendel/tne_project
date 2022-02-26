#! /bin/sh

#SBATCH --job-name=awesome
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --partition=studentbatch
#SBATCH --time=1
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4

python train.py