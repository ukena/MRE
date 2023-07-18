#!/bin/bash
#SBATCH --job-name=LSTM-MRE
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=job.%j.out

module purge
module load tools/python/3.8
module load nvidia/cuda/11.7

export PYTHONPATH=$PYTHONPATH:/home/yo76doh/mre
export TF_CPP_MIN_LOG_LEVEL=2  # supress unimportant warnings

python -m pip install --user -r requirements.txt

python -u main.py