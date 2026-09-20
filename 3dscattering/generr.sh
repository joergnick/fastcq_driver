#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=4
#SBATCH --time=8:00:00
#SBATCH --job-name="small"
#SBATCH --mem-per-cpu=32GB

python3 create_error_data.py
