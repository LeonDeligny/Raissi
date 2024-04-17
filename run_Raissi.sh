#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=python_job_%j.out
#SBATCH --error=python_job_%j.err
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --gres=gpu:1                 # Request GPU
#SBATCH --ntasks-per-node=32         # Number of tasks per node
#SBATCH --mem=100G                   # Memory per node
#SBATCH --time=00:10:00              # Time (hh:mm:ss)

# Execute the Python script
python __main__.py
