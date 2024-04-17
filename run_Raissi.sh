#!/bin/bash
#SBATCH --job-name=out
#SBATCH --output=out_%j.out
#SBATCH --error=out_%j.err
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=32         # Number of tasks per node
#SBATCH --mem=100G                   # Memory per node
#SBATCH --time=00:10:00              # Time (hh:mm:ss)

# Execute the Python script
python __main__.py
