#!/bin/bash
#SBATCH --job-name=python_job
#SBATCH --output=python_job_%j.out
#SBATCH --error=python_job_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=4G

# Navigate to the directory containing the script and modules
cd /Raissi

# Execute the Python script
python __main__.py
