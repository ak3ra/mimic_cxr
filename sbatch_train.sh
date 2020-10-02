#!/bin/bash
#SBATCH --account=rpp-bengioy            # Yoshua pays for your job
#SBATCH --cpus-per-task=6                # Ask for 6 CPUs
#SBATCH --gres=gpu:1                     # Ask for 1 GPU
#SBATCH --mem=128G                        # Ask for 32 GB of RAM
#SBATCH --time=14:00:00                   # The job will run for 13 hours
#SBATCH -o /home/akera/logs/mimic_train-%j.out  # Write the log in $SCRATCH
#SBATCH -e /home/akera/logs/mimic_train-%j.err  # Write the err in $SCRATCH

module load python/3.6
source /home/akera/glacier_env/bin/activate
python src/train.py
