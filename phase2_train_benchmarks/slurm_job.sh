#!/usr/bin/env bash
#SBATCH --job-name=phase2
#SBATCH --mem=128g
#SBATCH --mem-per-cpu=8192

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source activate samp-phase2
python run_all.py
