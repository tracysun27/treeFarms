#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --time=48:00:00
#SBATCH --mem=300GB
#SBATCH --output=./jupyter_log.log
source /home/users/vb97/anaconda3/bin/activate
export PATH=$PATH:/home/users/vb97/.local/bin
jupyter-lab --ip=0.0.0.0 --port=8889 --no-browser
