#!/bin/bash

#SBATCH -o /home/haicu/alaa.bessadok/slurm/logs/slurm_jupyter_%j.job
#SBATCH -e /home/haicu/alaa.bessadok/slurm/logs/slurm_jupyter_%j.job
#SBATCH -J jupyter
#SBATCH -p gpu_p
#SBATCH --qos gpu
#SBATCH --exclude=icb-gpusrv01 # in case you want to exclude individual nodes from your job submission
#SBATCH --gres=gpu:1
#SBATCH -c 6 #CPU cores required
#SBATCH --mem=16G #Memory required
#SBATCH -t 04:00:00 #Job runtime
#SBATCH --nice=10000 #Manual priority. Do not change this.

source ./run_train_script.sh

