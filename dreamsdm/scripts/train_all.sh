#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constrain=a100-80gb&sxm4
#SBATCH --job-name=train
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1
#SBATCH --output=/mnt/home/dmohan/dreams-dm/logs/out-slurm_%A_%a.out

echo "Running on:"
hostname

module load python gcc
module load cudnn

cd /mnt/home/dmohan/dreams-dm/

source /mnt/home/dmohan/dreams-dm/venv/venvn/bin/activate
python $1 $2 $3 $4 #python script, config file, output path, pred_param {'AGN', 'SN1', 'SN2', 'WDM'}

# example use:
# sbatch train_all.sh scripts/train_combi.py 'configs/combimaps_full.json' '/mnt/ceph/users/dmohan/dreams/results/combinedmaps' 'AGN'