#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constrain=ib-h100p
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
python $1 $2 #python script, config file
# python /mnt/home/dmohan/dreams-dm/dreamsdm/test.py '/mnt/home/dmohan/dreams-dm/dreamsdm/config_dmmaps.yaml'

# rm /tmp/mwzooms_small_train.hdf5