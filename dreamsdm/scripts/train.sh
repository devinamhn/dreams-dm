#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --constrain=a100-80gb&sxm4
#SBATCH --job-name=train
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --nodes=1

#SBATCH --output=/mnt/home/dmohan/dreams-dm/out-slurm_%A_%a.out


echo "Running on:"
hostname

module load python gcc
module load cudnn

cd /mnt/home/dmohan/dreams-dm/
# cp /mnt/ceph/users/dmohan/dreams/data/dreams/mwzooms_small_train.hdf5 /tmp

source /mnt/home/dmohan/dreams-dm/venv/venvn/bin/activate
python /mnt/home/dmohan/dreams-dm/dreamsdm/test.py

# rm /tmp/mwzooms_small_train.hdf5