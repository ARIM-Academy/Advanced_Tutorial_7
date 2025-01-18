#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --account=def-rjdmille
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-06:00     # DD-HH:MM:SS


module load StdEnv/2018.3 python/3.7 cuda/10.1 cudnn gcc
SOURCEDIR=~/projects/def-janehowe/shibin2

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index tensorflow_gpu
pip install --no-index matplotlib
pip install --no-index keras
pip install --no-index numpy
pip install --no-index scikit_image
pip install --no-index $SOURCEDIR/wheels/split_folders-0.4.3-py3-none-any.whl
cp $SOURCEDIR/wheels/split_folders-0.4.3.tar ./
tar -xvf split_folders-0.4.3.tar
cd split_folders-0.4.3
python setup.py install

cp -rf $SOURCEDIR/unet_mp_cc_unet22 unet_mp_cc_unett42
cd unet_mp_cc_unett42/data/microplastics
cp -r ~/scratch/datasets_particle_part2.zip ./
unzip datasets_particle_part2.zip
cd ../..
python main.py


