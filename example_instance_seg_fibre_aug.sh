#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --account=def-janehowe
#SBATCH --cpus-per-task=3  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=16000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-04:00     # DD-HH:MM:SS

module load  StdEnv/2018 
module load python/3.6 
module load cuda 
# module load cudnn 

SOURCEDIR=~/projects/def-janehowe/shibin2

# Prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/env2
source $SLURM_TMPDIR/env2/bin/activate
pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index scipy==1.2.1
pip install --no-index Pillow
pip install --no-index cython
pip install --no-index matplotlib
pip install --no-index scikit_image==0.16.2
pip install --no-index tensorflow_gpu==1.14.1
pip install --no-index keras==2.1.2
pip install --no-index opencv_python
pip install --no-index h5py==2.10.0
pip install --no-index imgaug
pip install --no-index ipython
pip install --no-index scikit_learn

cp -r $SOURCEDIR/instance_segmentation_with_pixel_embeddings $SLURM_TMPDIR/. 
cd instance_segmentation_with_pixel_embeddings

phase="train"
dist_branch=True
include_bg=True
embedding_dim=16

train_dir="fibre/fibre_cc/cc_all_neighbor/train"
validation=True
val_dir="fibre/fibre_cc/cc_all_neighbor/val"

image_depth="uint8" #"uint16"
image_channels=1
model_dir=$SOURCEDIR/pixel_emb_result7

lr=0.0001
batch_size=4 #4
training_epoches=300 #300

python3 main.py --phase="$phase" \
	--dist_branch="$dist_branch"  --include_bg="$include_bg" \
	--embedding_dim="$embedding_dim" \
	--train_dir="$train_dir" \
	--validation="$validation" \
	--val_dir="$val_dir" \
	--image_depth="$image_depth" \
	--image_channels="$image_channels" \
	--model_dir=$model_dir \
	--lr="$lr" \
	--batch_size="$batch_size" \
	--training_epoches="$training_epoches" \
	--validation_steps=100 \
	--save_steps=2000

python3 main.py --phase=prediction \
	--test_dir=fibre/fibre_cc/cc/test/image --test_res=$model_dir \
	--model_dir=$model_dir  \
	--dist_branch="$dist_branch"  --include_bg="$include_bg" \
	--embedding_dim="$embedding_dim" --train_dir="$train_dir" \
	--validation="$validation" \
	--val_dir="$val_dir" \
	--image_depth="$image_depth" \
	--image_channels="$image_channels" 

