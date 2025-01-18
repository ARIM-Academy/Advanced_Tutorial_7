# Microplastics_UNet: Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning. 

This repository shared the source code for the paper *"Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning"* published in the journal 
Science of the Total Environment. Four models were adapted for microplastics semantic segmentation, instance segmentation and shape classification.  Codes were adapted from the following repositories:

- U-Net: [https://github.com/zhixuhao/unet](https://github.com/zhixuhao/unet) semantic segmentation
- MultiResUNet: [https://github.com/nibtehaz/MultiResUNet](https://github.com/nibtehaz/MultiResUNet) semantic segmentation
- VGG-16: VGG16 with ImageNet weights: shape classification
- Pixel-embedding U-Net: [https://github.com/looooongChen/instance_segmentation_with_pixel_embeddings](https://github.com/looooongChen/instance_segmentation_with_pixel_embeddings) instance segmentation

<img src='imgs/pipeline.png' />

## Requirements
- python 3.6/3.7
- tensorflow_gpu 1.14.1
- please check example_scripts for other libraries


## Data

A manually labelled SEM dataset of microplastics was built. Image segmentation and shape classification were performed on 3 classes: fibres, beads and fragments as shown in the paper [https://doi.org/10.1016/j.scitotenv.2022.153903](https://doi.org/10.1016/j.scitotenv.2022.153903). Datasets were uploaded to Mendeley Data [https://data.mendeley.com/datasets/z6459vntbr/1](https://data.mendeley.com/datasets/z6459vntbr/1) (will soon be updated to https://data.mendeley.com/datasets/z6459vntbr/2)


## Models 

### Semantic Segmentation
Training U-Net/MultiResUNet
```
python main.py
```

The hyperparameters can be changed within main.py.  For example, this line controls training/testing of U-Net or MultiResUNet. The splitfolder was used to performed 5-fold cross validation. 
```
mode = 'training' ## 'multiresunet_training', 'testing', 'multiresunet_testing'
```


### Instance Segmentation
Some modifications had been applied: such as post-processing methods and improved loss function.

Training phase
```
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
python3 main.py --phase=train\
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
```

Prediction phase
```
python3 main.py --phase=prediction \
	--test_dir=fibre/fibre_cc/cc/test/image --test_res=$model_dir \
	--model_dir=$model_dir  \
	--dist_branch="$dist_branch"  --include_bg="$include_bg" \
	--embedding_dim="$embedding_dim" --train_dir="$train_dir" \
	--validation="$validation" \
	--val_dir="$val_dir" \
	--image_depth="$image_depth" \
	--image_channels="$image_channels" 
```

### Shape Classification

Run 'classification.py' with 5-fold cross-validation. Pre-trained convolutional layers using the ImageNet weights. Retrained the last fewer layers of the model with fine-tuning.
```
conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)
```

## Citation

Cite our paper using the following bibtex item:
```
@article{shi2022automatic,
  title={Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning},
  author={Shi, Bin and Patel, Medhavi and Yu, Dian and Yan, Jihui and Li, Zhengyu and Petriw, David and Pruyn, Thomas and Smyth, Kelsey and Passeport, Elodie and Miller, RJ Dwayne and others},
  journal={Science of The Total Environment},
  volume={825},
  pages={153903},
  year={2022},
  publisher={Elsevier}
}
```
Or: <br>
```
Shi, Bin, et al. "Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning." Science of The Total Environment 825 (2022): 153903
```


## Acknowledgements

The project is supported by the WaterSeed Fund from Institute for Water Innovation, University of Toronto and by the Natural Sciences and Engineering Research Council of Canada (NSERC)'s Discovery Grant. Electron microscopy was performed at the Open Centre for the Characterization of Advanced Materials (OCCAM), funded by the Canada Foundation for Innovation. This research was enabled in part by support provided by Compute Canada. Welcome to contact Professor Jane Y. Howe and  R.J. Dwayne Miller for further collaboration. 

## To-do lists
- Pre-trained weights
