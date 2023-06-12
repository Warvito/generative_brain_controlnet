# ControlNet for Brain T1w Images Generation from FLAIR images using MONAI Generative Models

Script to train a ControlNet (from [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543))
on UK BIOBANK dataset to transform FLAIRs to T1w 2D images using [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
package.

This repository is part of this Medium post.

## Instructions
### Preprocessing
After Downloading UK Biobank dataset and organising it following the BIDS format, we obtain the list of the image paths
for the T1w and FLAIR images. For that, we use the following script:

1) `src/python/preprocessing/create_ids.py` -  Create files with datalist for training, validation and test

### Training
After we obtain the paths, we can train the models using similar commands as in the following files (note: This project was
executed on a cluster with RunAI platform):

1) `cluster/runai/training/stage1.sh` - Command to start to execute in the server the training the first stage of the model.
The main python script in for this is the `src/python/training/train_aekl.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.
2) `src/python/training/eda_ldm_scaling_factor.py` - Script to find the best scaling factor for the latent diffusion model.
3) `cluster/runai/training/ldm.sh` - Command to start to execute in the server the training the diffusion model on the latent representation.
The main python script in for this is the `src/python/training/train_ldm.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.
4) `cluster/runai/training/controlnet.sh` - Command to start to execute in the server the training the ControlNet model using the pretrained LDM.
The main python script in for this is the `src/python/training/train_controlnet.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.

These `.sh` files indicates which parameters and configuration file was used for training, as well how the host directories
were mounted in the used Docker container.
