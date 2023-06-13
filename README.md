# ControlNet for Brain T1w Images Generation from FLAIR images using MONAI Generative Models

Script to train a ControlNet (from [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543))
on UK BIOBANK dataset to transform FLAIRs to T1w 2D images using [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
package.

![ControlNet Samples](https://github.com/Warvito/generative_brain_controlnet/blob/main/assets/figure_samples.png?raw=true)

This repository is part of this Medium post.

## Instructions
### Preprocessing
After Downloading UK Biobank dataset and preprocessing it, we obtain the list of the image paths for the T1w and FLAIR
images. For that, we use the following script:

1) `src/python/preprocessing/create_png_dataset.py` -  Create png images from the nifti files
2) `src/python/preprocessing/create_ids.py` -  Create files with datalist for training, validation and test

### Training
After we obtain the paths, we can train the models using similar commands as in the following files (note: This project was
executed on a cluster with RunAI platform):

1) `cluster/runai/training/stage1.sh` - Command to start to execute in the server the training the first stage of the model.
The main python script in for this is the `src/python/training/train_aekl.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.
3) `cluster/runai/training/ldm.sh` - Command to start to execute in the server the training the diffusion model on the latent representation.
The main python script in for this is the `src/python/training/train_ldm.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.
4) `cluster/runai/training/controlnet.sh` - Command to start to execute in the server the training the ControlNet model using the pretrained LDM.
The main python script in for this is the `src/python/training/train_controlnet.py` script. The `--volume` flags indicate how the dataset
is mounted in the Docker container.

These `.sh` files indicates which parameters and configuration file was used for training, as well how the host directories
were mounted in the used Docker container.


### Inference and evaluation
Finally, we converted the mlflow model to .pth files (for easily loading with MONAI), sampled images from the diffusion
model and controlnet, and evaluated the models. The following is the list of execution for inference and evaluation:

1) `src/python/testing/convert_mlflow_to_pytorch.py` - Convert mlflow model to .pth files
2) `src/python/testing/sample_t1w.py` - Sample T1w images from the diffusion model without using contditioning.
3) `cluster/runai/testing/sample_flair_to_t1w.py` - Sample T1w images from the controlnet using the test set's FLAIR
images as conditionings.
4) `src/python/testing/compute_msssim_reconstruction.py` - Measure the mean structural similarity index between images and
reconstruction to measure the preformance of the autoencoder.
5) `src/python/testing/compute_msssim_sample.py` - Measure the mean structural similarity index between samples in order
to measure the diversity of the synthetic data.
6) `src/python/testing/compute_fid.py` - Compute FID score between generated images and real images.
7) `src/python/testing/compute_controlnet_performance.py` - Compute the performance of the controlnet using MAE, PSNR and
MS-SSIM metrics.
