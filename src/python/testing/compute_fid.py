""" Script to compute the Frechet Inception Distance (FID) of the samples of the LDM.

In order to measure the quality of the samples, we use the Frechet Inception Distance (FID) metric between 1200 images
from the MIMIC-CXR dataset and 1000 images from the LDM.
"""
import argparse
from pathlib import Path

import torch
from generative.metrics import FIDMetric
from monai import transforms
from monai.config import print_config
from monai.data import Dataset
from monai.utils import set_determinism
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import get_test_dataloader


def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x


def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)


def get_features(image, radnet):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = radnet.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--sample_dir", help="Location of the samples to evaluate.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    samples_dir = Path(args.sample_dir)

    # Load pretrained model
    device = torch.device("cuda")
    model = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
    model = model.to(device)
    model.eval()

    # Samples
    samples_datalist = []
    for sample_path in sorted(list(samples_dir.glob("*.png"))):
        samples_datalist.append(
            {
                "t1w": str(sample_path),
            }
        )
    print(f"{len(samples_datalist)} images found in {str(samples_dir)}")

    sample_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["t1w"]),
            transforms.EnsureChannelFirstd(keys=["t1w"]),
            transforms.Rotate90d(keys=["t1w"], k=-1, spatial_axes=(0, 1)),  # Fix flipped image read
            transforms.Flipd(keys=["t1w"], spatial_axis=1),  # Fix flipped image read
            transforms.ScaleIntensityRanged(keys=["t1w"], a_min=0.0, a_max=255.0, b_min=0.0, b_max=1.0, clip=True),
            transforms.ToTensord(keys=["t1w"]),
        ]
    )

    samples_ds = Dataset(
        data=samples_datalist,
        transform=sample_transforms,
    )
    samples_loader = DataLoader(
        samples_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
    )

    samples_features = []
    for batch in tqdm(samples_loader):
        img = batch["t1w"]
        with torch.no_grad():
            outputs = get_features(img.to(device), radnet=model)

        samples_features.append(outputs.cpu())
    samples_features = torch.cat(samples_features, dim=0)

    # Test set
    test_loader = get_test_dataloader(
        batch_size=args.batch_size,
        test_ids=args.test_ids,
        num_workers=args.num_workers,
        upper_limit=1000,
    )

    test_features = []
    for batch in tqdm(test_loader):
        img = batch["t1w"]
        with torch.no_grad():
            outputs = get_features(img.to(device), radnet=model)

        test_features.append(outputs.cpu())
    test_features = torch.cat(test_features, dim=0)

    # Compute FID
    metric = FIDMetric()
    fid = metric(samples_features, test_features)

    print(f"FID: {fid:.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
