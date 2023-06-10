""" Script to convert the model from mlflow format to a format suitable for release (.pth).

All the following scripts will use the .pth format (easly shared).
"""
import argparse
from pathlib import Path

import mlflow.pytorch
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1_mlflow_path", help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--diffusion_mlflow_path", help="Path to the MLFlow artifact of the diffusion model.")
    parser.add_argument("--controlnet_mlflow_path", help="Path to the MLFlow artifact of the diffusion model.")
    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.")

    args = parser.parse_args()
    return args


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    stage1_model = mlflow.pytorch.load_model(args.stage1_mlflow_path)
    torch.save(stage1_model.state_dict(), output_dir / "autoencoder.pth")

    diffusion_model = mlflow.pytorch.load_model(args.diffusion_mlflow_path)
    torch.save(diffusion_model.state_dict(), output_dir / "diffusion_model.pth")

    controlnet_model = mlflow.pytorch.load_model(args.controlnet_mlflow_path)
    torch.save(controlnet_model.state_dict(), output_dir / "controlnet_model.pth")


if __name__ == "__main__":
    args = parse_args()
    main(args)
