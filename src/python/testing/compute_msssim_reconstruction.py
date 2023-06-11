""" Script to compute the MS-SSIM score of the reconstructions of the Autoencoder.

Here we compute the MS-SSIM score between the images of the test set of the MIMIC-CXR dataset and the reconstructions
created byt the AutoencoderKL.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from generative.metrics import MultiScaleSSIMMetric
from generative.networks.nets import AutoencoderKL
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tqdm import tqdm
from util import get_test_dataloader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--output_dir", help="Location to save the output.")
    parser.add_argument("--test_ids", help="Location of file with test ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Testing batch size.")
    parser.add_argument("--config_file", help="Location of config file.")
    parser.add_argument("--stage1_path", help="Location of stage1 model.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("Getting data...")
    test_loader = get_test_dataloader(
        batch_size=args.batch_size,
        test_ids=args.test_ids,
        num_workers=args.num_workers,
    )

    print("Creating model...")
    device = torch.device("cuda")
    config = OmegaConf.load(args.config_file)
    stage1 = AutoencoderKL(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1 = stage1.to(device)
    stage1.eval()

    ms_ssim = MultiScaleSSIMMetric(spatial_dims=2, data_range=1.0, kernel_size=7)

    print("Computing MS-SSIM...")
    ms_ssim_list = []
    filenames = []
    for batch in tqdm(test_loader):
        x = batch["t1w"].to(device)

        with torch.no_grad():
            x_recon = stage1.reconstruct(x)

        ms_ssim_list.append(ms_ssim(x, x_recon))
        filenames.extend(batch["t1w_meta_dict"]["filename_or_obj"])

    ms_ssim_list = torch.cat(ms_ssim_list, dim=0)

    prediction_df = pd.DataFrame({"filename": filenames, "ms_ssim": ms_ssim_list.cpu()[:, 0]})
    prediction_df.to_csv(output_dir / "ms_ssim_reconstruction.tsv", index=False, sep="\t")

    print(f"Mean MS-SSIM: {ms_ssim_list.mean():.6f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
