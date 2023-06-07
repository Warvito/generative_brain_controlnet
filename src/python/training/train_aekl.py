""" Training script for the autoencoder with KL regulization. """
import argparse
import warnings
from pathlib import Path

import torch
import torch.optim as optim
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL
from generative.networks.nets.patchgan_discriminator import PatchDiscriminator
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from tensorboardX import SummaryWriter
from training_functions import train_aekl
from util import get_dataloader, log_mlflow

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--run_dir", help="Location of model to resume.")
    parser.add_argument("--training_ids", help="Location of file with training ids.")
    parser.add_argument("--validation_ids", help="Location of file with validation ids.")
    parser.add_argument("--config_file", help="Location of file with validation ids.")
    parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
    parser.add_argument("--n_epochs", type=int, default=25, help="Number of epochs to train.")
    parser.add_argument("--adv_start", type=int, default=25, help="Epoch when the adversarial training starts.")
    parser.add_argument("--eval_freq", type=int, default=10, help="Number of epochs to between evaluations.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of loader workers")
    parser.add_argument("--experiment", help="Mlflow experiment name.")

    args = parser.parse_args()
    return args


def main(args):
    set_determinism(seed=args.seed)
    print_config()

    output_dir = Path("/project/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = output_dir / args.run_dir
    if run_dir.exists() and (run_dir / "checkpoint.pth").exists():
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    print("Getting data...")
    cache_dir = output_dir / "cached_data_aekl"
    cache_dir.mkdir(exist_ok=True)

    train_loader, val_loader = get_dataloader(
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        training_ids=args.training_ids,
        validation_ids=args.validation_ids,
        num_workers=args.num_workers,
        model_type="autoencoder",
    )

    print("Creating model...")
    config = OmegaConf.load(args.config_file)
    model = AutoencoderKL(**config["stage1"]["params"])
    discriminator = PatchDiscriminator(**config["discriminator"]["params"])
    perceptual_loss = PerceptualLoss(**config["perceptual_network"]["params"])

    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        discriminator = torch.nn.DataParallel(discriminator)
        perceptual_loss = torch.nn.DataParallel(perceptual_loss)

    model = model.to(device)
    perceptual_loss = perceptual_loss.to(device)
    discriminator = discriminator.to(device)

    # Optimizers
    optimizer_g = optim.Adam(model.parameters(), lr=config["stage1"]["base_lr"])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["stage1"]["disc_lr"])

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0
    if resume:
        print(f"Using checkpoint!")
        checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        model.load_state_dict(checkpoint["state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        print(f"No checkpoint found.")

    # Train model
    print(f"Starting Training")
    val_loss = train_aekl(
        model=model,
        discriminator=discriminator,
        perceptual_loss=perceptual_loss,
        start_epoch=start_epoch,
        best_loss=best_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer_g=optimizer_g,
        optimizer_d=optimizer_d,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        writer_train=writer_train,
        writer_val=writer_val,
        device=device,
        run_dir=run_dir,
        kl_weight=config["stage1"]["kl_weight"],
        adv_weight=config["stage1"]["adv_weight"],
        perceptual_weight=config["stage1"]["perceptual_weight"],
        adv_start=args.adv_start,
    )

    log_mlflow(
        model=model,
        config=config,
        args=args,
        experiment=args.experiment,
        run_dir=run_dir,
        val_loss=val_loss,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
