""" Script to generate sample images from the diffusion model.

In the generation of the images, the script is using a DDIM scheduler.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from monai.config import print_config
from monai.utils import set_determinism
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to save the .pth file of the diffusion model.")
    parser.add_argument("--stage1_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--stage1_config_file_path", help="Path to the .pth model from the stage1.")
    parser.add_argument("--diffusion_config_file_path", help="Path to the .pth model from the diffusion model.")
    parser.add_argument("--start_seed", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--stop_seed", type=int, help="Path to the MLFlow artifact of the stage1.")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="")
    parser.add_argument("--x_size", type=int, default=64, help="Latent space x size.")
    parser.add_argument("--y_size", type=int, default=64, help="Latent space y size.")
    parser.add_argument("--scale_factor", type=float, help="Latent space y size.")
    parser.add_argument("--num_inference_steps", type=int, help="")

    args = parser.parse_args()
    return args


def main(args):
    print_config()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")

    config = OmegaConf.load(args.stage1_config_file_path)
    stage1 = AutoencoderKL(**config["stage1"]["params"])
    stage1.load_state_dict(torch.load(args.stage1_path))
    stage1.to(device)
    stage1.eval()

    config = OmegaConf.load(args.diffusion_config_file_path)
    diffusion = DiffusionModelUNet(**config["ldm"].get("params", dict()))
    diffusion.load_state_dict(torch.load(args.diffusion_path))
    diffusion.to(device)
    diffusion.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=config["ldm"]["scheduler"]["num_train_timesteps"],
        beta_start=config["ldm"]["scheduler"]["beta_start"],
        beta_end=config["ldm"]["scheduler"]["beta_end"],
        schedule=config["ldm"]["scheduler"]["schedule"],
        prediction_type=config["ldm"]["scheduler"]["prediction_type"],
        clip_sample=False,
    )
    scheduler.set_timesteps(args.num_inference_steps)

    tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="text_encoder")

    prompt = ["", "T1-weighted image of a brain."]
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.squeeze(1))
    prompt_embeds = prompt_embeds[0].to(device)

    for i in range(args.start_seed, args.stop_seed):
        set_determinism(seed=i)
        noise = torch.randn((1, config["ldm"]["params"]["in_channels"], args.x_size, args.y_size)).to(device)

        with torch.no_grad():
            progress_bar = tqdm(scheduler.timesteps)
            for t in progress_bar:
                noise_input = torch.cat([noise] * 2)
                model_output = diffusion(
                    noise_input, timesteps=torch.Tensor((t,)).to(noise.device).long(), context=prompt_embeds
                )
                noise_pred_uncond, noise_pred_text = model_output.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                noise, _ = scheduler.step(noise_pred, t, noise)

        with torch.no_grad():
            sample = stage1.decode_stage_2_outputs(noise / args.scale_factor)

        sample = np.clip(sample.cpu().numpy(), 0, 1)
        sample = (sample * 255).astype(np.uint8)
        im = Image.fromarray(sample[0, 0])
        im.save(output_dir / f"sample_{i}.png")


if __name__ == "__main__":
    args = parse_args()
    main(args)
