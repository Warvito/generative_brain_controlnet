""" Script to create png dataset. """
import argparse
from pathlib import Path

import nibabel as nib
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=int, help="Starting index")
    parser.add_argument("--stop", type=int, help="Stoping index")

    args = parser.parse_args()
    return args


def main(args):
    new_data_dir = Path("/data/")
    data_dir = Path("/source_data/")
    images_paths = sorted(list(data_dir.glob("**/*T1w.nii.gz")))
    images_paths = images_paths[args.start : args.stop]

    for image_path in tqdm(images_paths):
        if "unusable" not in str(image_path):
            flair_path = Path(str(image_path).replace("T1w", "FLAIR"))
            if flair_path.exists():
                new_sub_dir = new_data_dir / str(image_path.parents[1])[13:]
                new_sub_dir.mkdir(parents=True, exist_ok=True)

                t1w = nib.load(image_path)
                flair = nib.load(flair_path)

                t1w = t1w.get_fdata()
                flair = flair.get_fdata()

                t1w = t1w[16:176, 16:240, 166:171]
                flair = flair[16:176, 16:240, 166:171]

                t1w = t1w.astype("float32")
                flair = flair.astype("float32")

                t1w = (t1w - t1w.min()) / (t1w.max() - t1w.min())
                flair = (flair - flair.min()) / (flair.max() - flair.min())

                t1w = (t1w * 255).astype("uint8")
                flair = (flair * 255).astype("uint8")

                for i in range(t1w.shape[2]):
                    t1w_slice = Image.fromarray(t1w[:, :, i])
                    flair_slice = Image.fromarray(flair[:, :, i])

                    new_image_path = new_sub_dir / f"{image_path.stem.replace('_T1w.nii', f'_slice-{i}_T1w.png')}"
                    new_flair_path = new_sub_dir / f"{flair_path.stem.replace('_FLAIR.nii', f'_slice-{i}_FLAIR.png')}"

                    t1w_slice.save(new_image_path)
                    flair_slice.save(new_flair_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
