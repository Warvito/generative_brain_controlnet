""" Script to create train, validation and test data lists with paths to images and radiological reports. """
import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to directory to save files with paths.")

    args = parser.parse_args()
    return args


def main(args):
    data_dir = Path("/data/")
    images_paths = sorted(list(data_dir.glob("**/*T1w.nii.gz")))

    data_list = []
    for image_path in images_paths:
        if "unusable" not in str(image_path):
            data_list.append({"t1w": str(image_path), "flair": str(image_path).replace("T1w", "FLAIR")})

    data_df = pd.DataFrame(data_list)
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_data_list = data_df[:40000]
    val_data_list = data_df[40000:40500]
    test_data_list = data_df[40500:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data_list.to_csv(output_dir / "train.tsv", index=False, sep="\t")
    val_data_list.to_csv(output_dir / "validation.tsv", index=False, sep="\t")
    test_data_list.to_csv(output_dir / "test.tsv", index=False, sep="\t")


if __name__ == "__main__":
    args = parse_args()
    main(args)
