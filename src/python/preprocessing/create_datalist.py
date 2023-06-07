""" Script to create train, validation and test data lists with paths to t1w and flair images. """
from pathlib import Path

import pandas as pd


def create_datalist(sub_dirs):
    data_list = []
    for sub_dir in sub_dirs:
        images_paths = sorted(list(sub_dir.glob("**/*T1w.png")))
        for image_path in images_paths:
            flair_path = image_path.parent / (image_path.name.replace("T1w", "FLAIR"))
            data_list.append({"t1w": str(image_path), "flair": flair_path})

    return pd.DataFrame(data_list)


def main():
    output_dir = Path("/project/outputs/ids/")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("/data/")
    sub_dirs = sorted([x for x in data_dir.iterdir() if x.is_dir()])

    train_sub_dirs = sub_dirs[:35000]
    val_sub_dirs = sub_dirs[35000:35500]
    test_sub_dirs = sub_dirs[35500:]

    data_df = create_datalist(train_sub_dirs)
    data_df.to_csv(output_dir / "train.tsv", index=False, sep="\t")

    data_df = create_datalist(val_sub_dirs)
    data_df.to_csv(output_dir / "validation.tsv", index=False, sep="\t")

    data_df = create_datalist(test_sub_dirs)
    data_df.to_csv(output_dir / "test.tsv", index=False, sep="\t")


if __name__ == "__main__":
    main()
