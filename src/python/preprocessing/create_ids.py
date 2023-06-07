""" Script to create train, validation and test data lists with paths to images and radiological reports. """
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", help="Path to directory to save files with paths.")

    args = parser.parse_args()
    return args


def main(args):
    output_dir = Path(args.output_dir)
    # TODO: implement script


if __name__ == "__main__":
    args = parse_args()
    main(args)
