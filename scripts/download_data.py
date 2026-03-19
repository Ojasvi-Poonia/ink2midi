#!/usr/bin/env python3
"""Download all datasets for OMR training."""

import argparse

from omr.data.download import download_all
from omr.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Download OMR datasets")
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory to store raw downloaded data",
    )
    args = parser.parse_args()

    setup_logging("INFO")
    download_all(args.raw_dir)


if __name__ == "__main__":
    main()
