#!/usr/bin/env python3
"""
Download dataset from Kaggle,
and saving in ./data/raw/
"""

from __future__ import annotations

from pathlib import Path

import kagglehub


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data" / "raw"
DATASET = "irkaal/foodcom-recipes-and-reviews"


def main() -> None:
    """Download the Food.com recipes and reviews dataset to the raw data folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kagglehub.dataset_download(DATASET, output_dir=OUTPUT_DIR)
    print(f"Downloaded '{DATASET}' to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()