import json
import logging
import os
import random
import shutil
from pathlib import Path

import kagglehub
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def read_images(data_dir):
    """Read all images from subfolders, where each subfolder is a class."""
    data_dir = Path(data_dir)
    image_paths = []
    labels = []

    logger.debug(f"Reading images from {data_dir}")
    # Iterate through each class folder
    for class_dir in data_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            logger.debug(f"Processing class directory: {class_name}")
            # Find all images in the class folder
            for img_path in class_dir.glob("*.[jJ][pP][gG]") or class_dir.glob(
                "*.[pP][nN][gG]"
            ):
                image_paths.append(str(img_path))
                labels.append(class_name)
                logger.debug(f"Found image: {img_path}")

    return image_paths, labels


def stratified_split(image_paths, labels, test_size=0.1, val_size=0.2):
    """Perform stratified train-val-test split."""
    logger.debug("Performing stratified train-val-test split")
    # First, split into train+val and test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels, random_state=42
    )
    logger.debug(f"Test split: {len(test_paths)} images")

    # From train+val, split into train and val
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths,
        train_val_labels,
        test_size=val_size / (1 - test_size),
        stratify=train_val_labels,
        random_state=42,
    )
    logger.debug(
        f"Train split: {len(train_paths)} images, Validation split: {len(val_paths)} images"
    )

    return (
        (train_paths, train_labels),
        (val_paths, val_labels),
        (test_paths, test_labels),
    )


def calculate_normalization(image_paths):
    """Calculate mean and std for RGB channels of training images."""
    logger.debug("Calculating normalization stats for training images")
    means = np.zeros(3)  # RGB
    stds = np.zeros(3)
    n_pixels = 0

    for img_path in image_paths:
        logger.debug(f"Processing image for normalization: {img_path}")
        img = np.array(Image.open(img_path).convert("RGB"))
        img = img / 255.0  # Normalize to [0, 1]
        means += img.mean(axis=(0, 1))  # Mean per channel
        stds += img.std(axis=(0, 1))  # Std per channel
        n_pixels += 1

    means /= n_pixels
    stds /= n_pixels

    return means, stds


def save_normalization_stats(means, stds, output_dir):
    """Save mean and std to a JSON file in the output directory."""
    logger.debug(f"Saving normalization stats to {output_dir}")
    stats = {
        "mean": means.tolist(),  # Convert numpy array to list for JSON
        "std": stds.tolist(),
    }
    output_path = Path(output_dir) / "normalization_stats.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
    logger.debug(f"Normalization stats saved to {output_path}")


def save_images(image_paths, labels, output_dir, split_name):
    """Save images to output_dir/split_name/class_name/."""
    output_dir = Path(output_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Saving {split_name} images to {output_dir}")

    for img_path, label in zip(image_paths, labels):
        class_dir = output_dir / label
        class_dir.mkdir(exist_ok=True)
        dest_path = class_dir / Path(img_path).name
        shutil.copy(img_path, dest_path)
        logger.debug(f"Copied {img_path} to {dest_path}")


def main(data_dir, output_dir, cfg):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created at {output_dir}")

    # Read images and labels
    image_paths, labels = read_images(data_dir)
    logger.info(f"Found {len(image_paths)} images across {len(set(labels))} classes")

    # Perform stratified split
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = (
        stratified_split(
            image_paths,
            labels,
            test_size=cfg.dataset.test_size,
            val_size=cfg.dataset.validation_size,
        )
    )
    logger.info(
        f"Train: {len(train_paths)} images, Val: {len(val_paths)} images, Test: {len(test_paths)} images"
    )

    # Calculate normalization stats on training data
    means, stds = calculate_normalization(train_paths)
    logger.info("Normalization stats (RGB):")
    logger.info(f"Mean: {means}")
    logger.info(f"Std: {stds}")

    # Save normalization stats to JSON
    save_normalization_stats(means, stds, output_dir)
    logger.info(
        f"Normalization stats saved to {output_dir / 'normalization_stats.json'}"
    )

    # Save images to output directory
    save_images(train_paths, train_labels, output_dir, "train")
    save_images(val_paths, val_labels, output_dir, "val")
    save_images(test_paths, test_labels, output_dir, "test")
    logger.info(f"Images saved to {output_dir}")


if __name__ == "__main__":
    # Load configuration
    cfg = OmegaConf.load("params.yaml")

    # Set random seed for reproducibility
    random.seed(cfg.dataset.random_seed)
    np.random.seed(cfg.dataset.random_seed)

    # Download latest version
    path = kagglehub.dataset_download(cfg.dataset.kaggle_dataset_name)

    # Prepare input and output data path and start data preparation
    data_dir = os.path.join(path, cfg.dataset.input_data_subpath)
    output_dir = os.path.join(os.getcwd(), cfg.dataset.output_data_path)
    main(data_dir, output_dir, cfg)
