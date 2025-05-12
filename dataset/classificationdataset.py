import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageClassificationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        PyTorch Dataset for image classification with one-hot encoded labels.

        Args:
            data_dir (str): Path to the data directory (e.g., 'output_dir/train').
            transform (callable, optional): Additional transforms to apply.
        """
        self.data_dir = Path(data_dir)
        self.transform = transform

        # Load normalization stats from parent directory
        stats_path = self.data_dir.parent / "normalization_stats.json"
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.mean = stats["mean"]
        self.std = stats["std"]

        # Define base transforms (convert to tensor and normalize)
        self.base_transforms = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts to [C, H, W] and [0, 1]
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Read images and labels
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        # Iterate through class subfolders
        for idx, class_dir in enumerate(sorted(self.data_dir.iterdir())):
            if class_dir.is_dir():
                class_name = class_dir.name
                self.class_to_idx[class_name] = idx
                for img_path in class_dir.glob("*.[jJ][pP][gG]") or class_dir.glob(
                    "*.[pP][nN][gG]"
                ):
                    self.image_paths.append(img_path)
                    self.labels.append(idx)

        self.num_classes = len(self.class_to_idx)

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Return a single image and its one-hot encoded label."""
        img_path = self.image_paths[idx]
        label_idx = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply base transforms (to tensor and normalize)
        img = self.base_transforms(img)

        # Apply additional transforms if provided
        if self.transform is not None:
            img = self.transform(img)

        # One-hot encode the label
        one_hot_label = torch.zeros(self.num_classes)
        one_hot_label[label_idx] = 1.0

        return img, one_hot_label

    def get_class_names(self):
        """Return a list of class names."""
        return sorted(self.class_to_idx.keys())
