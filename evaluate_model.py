import importlib
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import torchmetrics
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from dataset.classificationdataset import ImageClassificationDataset

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def evaluate_model(cfg: DictConfig):
    """Evaluate the model on the validation set and save plots."""
    logger.info("Starting model evaluation")
    logger.debug(f"Configuration: {cfg}")

    # Paths
    val_dir = os.path.join(cfg.dataset.output_data_path, "val")
    checkpoint_path = Path(os.getcwd()) / "best-checkpoint.ckpt"
    plot_dir = Path(cfg.evaluation_results_path)

    logger.info(f"Validation directory: {val_dir}")
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    logger.info(f"Saving plots to: {plot_dir}")

    # Ensure plot directory exists
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created plot directory: {plot_dir}")

    # Define transforms
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
        ]
    )
    logger.debug("Defined validation transforms")

    # Load validation dataset
    logger.info("Loading validation dataset")
    val_dataset = ImageClassificationDataset(val_dir, transform=val_transform)
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.debug(f"Number of classes: {val_dataset.num_classes}")
    logger.debug(f"Class names: {val_dataset.get_class_names()}")

    # Get sample image shape for model instantiation
    sample_image, _ = val_dataset[0]
    logger.debug(f"Sample image shape: {sample_image.shape}")

    # Create data loader
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=4,
    )
    logger.info("Created validation data loader")
    logger.debug(f"Val loader: batch_size={cfg.trainer.batch_size}")

    # Load model class using importlib
    logger.info("Loading model class from config using importlib")
    target = cfg.model._target_
    module_name, class_name = target.rsplit(".", 1)
    logger.debug(f"Loading module: {module_name}, class: {class_name}")

    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    logger.debug(f"Retrieved model class: {model_class}")

    # Load model from checkpoint
    logger.info(
        f"Loading model from checkpoint using {class_name}.load_from_checkpoint"
    )
    model = model_class.load_from_checkpoint(
        checkpoint_path,
        in_channels=sample_image.shape[0],
        num_classes=val_dataset.num_classes,
    )
    model.eval()
    logger.debug(f"Model loaded and set to evaluation mode: {model}")

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.debug(f"Model moved to device: {device}")

    # Initialize metrics
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=val_dataset.num_classes
    ).to(device)
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    # Evaluate on validation set
    logger.info("Evaluating on validation set")
    with torch.no_grad():
        for batch in val_loader:
            images, one_hot_labels = batch
            images = images.to(device)
            labels = torch.argmax(one_hot_labels, dim=1).to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            accuracy(preds, labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

    # Compute final metrics
    val_loss = total_loss / num_batches
    val_acc = accuracy.compute().item()
    logger.info(f"Validation Loss: {val_loss:.4f}")
    logger.info(f"Validation Accuracy: {val_acc:.4f}")

    # Compute confusion matrix
    logger.info("Generating confusion matrix")
    cm = confusion_matrix(all_labels, all_preds)
    class_names = val_dataset.get_class_names()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = plot_dir / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logger.info(f"Confusion matrix saved to: {cm_path}")

    # Compute per-class accuracy
    logger.info("Computing per-class accuracies")
    per_class_acc = []
    for i in range(val_dataset.num_classes):
        class_correct = np.sum((np.array(all_labels) == i) & (np.array(all_preds) == i))
        class_total = np.sum(np.array(all_labels) == i)
        class_acc = class_correct / class_total if class_total > 0 else 0.0
        per_class_acc.append(class_acc)

    # Plot per-class accuracy
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, per_class_acc)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    acc_path = plot_dir / "class_accuracies.png"
    plt.savefig(acc_path)
    plt.close()
    logger.info(f"Per-class accuracy plot saved to: {acc_path}")


if __name__ == "__main__":
    # Load configuration
    cfg = OmegaConf.load("params.yaml")

    evaluate_model(cfg)
