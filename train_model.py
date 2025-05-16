import logging
import os

import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from dataset.classificationdataset import ImageClassificationDataset

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def train(cfg: DictConfig):
    # Log configuration loading
    logger.info("Starting training with provided configuration")
    logger.debug(f"Configuration: {cfg}")

    # Paths
    train_dir = os.path.join(cfg.dataset.output_data_path, "train")
    val_dir = os.path.join(cfg.dataset.output_data_path, "val")
    logger.info(f"Training directory: {train_dir}")
    logger.info(f"Validation directory: {val_dir}")

    # Define transforms (resize to ensure consistent input size)
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
        ]
    )
    logger.debug("Defined training and validation transforms")

    # Create datasets
    logger.info("Loading training dataset")
    train_dataset = ImageClassificationDataset(train_dir, transform=train_transform)
    logger.info("Loading validation dataset")
    val_dataset = ImageClassificationDataset(val_dir, transform=val_transform)
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.debug(f"Number of classes: {train_dataset.num_classes}")
    logger.debug(f"Class names: {train_dataset.get_class_names()}")

    # Get a sample from the train_set to inspect its shape
    sample_image, sample_label = train_dataset[
        0
    ]  # Assuming dataset returns (image, label)
    logger.debug(f"Sample image shape: {sample_image.shape}")
    logger.debug(f"Sample label shape: {sample_label.shape}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=4,
    )
    logger.info("Created data loaders")
    logger.debug(f"Train loader: batch_size={cfg.trainer.batch_size}")
    logger.debug(f"Val loader: batch_size={cfg.trainer.batch_size}")

    # Instantiate model from config
    logger.info("Instantiating model")
    model = instantiate(
        cfg.model,
        in_channels=sample_image.shape[0],
        num_classes=train_dataset.num_classes,
    )
    logger.debug(f"Model instantiated: {model}")

    # ModelCheckpoint callback to save best model by val loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-checkpoint",
        dirpath=os.getcwd(),
        save_weights_only=True,
    )
    logger.debug("Configured ModelCheckpoint callback")

    # EarlyStopping callback to stop training after no improvement
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=cfg.trainer.patience,
        verbose=True,
    )
    logger.debug(
        f"Configured EarlyStopping callback with patience={cfg.trainer.patience}"
    )

    # Trainer with logger + callbacks
    logger.info("Initializing PyTorch Lightning trainer")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=False,
    )
    logger.debug(
        f"Trainer configured: max_epochs={cfg.trainer.max_epochs}, accelerator={'gpu' if torch.cuda.is_available() else 'cpu'}"
    )

    # Train
    logger.info("Starting training")
    trainer.fit(model, train_loader, val_loader)
    logger.info("Training completed")


if __name__ == "__main__":
    # Load configuration
    cfg = OmegaConf.load("params.yaml")

    train(cfg)
