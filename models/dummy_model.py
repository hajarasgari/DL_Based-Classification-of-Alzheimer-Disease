import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class SimpleCNN(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        learning_rate=1e-3,
    ):
        """
        Simple CNN for image classification, independent of input size.

        Args:
            in_channels (int): Number of input channels (e.g., 3 for RGB).
            num_classes (int): Number of classes.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
        )

        # Global average pooling to handle variable input sizes
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        """Forward pass."""
        x = self.conv_layers(x)  # [batch, channels, H, W]
        x = self.global_pool(x)  # [batch, channels, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, channels]
        x = self.fc(x)  # [batch, num_classes]
        return x

    def training_step(self, batch, batch_idx):
        """Training step."""
        images, one_hot_labels = batch
        # Convert one-hot labels to class indices
        labels = torch.argmax(one_hot_labels, dim=1)

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Log loss and accuracy
        self.train_accuracy(logits, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        images, one_hot_labels = batch
        labels = torch.argmax(one_hot_labels, dim=1)

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Log loss and accuracy
        self.val_accuracy(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step."""
        images, one_hot_labels = batch
        labels = torch.argmax(one_hot_labels, dim=1)

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        # Log loss and accuracy
        self.test_accuracy(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
