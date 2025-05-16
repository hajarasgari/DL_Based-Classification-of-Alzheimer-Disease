import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision.models as models


class ResNetModel34(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        num_classes=4,
        learning_rate=1e-3,
        freeze_backbone=True,
    ):
        """
        ResNet with transfer learning for image classification.

        Args:
            num_classes (int): Number of target classes.
            learning_rate (float): Learning rate for the optimizer.
            freeze_backbone (bool): Whether to freeze the ResNet backbone.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load pretrained ResNet18 model
        self.model = models.resnet34(pretrained=True)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        # Replace final classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        # Metrics
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, one_hot_labels = batch
        labels = torch.argmax(one_hot_labels, dim=1)

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

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
        images, one_hot_labels = batch
        labels = torch.argmax(one_hot_labels, dim=1)

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        self.val_accuracy(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, one_hot_labels = batch
        labels = torch.argmax(one_hot_labels, dim=1)

        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        self.test_accuracy(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
