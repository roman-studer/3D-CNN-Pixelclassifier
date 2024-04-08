import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import torchmetrics


class HyperSN(LightningModule):
    """Implementation of  HyperSN for Hyperspectral Cubes (3D Conv) from https://ieeexplore.ieee.org/document/8736016
    based on https://github.com/Pancakerr/HybridSN/blob/master/HybridSN.ipynb"""

    def __init__(self, in_channels, patch_size, class_nums, learning_rate=0.001):
        """
        Initialize the HyperSN model.

        Args:
            in_channels (int): The number of input channels.
            patch_size (int): The size of the patch to be processed.
            class_nums (int): The number of classes for classification.
            learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        """
        super().__init__()
        self.lr_scheduler = None
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.class_nums = class_nums
        self.learning_rate = learning_rate

        task = "binary" if class_nums == 2 else "multiclass"
        if task == "binary":
            self.train_acc = torchmetrics.classification.BinaryAccuracy()
            self.val_acc = torchmetrics.classification.BinaryAccuracy()
            self.precision = torchmetrics.classification.BinaryPrecision()
            self.recall = torchmetrics.classification.BinaryRecall()
            self.f1 = torchmetrics.classification.BinaryF1Score()

        else:
            self.train_acc = torchmetrics.classification.MulticlassAccuracy(
                self.class_nums
            )
            self.test_acc = torchmetrics.classification.MulticlassAccuracy(
                self.class_nums
            )
            self.val_acc = torchmetrics.classification.MulticlassAccuracy(
                self.class_nums
            )
            self.precision = torchmetrics.classification.MulticlassPrecision(
                self.class_nums
            )
            self.recall = torchmetrics.classification.MulticlassRecall(self.class_nums)
            self.f1 = torchmetrics.classification.MulticlassF1Score(self.class_nums)

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, out_channels=12, kernel_size=(5, 4, 4), padding=(1, 1, 1)),
            nn.BatchNorm3d(12),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(12, out_channels=16, kernel_size=(4, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, out_channels=32, kernel_size=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.x1_shape = self.get_shape_after_3d_conv()
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                self.x1_shape[1] * self.x1_shape[2],
                out_channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.x2_shape = self.get_shape_after_2d_conv()

        self.dense1 = nn.Sequential(
            nn.Linear(self.x2_shape, 254), nn.ReLU(inplace=True), nn.Dropout(0.4)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(254, 64), nn.ReLU(inplace=True), nn.Dropout(0.4)
        )
        self.dense3 = nn.Linear(64, self.class_nums)

    def training_step(self, batch, batch_idx):
        x, mask = batch
        x = x.float()
        x.unsqueeze_(1)

        y = (mask[:, self.patch_size // 2, self.patch_size // 2] / 255).round().long()
        y = torch.nn.functional.one_hot(y, num_classes=self.class_nums).float()

        out = self.forward_pass(x)

        if self.class_nums == 2:
            loss = nn.BCEWithLogitsLoss()(out, y)
        else:
            loss = nn.CrossEntropyLoss()(out, y)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        # Calculate metrics

        acc = self.train_acc(out, y)
        precision = self.precision(out, y)
        recall = self.recall(out, y)
        f1 = self.f1(out, y)

        # Log metrics
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_precision",
            precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_recall",
            recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, mask = batch
        x = x.float()
        x = x.unsqueeze(1)

        y = (mask[:, self.patch_size // 2, self.patch_size // 2] / 255).round().long()
        y = torch.nn.functional.one_hot(y, num_classes=self.class_nums).float()

        out = self.forward_pass(x)

        if self.class_nums == 2:
            loss = nn.BCEWithLogitsLoss()(out, y)
        else:
            loss = nn.CrossEntropyLoss()(out, y)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Calculate metrics
        acc = self.val_acc(out, y)
        precision = self.precision(out, y)
        recall = self.recall(out, y)
        f1 = self.f1(out, y)

        # Log metrics
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_precision",
            precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_recall",
            recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("val_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, mask = batch
        x = x.float()
        x = x.unsqueeze(1)

        y = (mask[:, self.patch_size // 2, self.patch_size // 2] / 255).round().long()
        y = torch.nn.functional.one_hot(y, num_classes=self.class_nums).float

        out = self.forward_pass(x)

        if self.class_nums == 2:
            loss = nn.BCEWithLogitsLoss()(out, y)
        else:
            loss = nn.CrossEntropyLoss()(out, y)

        self.log(
            "test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Calculate metrics
        acc = self.test_acc(out, y)
        precision = self.precision(out, y)
        recall = self.recall(out, y)
        f1 = self.f1(out, y)

        # Log metrics
        self.log(
            "test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_precision",
            precision,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_recall",
            recall,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log("test_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"])

        if self.trainer.sanity_checking:
            self.log("train_loss", np.inf)
            self.log("val_loss", np.inf)

        if "val_loss" not in self.trainer.callback_metrics:
            self.log("val_loss", np.inf)

        val_loss = self.trainer.callback_metrics["val_loss"]
        if val_loss is not None:
            self.lr_scheduler.step(val_loss)

        # every n epochs run

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=25, verbose=True, factor=0.5, min_lr=1e-6)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": self.lr_scheduler,
            "monitor": "val_loss",
        }

    def forward_pass(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.conv4(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out

    def get_shape_after_2d_conv(self):
        x = torch.zeros(
            (1, self.x1_shape[1] * self.x1_shape[2], self.x1_shape[3], self.x1_shape[4])
        )
        with torch.no_grad():
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]

    def get_shape_after_3d_conv(self):
        x = torch.zeros((1, 1, self.in_channels, self.patch_size, self.patch_size))
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape
