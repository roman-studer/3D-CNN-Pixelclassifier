import torch
import torch.nn as nn
from pytorch_lightning import LightningModule


class HyperSN(LightningModule):
    """Implementation of  HyperSN for Hyperspectral Cubes (3D Conv) from https://ieeexplore.ieee.org/document/8736016
    based on https://github.com/Pancakerr/HybridSN/blob/master/HybridSN.ipynb"""

    def __init__(self, in_channels, patch_size, class_nums):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.class_nums = class_nums

        self.conv1 = nn.Sequential(
            nn.Conv3d(1, out_channels=8, kernel_size=(7, 23, 23), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(8, out_channels=16, kernel_size=(5, 21, 21), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(16, out_channels=32, kernel_size=(3, 19, 19), padding=(1, 1, 1)),
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
            nn.Linear(self.x2_shape, 1024), nn.ReLU(inplace=True), nn.Dropout(0.4)
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 128), nn.ReLU(inplace=True), nn.Dropout(0.4)
        )
        self.dense3 = nn.Linear(128, self.class_nums)

    def training_step(self, batch, batch_idx):
        x, mask = batch
        x = x.float()

        y = (mask[:, self.patch_size // 2, self.patch_size // 2] / 255).round().long()
        y = torch.nn.functional.one_hot(y, num_classes=self.class_nums).float()

        x.unsqueeze_(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        x = self.conv4(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)

        loss = nn.CrossEntropyLoss()(out, y)
        return loss

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

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
