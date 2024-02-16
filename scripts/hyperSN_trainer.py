import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from dataloader import HyperspectralDataModule
from hyperSN_model import HyperSN
import yaml
import torch
import os

print(os.getcwd())

torch.set_float32_matmul_precision("medium")

# load config from configurations folder (yaml)
config = yaml.safe_load(
    open("./3D-CNN-Pixelclassifier/configurations/hyperSN_config.yaml")
)

config_hyperSN = config["hyperSN"]
config_dataloader = config["hyperSN_dataloader"]
paths = config["paths"]

# Initialize the model and data module
model = HyperSN(
    in_channels=config_hyperSN["in_channels"],
    patch_size=config_hyperSN["patch_size"],
    class_nums=config_hyperSN["class_nums"],
)

data_module = HyperspectralDataModule(
    batch_size=config_dataloader["batch_size"],
    path_train=paths["train"]["path_input"],
    path_test=paths["test"]["path_input"],
    stride_train=config_dataloader["stride_train"],
    stride_test=config_dataloader["stride_test"],
    window_size=config_dataloader["window_size"],
)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="hyperSN", entity="biocycle")

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=10,
        logger=wandb_logger,
        fast_dev_run=config["fast_dev_run"],
        enable_checkpointing=True,
        default_root_dir=paths["model"],
    )

    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Close the logger
    wandb_logger.finalize("success")
