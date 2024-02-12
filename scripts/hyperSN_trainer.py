import pytorch_lightning as pl
import lightning as lightning
from pytorch_lightning.loggers import WandbLogger
from dataloader import HyperspectralDataModule
from hyperSN_model import HyperSN
import yaml


# load config from configurations folder (yaml)
config = yaml.safe_load(open("../configurations/hyperSN_config.yaml"))

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
    stride=config_dataloader["stride"],
    window_size=config_dataloader["window_size"],
)

if __name__ == "__main__":
    wandb_logger = WandbLogger(name="3D-CNN-Pixelclassifier", project="hyperSN")

    # Initialize the trainer
    trainer = pl.Trainer(max_epochs=10, logger=wandb_logger, accelerator="cpu")

    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Close the logger
    wandb_logger.finalize("success")

# %%
