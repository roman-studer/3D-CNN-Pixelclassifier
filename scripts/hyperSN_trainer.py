import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import LearningRateFinder
from dataloader import HyperspectralDataModule
from hyperSN_model import HyperSN
import yaml
import torch
import os

print(os.getcwd())

torch.set_float32_matmul_precision("medium")

# load config from configurations folder (yaml)
# TODO path to config file changes between local and server
try:
    config = yaml.safe_load(
        open("./3D-CNN-Pixelclassifier/configurations/hyperSN_config.yaml")
    )
except FileNotFoundError:
    config = yaml.safe_load(open("../configurations/hyperSN_config.yaml"))

config_hyperSN = config["hyperSN"]
config_dataloader = config["hyperSN_dataloader"]
paths = config["paths"]

# Initialize the model and data module
model = HyperSN(
    in_channels=config_hyperSN["in_channels"],
    patch_size=config_hyperSN["patch_size"],
    class_nums=config_hyperSN["class_nums"],
    learning_rate=config_hyperSN["learning_rate"],
)

data_module = HyperspectralDataModule(
    batch_size=config_dataloader["batch_size"],
    path_train=paths["train"]["path_input"],
    path_test=paths["test"]["path_input"],
    stride_train=config_dataloader["stride_train"],
    stride_test=config_dataloader["stride_test"],
    gradient_masking=config_dataloader["gradient_masking"],
    in_channels=config_hyperSN["in_channels"],
    window_size=config_dataloader["window_size"],
    n_per_class=config_dataloader["n_per_class"],
    n_per_cube=config_dataloader["n_per_cube"],
    sample_strategy=config_dataloader["patch_sample_strategy"],
)


if __name__ == "__main__":
    wandb_logger = WandbLogger(project="hyperSN", entity="biocycle")
    wandb_logger.log_hyperparams(config)

    # save hyperSN_trainer.py, hyperSN_model.py and dataloader.py script for later use
    wandb_logger.experiment.log_artifact("hyperSN_trainer.py")
    wandb_logger.experiment.log_artifact("dataloader.py")
    wandb_logger.experiment.log_artifact("hyperSN_model.py")

    # Initialize the trainerhange code so that checkpoints scripts configuration etc. is saved in the same individual folder for each project
    trainer = pl.Trainer(
        max_epochs=config_hyperSN["max_epochs"],
        logger=wandb_logger,
        fast_dev_run=config["fast_dev_run"],
        enable_checkpointing=True,
        default_root_dir=os.path.join(paths["model"], wandb_logger.experiment.name),
        limit_train_batches=0.2,
        callbacks=[
            EarlyStopping(
                monitor="val_loss_epoch", mode="min", patience=config["patience"]
            ),
            ModelCheckpoint(
                monitor="val_loss_epoch",
                mode="min",
                save_top_k=1,
                dirpath=os.path.join(paths["model"], wandb_logger.experiment.name),
                filename="best_model",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Close the logger
    wandb_logger.finalize("success")
