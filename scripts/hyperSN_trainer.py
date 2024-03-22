import logging

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from dataloader import HyperspectralDataModule
from hyperSN_model import HyperSN
import yaml
import torch
import os
import pickle
from sklearn.decomposition import IncrementalPCA
import numpy as np
from glob import glob
import shutil

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


def get_exp_files(path_data):
    cube_files = [i.split("\\")[-1] for i in glob(os.path.join(path_data, "E*"))]
    # catch changing behavior of glob
    if "\\" in cube_files[0]:
        cube_files = [i.split("\\")[-1] for i in glob(os.path.join(path_data, "E*"))]

    elif "/" in cube_files[0]:
        cube_files = [i.split("/")[-1] for i in glob(os.path.join(path_data, "E*"))]

    return cube_files


def snv_transform(cube=None):
    """Perform Standard Normal Variate (SNV) transformation on spectra.
    This transformation is performed on each spectrum individually. The mean of each spectrum is subtracted from the
    spectrum and the result is divided by the standard deviation of the spectrum
    """
    return (cube - np.mean(cube, axis=2, keepdims=True)) / np.std(
        cube, axis=2, keepdims=True
    )


def crop_bands(cube=None):
    """Removes bands 0-8 and 210-224. Assumes cube is of shape (w, h, 224).

    Note:
        - Function assumes that edge bands are removed, i.e. spectra are cropped.
    """
    return cube[:, :, 8:210]


def pre_process_cube(cube=None):
    # TODO: implement pca, random occlusion, gradient masking (if necessary)
    cube = crop_bands(cube)
    # self.remove_background()
    cube = snv_transform(cube)
    return cube


def train_incremental_pca(n_pc, cube_files, cube_dir, pca_model_path):
    path = os.path.join(pca_model_path, f"{n_pc}_pca.pkl")
    if not os.path.exists(path):
        print("Training PCA model")
        pca = IncrementalPCA(n_components=n_pc)
        for cube_index, cube_file in enumerate(cube_files):
            cube_path = os.path.join(cube_dir, cube_file, "hsi.npy")
            cube = np.load(cube_path)

            cube = pre_process_cube(cube)

            x = cube.reshape(-1, cube.shape[-1])
            x = x.astype(float)
            pca.partial_fit(x)

        os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(pca, f)

        print("PCA model saved")


def preprocess_cubes(pca_model_dir, cube_dir, out_dir):
    """
    Apply PCA and other preprocessing steps to the cubes in train, test and val folder
    :param pca_model_dir: str, path to the PCA model
    :param cube_dir: str, path to the base folder containing the cubes
    :param out_dir: str, path to the base output folder
    :return: None
    """
    # check if out_dir is not empty
    if (
        os.path.exists(
            os.path.join(out_dir, f"train_pca_{config_hyperSN['in_channels']}")
        )
        and len(
            os.listdir(
                os.path.join(out_dir, f"train_pca_{config_hyperSN['in_channels']}")
            )
        )
        > 0
    ):
        logging.info(
            f"Output directory {out_dir} is not empty. Skipping preprocessing."
        )
        return None

    with open(pca_model_dir, "rb") as f:
        pca = pickle.load(f)

    for dataset in ["train", "test"]:
        logging.info(f"Preprocessing {dataset} dataset")
        dataset_path = os.path.join(cube_dir.replace("train", dataset))
        out_dataset_path = os.path.join(
            out_dir, f"{dataset}_pca_{config_hyperSN['in_channels']}"
        )
        os.makedirs(out_dataset_path, exist_ok=True)

        cube_files = [i.split("\\")[-1] for i in glob(os.path.join(dataset_path, "E*"))]
        for cube_index, cube_file in enumerate(cube_files):
            try:
                cube_path = os.path.join(dataset_path, cube_file, "hsi.npy")
            except FileNotFoundError:
                cube_path = os.path.join(dataset_path, cube_index, "hsi.npy")

            cube = np.load(cube_path)

            cube = pre_process_cube(cube)

            x = cube.reshape(-1, cube.shape[-1])
            x = x.astype(float)
            x_pca = pca.transform(x)
            cube_pca = x_pca.reshape(cube.shape[0], cube.shape[1], -1)

            out_cube_path = os.path.join(out_dataset_path, cube_file)
            os.makedirs(out_cube_path, exist_ok=True)
            np.save(os.path.join(out_cube_path, "hsi.npy"), cube_pca)

    return None


train_incremental_pca(
    config_hyperSN["in_channels"],
    get_exp_files(paths["train"]["path_input"]),
    paths["train"]["path_input"],
    paths["pca_model"],
)

preprocess_cubes(
    os.path.join(paths["pca_model"], f"{config_hyperSN['in_channels']}_pca.pkl"),
    paths["train"]["path_input"],
    paths["train"]["path_output"],
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
    pca_model_path=paths["pca_model"],
    num_workers=config_dataloader["num_workers"],
)


if __name__ == "__main__":
    if config["online_logger"] is False:
        wandb_logger = None
    elif config["online_logger"] is True:
        wandb_logger = WandbLogger(project="hyperSN", entity="biocycle")
        wandb_logger.log_hyperparams(config)

        # save scripts in the same folder as the model (windows and linux)
        os.makedirs(
            os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            exist_ok=True,
        )

        try:
            shutil.copy(
                "./hyperSN_model.py",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )
            shutil.copy(
                "./dataloader.py",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )
            shutil.copy(
                "./hyperSN_trainer.py",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )
            shutil.copy(
                "../configurations/hyperSN_config.yaml",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )
        except FileNotFoundError:
            shutil.copy(
                "./3D-CNN-Pixelclassifier/scripts/hyperSN_model.py",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )
            shutil.copy(
                "./3D-CNN-Pixelclassifier/scripts/dataloader.py",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )
            shutil.copy(
                "./3D-CNN-Pixelclassifier/scripts/hyperSN_trainer.py",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )
            shutil.copy(
                "./3D-CNN-Pixelclassifier/configurations/hyperSN_config.yaml",
                os.path.join(paths["model"], wandb_logger.experiment.name, "scripts"),
            )

    # Initialize the trainer code so that checkpoints scripts configuration etc.
    # is saved in the same individual folder for each project
    default_root_dir = (
        os.path.join(paths["model"], wandb_logger.experiment.name)
        if config["online_logger"]
        else None
    )

    trainer = pl.Trainer(
        max_epochs=config_hyperSN["max_epochs"],
        logger=wandb_logger,
        fast_dev_run=config["fast_dev_run"],
        enable_checkpointing=True,
        default_root_dir=default_root_dir,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        callbacks=[
            EarlyStopping(
                monitor="val_loss_epoch", mode="min", patience=config["patience"]
            ),
            ModelCheckpoint(
                monitor="val_loss_epoch",
                mode="min",
                save_top_k=1,
                dirpath=default_root_dir,
                filename="best_model",
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    # Train the model
    trainer.fit(model=model, datamodule=data_module)

    # Close the logger
    if config["online_logger"] is True:
        wandb_logger.finalize("success")

    print("Training complete")
