from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
from glob import glob
from pytorch_lightning import LightningDataModule
import pickle


class HyperspectralDataset(Dataset):
    def __init__(
        self,
        path_data,
        window_size,
        stride,
        in_channels,
        mode,
        sample_strategy,
        gradient_masking=False,
        n_per_class=None,
        n_per_cube=None,
        pca_model_path=None,
    ):
        self.cube_dir = path_data
        self.mask_dir = path_data
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.sample_strategy = sample_strategy
        self.n_windows_per_class = n_per_class
        self.n_windows_per_cube = n_per_cube
        self.p = window_size // 2
        self.gradient_masking = gradient_masking
        # list subfolders starting with E
        self.get_exp_files(path_data)

        self.current_cube = None
        self.current_mask = None
        self.current_cube_index = -1
        self.image_shape = None
        self.n_pc = in_channels
        self.window_indices = self.prepare_window_indices()
        self.gradient_mask = self.get_gradient_mask()

        self.patches_loaded = 0
        self.total_patches = len(self.window_indices) // len(self.cube_files)

        self.pca_model_path = pca_model_path
        self.pca = self.get_pca()

    def get_pca(self):
        path = os.path.join(self.pca_model_path, f"{self.n_pc}_pca.pkl")
        if os.path.exists(path):
            print("Loading PCA model from file:", path)
            with open(path, "rb") as f:
                pca = pickle.load(f)

        if not os.path.exists(path) and self.mode == "test":
            raise ValueError(
                f"PCA model not found at {path}. Current mode: {self.mode}"
            )
        return pca

    def get_exp_files(self, path_data):
        self.cube_files = [
            i.split("\\")[-1] for i in glob(os.path.join(path_data, "E*"))
        ]
        # catch changing behavior of glob
        if "\\" in self.cube_files[0]:
            self.cube_files = [
                i.split("\\")[-1] for i in glob(os.path.join(path_data, "E*"))
            ]

        elif "/" in self.cube_files[0]:
            self.cube_files = [
                i.split("/")[-1] for i in glob(os.path.join(path_data, "E*"))
            ]

    def prepare_window_indices(self):
        window_indices = []
        for cube_index, cube_file in enumerate(self.cube_files):
            cube_path = os.path.join(self.cube_dir, cube_file, "hsi.npy")
            cube = np.load(cube_path)

            # cube = np.transpose(cube, (1, 0, 2))
            self.image_shape = cube.shape[:2]

            if self.sample_strategy == "grid":
                for i in range(cube.shape[0] // self.stride):
                    for j in range(cube.shape[1] // self.stride):
                        window_indices.append(
                            (cube_index, i * self.stride, j * self.stride)
                        )

            elif self.sample_strategy == "random":
                for _ in range(self.n_windows_per_cube):
                    i = np.random.randint(0, cube.shape[0])
                    j = np.random.randint(0, cube.shape[1])
                    window_indices.append((cube_index, i, j))

            elif self.sample_strategy == "uniform":
                if os.path.exists(os.path.join(self.mask_dir, cube_file, "mask.bmp")):
                    mask = cv2.imread(
                        os.path.join(self.mask_dir, cube_file, "mask.bmp"),
                        cv2.IMREAD_GRAYSCALE,
                    )

                elif os.path.exists(
                    os.path.join(self.mask_dir, cube_file, "hsi_masks")
                ):
                    mask = self.combine_obj_masks(
                        os.path.join(self.mask_dir, cube_file)
                    )

                else:
                    raise ValueError(
                        f"Mask for cube {cube_file} not found at {cube_path}"
                    )

                # get classes, unique values in mask
                classes = np.unique(mask)

                # sample n random indices per class
                for c in classes:
                    indices = np.argwhere(mask == c)
                    indices = indices[
                        np.random.choice(
                            indices.shape[0], self.n_windows_per_class, replace=True
                        )
                    ]
                    for i, j in indices:
                        window_indices.append((cube_index, i, j))

                # shuffle
                np.random.shuffle(window_indices)

            else:
                raise ValueError(
                    f"Sampling strategy {self.sample_strategy} not recognized or implemented"
                )

        return window_indices

    def load_cube(self, cube_index):
        if cube_index != self.current_cube_index:
            cube_path = self.cube_files[cube_index]

            if os.path.exists(os.path.join(self.cube_dir, cube_path, "mask.bmp")):
                mask_all = cv2.imread(
                    os.path.join(self.cube_dir, cube_path, "mask.bmp"),
                    cv2.IMREAD_GRAYSCALE,
                )

            elif os.path.exists(os.path.join(self.cube_dir, cube_path, "hsi_masks")):
                mask_all = self.combine_obj_masks(
                    os.path.join(self.mask_dir, self.cube_files[cube_index])
                )

            else:
                raise ValueError(f"Mode {self.mode} not recognized or implemented")

            if mask_all is None:
                raise ValueError(f"No mask found for cube {cube_index}")

            print(f"Loading cube {cube_index} from {cube_path}")

            self.current_cube = np.load(
                os.path.join(self.cube_dir, cube_path, "hsi.npy")
            )

            self.pre_process_cube()
            self.current_cube = np.pad(
                self.current_cube,
                ((self.p, self.p), (self.p, self.p), (0, 0)),
                mode="constant",
                constant_values=0,
            )

            self.current_cube = self.pca.transform(
                self.current_cube.reshape(-1, self.current_cube.shape[-1])
            ).reshape(
                self.current_cube.shape[0],
                self.current_cube.shape[1],
                self.n_pc,
            )

            self.current_mask = mask_all
            self.current_mask = np.pad(
                self.current_mask,
                ((self.p, self.p), (self.p, self.p)),
                mode="constant",
                constant_values=0,
            )
            self.current_mask = self.current_mask.astype(int)
            self.current_cube_index = cube_index

        else:
            pass

    def combine_obj_masks(self, mask_dir):
        mask_files = glob(os.path.join(mask_dir, "hsi_masks", "*.bmp"))
        if len(mask_files) == 0:
            raise ValueError(f"No mask files found in {mask_dir}")

        mask = None
        for mask_file in mask_files:
            mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = mask_img
            else:
                mask = np.logical_or(mask, mask_img)

        return mask

    def pre_process_cube(self, cube=None):
        # TODO: implement pca, random occlusion, gradient masking (if necessary)
        cube = self.crop_bands(cube)
        # self.remove_background()
        cube = self.snv_transform(cube)
        return cube

    def remove_background(self):
        """Sets spectra with mean intensity below 600 to zero on all bands. Treats overall low intensity spectra as
        background.

        Note:
            - Changes in light intensity between cubes are not considered.
            - Function assumes that edge bands are removed, i.e. spectra are cropped.
            - Function must be applied before normalization.
        """
        mean_intensity = np.mean(self.current_cube, axis=2)
        self.current_cube[mean_intensity < 600] = 0

    def crop_bands(self, cube=None):
        """Removes bands 0-8 and 210-224. Assumes cube is of shape (w, h, 224).

        Note:
            - Function assumes that edge bands are removed, i.e. spectra are cropped.
        """
        if cube is None:
            self.current_cube = self.current_cube[:, :, 8:210]
        else:
            return cube[:, :, 8:210]

    def apply_gradient_mask(self, window):
        """Applies gradient mask to window as described in https://www.mdpi.com/2072-4292/15/12/3123"""
        return window * self.gradient_mask

    def get_gradient_mask(self):
        s = self.window_size
        p = self.n_pc
        center = (s + 1) / 2
        mask = np.zeros((s, s))
        for i in range(s):
            for j in range(s):
                mask[i, j] = 1 - ((i - center + 1) ** 2 + (j - center + 1) ** 2) / (
                    2 * center**2
                )

        mask = np.expand_dims(mask, axis=0)
        mask = np.repeat(mask, p, axis=0)

        return mask

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        cube_index, i, j = self.window_indices[idx]

        if self.current_cube is None:
            self.load_cube(cube_index)

        if self.patches_loaded >= self.total_patches:
            self.load_cube(cube_index)
            self.patches_loaded = 0

        window = self.current_cube[
            i : i + self.window_size, j : j + self.window_size, :
        ].astype(np.float32)
        window = np.transpose(window, (2, 0, 1))
        window_mask = self.current_mask[
            i : i + self.window_size, j : j + self.window_size
        ]

        if window.shape != (self.n_pc, self.window_size, self.window_size):
            # resize window to correct shape
            window = np.array(
                [
                    cv2.resize(
                        s,
                        (self.window_size, self.window_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    for s in window
                ]
            )

            window_mask = cv2.resize(
                window_mask,
                (self.window_size, self.window_size),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.gradient_masking:
            window = self.apply_gradient_mask(window)

        self.patches_loaded += 1

        return window, window_mask

    def iterate_full_cube(self):
        """Iterates over the full cube and returns a generator of windows and masks."""

        for idx in range(len(self.window_indices)):
            cube_index, i, j = self.window_indices[idx]

            if self.current_cube is None:
                self.load_cube(cube_index)

            if self.patches_loaded >= self.total_patches:
                self.load_cube(cube_index)
                self.patches_loaded = 0

            window = self.current_cube[
                i : i + self.window_size, j : j + self.window_size, :
            ].astype(np.float32)
            window = np.transpose(window, (2, 0, 1))
            window_mask = self.current_mask[
                i : i + self.window_size, j : j + self.window_size
            ]

            if window.shape != (self.n_pc, self.window_size, self.window_size):
                # resize window to correct shape
                window = np.array(
                    [
                        cv2.resize(
                            s,
                            (self.window_size, self.window_size),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for s in window
                    ]
                )

                window_mask = cv2.resize(
                    window_mask,
                    (self.window_size, self.window_size),
                    interpolation=cv2.INTER_NEAREST,
                )

            window = self.apply_gradient_mask(window)

            yield window, window_mask, cube_index, i, j

    def snv_transform(self, cube=None):
        """Perform Standard Normal Variate (SNV) transformation on spectra.
        This transformation is performed on each spectrum individually. The mean of each spectrum is subtracted from the
        spectrum and the result is divided by the standard deviation of the spectrum
        """
        if cube is None:
            self.current_cube = (
                self.current_cube - np.mean(self.current_cube, axis=2, keepdims=True)
            ) / np.std(self.current_cube, axis=2, keepdims=True)
        else:
            return (cube - np.mean(cube, axis=2, keepdims=True)) / np.std(
                cube, axis=2, keepdims=True
            )


class HyperspectralDataModule(LightningDataModule):
    def __init__(
        self,
        path_train,
        path_test,
        window_size,
        stride_train,
        stride_test,
        gradient_masking,
        in_channels,
        batch_size,
        n_per_class,
        n_per_cube,
        sample_strategy,
        pca_model_path,
    ):
        super().__init__()
        self.path_train = path_train
        self.path_test = path_test
        self.window_size = window_size
        self.stride_train = stride_train
        self.stride_test = stride_test
        self.gradient_masking = gradient_masking
        self.in_channels = in_channels
        self.batch_size = batch_size
        self.n_per_class = n_per_class
        self.n_per_cube = n_per_cube
        self.sample_strategy = sample_strategy
        self.pca_model_path = pca_model_path

        print("dataloader: ", os.getcwd())

    def test_dataloader(self):
        test_dataset = HyperspectralDataset(
            self.path_test,
            self.window_size,
            self.stride_test,
            self.in_channels,
            "test",
            self.sample_strategy,
            self.gradient_masking,
            self.n_per_class,
            self.n_per_cube,
            self.pca_model_path,
        )
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def train_dataloader(self):
        train_dataset = HyperspectralDataset(
            self.path_train,
            self.window_size,
            self.stride_train,
            self.in_channels,
            "train",
            self.sample_strategy,
            self.gradient_masking,
            self.n_per_class,
            self.n_per_cube,
            self.pca_model_path,
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return self.test_dataloader()
