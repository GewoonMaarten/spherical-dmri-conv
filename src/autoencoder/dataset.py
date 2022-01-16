import copy
import itertools
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from autoencoder.argparse import file_path
from autoencoder.logger import logger
from autoencoder.spherical.harmonics import gram_schmidt_sh_inv, sh_basis_real


class SphericalTransformer(object):
    def __init__(
        self, l_max: int = 0, symmetric: bool = True, inversion_n_iters: int = 1000
    ) -> None:
        self._l_max = l_max
        self._symmetric = 2 if symmetric else 1
        self._inversion_n_iters = inversion_n_iters

    def __call__(self, **kwargs):
        data = kwargs["data"]
        scheme = kwargs["scheme"]

        b_s = np.unique(scheme[:, 3])  # 5 unique values
        ti_s = np.unique(scheme[:, 4])  # 28 unique values
        te_s = np.unique(scheme[:, 5])  # 3 unique values

        ti_n = ti_s.shape[0]
        te_n = te_s.shape[0]
        b_n = b_s.shape[0]

        prev_b = b_s[0]

        # Fit the spherical harmonics on the gradients.
        y = sh_basis_real(torch.from_numpy(scheme[:, :3]), self._l_max).to(data.device)
        y_inv = gram_schmidt_sh_inv(y, self._l_max, n_iters=self._inversion_n_iters)

        sh_coefficients_b_idx = dict()
        sh_coefficients = dict()
        l_sizes = dict()
        for l in range(0, self._l_max + 1, self._symmetric):
            o = 2 * l + 1
            sh_coefficients_b_idx[l] = 0
            sh_coefficients[l] = torch.zeros(
                (data.shape[0], ti_n, te_n, b_n, o), device=data.device
            )
            l_sizes[l] = o

        for (ti_idx, ti), (te_idx, te), b in itertools.product(
            enumerate(ti_s),
            enumerate(te_s),
            b_s,
        ):
            # If we've visited all b values, we reset the counter
            if prev_b == b:
                sh_coefficients_b_idx = {k: 0 for k in sh_coefficients_b_idx}
                prev_b = b

            filter_scheme = (
                (scheme[:, 3] == b) & (scheme[:, 4] == ti) & (scheme[:, 5] == te)
            )

            if not np.any(filter_scheme):
                continue

            data_filtered = data[:, filter_scheme]

            # Get the maximum l where the amount of coefficients is smaller or equal to the amount of voxels.
            l = [l for l, o in l_sizes.items() if o <= data_filtered.shape[1]][-1]
            l_size = np.sum([2 * s + 1 for s in range(0, l + 1, 2)])
            y_inv_filtered = y_inv[:l_size, filter_scheme]

            sh_coefficient = torch.einsum("np,lp->nl", data_filtered, y_inv_filtered)

            # Extract even covariants.
            s = 0
            for l in range(0, l + 1, self._symmetric):
                o = 2 * l + 1

                sh_coefficients[l][
                    :, ti_idx, te_idx, sh_coefficients_b_idx[l]
                ] = sh_coefficient[:, torch.arange(s, s + o)]

                s += o
                sh_coefficients_b_idx[l] += 1

        return sh_coefficients


class MRIMemoryDataset(Dataset):
    def __init__(
        self,
        data_file_path: Path,
        subject_list: list[int],
        exclude: Optional[list[int]] = None,
        include: Optional[list[int]] = None,
        do_preload_in_gpu: bool = True,
        transform=None,
    ):
        """Create a dataset from the selected subjects in the subject list

        Args:
            data_file_path (Path): Data h5 file path.
            subject_list (np.ndarray): ist of all the subjects to include.
            exclude (list[int], optional): list of features to exclude from training. Cannot be used together with include. Defaults to None.
            include (list[int], optional): list of features to only include for training. Cannot be used together with exclude. Defaults to None.
            do_preload_in_gpu (bool, optional): preload all data in GPU memory, instead of for each batch. Defaults to True.
        """

        assert (
            exclude is None or include is None
        ), "Only specify include or exclude, not both."

        with h5py.File(data_file_path, "r") as archive:
            scheme = archive.get("scheme")[()]
            indexes = archive.get("index")[()]
            # indexes of the data we want to load
            (selection, *_) = np.where(np.isin(indexes, subject_list))
            self.target = archive.get("data")[selection]
            self.sample = copy.deepcopy(self.target)

        if include is not None:
            self.sample = self.sample[:, include]
            scheme = scheme[include]
        elif exclude is not None:
            self.sample = np.delete(self.sample, exclude)
            scheme = np.delete(scheme, exclude)

        self.target = torch.from_numpy(self.target)
        self.sample = torch.from_numpy(self.sample)

        if do_preload_in_gpu:
            self.target = self.target.to("cuda")
            self.sample = self.sample.to("cuda")

        if transform:
            self.sample = transform(data=self.sample, scheme=scheme)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.target)

    def __getitem__(self, index):
        """Generates one sample of data"""
        if isinstance(self.sample, dict):
            return {
                "target": self.target[index],
                "sample": {k: v[index] for (k, v) in self.sample.items()},
            }
        else:
            return {"target": self.target[index], "sample": self.sample[index]}

    def __getstate__(self):
        """Return state values to be pickled."""
        return None

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        pass


class MRIDataset(Dataset):
    def __init__(
        self,
        data_file_path: Path,
        subject_list: list[int],
        exclude: Optional[list[int]] = None,
        include: Optional[list[int]] = None,
        transform=None,
    ):
        """Create a dataset from the selected subjects in the subject list

        Args:
            data_file_path (Path): Data h5 file path.
            subject_list (np.ndarray): ist of all the subjects to include.
            exclude (list[int], optional): list of features to exclude from training. Cannot be used together with include. Defaults to None.
            include (list[int], optional): list of features to only include for training. Cannot be used together with exclude. Defaults to None.
        """
        assert (
            exclude is None or include is None
        ), "Only specify include or exclude, not both."

        logger.warning(
            "MRIDataset is very slow compared to MRIMemoryDataset, only use MRIDataset if you don't have enough memory. "
            + "You can enable the use of MRIMemoryDataset by setting --in_memory in the console"
        )

        self._exclude = exclude
        self._include = include
        self._data_file_path = data_file_path
        self._transform = transform

        with h5py.File(self.data_file_path, "r") as archive:
            self._scheme = archive.get("scheme")[()]
            indexes = archive.get("index")[()]

        # indexes of the data we want to load
        (self.selection, *_) = np.where(np.isin(indexes, subject_list))

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.selection)

    def __getitem__(self, index):
        """Generates one sample of data"""
        with h5py.File(self.data_file_path, "r") as archive:
            data = archive.get("data")[self.selection[index]]

        if self._include is not None:
            data = data[self._include]
        elif self._exclude is not None:
            data = np.delete(data, self._exclude)

        if self._transform:
            self.sample = self._transform(data=self.sample, scheme=self._scheme)

        if isinstance(self.sample, dict):
            return {
                "target": data,
                "sample": {k: v[index] for (k, v) in self.sample.items()},
            }
        else:
            return {"target": data, "sample": self.sample[index]}

    def __getstate__(self):
        """Return state values to be pickled."""
        return None

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        pass


class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_file: Path,
        batch_size: int = 256,
        subject_list_train: list[int] = [11, 12, 13, 14],
        subject_list_val: list[int] = [15],
        in_memory: bool = False,
        num_workers: int = 0,
    ):
        """Collection of train and validation data sets.

        Args:
            data_dir (Path): Path to the data directory.
            data_file_name (str): file name of the H5 file.
            header_file_name (str): file name of the CSV file.
            batch_size (int, optional): training batch size. Defaults to 265.
            subject_list_train (list[int], optional): subjects to include in training. Defaults to [11, 12, 13, 14].
            subject_list_val (list[int], optional): subject(s) to include in validation. Defaults to [15].
            in_memory (bool, optional): Whether to load the entire dataset in memory. Defaults to False.
            num_workers (bool, optional): Amount of threads to use when loading data from disk. Defaults to 0.
        """
        super(MRIDataModule, self).__init__()

        self._data_file = data_file
        self._batch_size = batch_size
        self._subject_list_train = np.array(subject_list_train)
        self._subject_list_val = np.array(subject_list_val)
        self._in_memory = in_memory
        self._num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        """Add model specific arguments to argparse.

        Args:
            parent_parser (ArgumentParser): parent argparse to add the new arguments to.

        Returns:
            ArgumentParser: parent argparse.
        """
        parser = parent_parser.add_argument_group("autoencoder.MRIDataModule")
        parser.add_argument(
            "--data_file",
            "-i",
            type=file_path,
            required=True,
            metavar="PATH",
            help="file name of the H5 file",
        )
        parser.add_argument(
            "--subject_train",
            nargs="+",
            type=int,
            help="subjects to include in training (default: [11, 12, 13, 14])",
        )
        parser.add_argument(
            "--subject_val",
            nargs="+",
            type=int,
            help="subjects to include in validation (default: [15])",
        )
        parser.add_argument(
            "--batch_size",
            default=256,
            type=int,
            metavar="N",
            help="input batch size for training (default: 256)",
        )
        parser.add_argument(
            "--in_memory",
            action="store_true",
            help="load the entire dataset into memory",
        )

        return parent_parser

    def setup(self, stage: Optional[str]) -> None:
        print("stage:", stage)
        DatasetClass = MRIMemoryDataset if self._in_memory else MRIDataset

        self.train_set = DatasetClass(
            self._data_file,
            self._subject_list_train,
        )
        self.val_set = DatasetClass(
            self._data_file,
            self._subject_list_val,
        )

    def train_dataloader(self) -> DataLoader:
        args = dict(
            batch_size=self._batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=0 if self._in_memory else self._num_workers,
            peristent_workers=not self._in_memory,
            drop_last=True,
        )

        return DataLoader(self.train_set, **args)

    def val_dataloader(self) -> DataLoader:
        args = dict(
            batch_size=self._batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=0 if self._in_memory else self._num_workers,
            peristent_workers=not self._in_memory,
            drop_last=True,
        )

        return DataLoader(self.val_set, **args)

    # def test_dataloader(self) -> DataLoader:
    #     if self.in_memory:
    #         return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
    #     else:
    #         return DataLoader(
    #             self.train_set,
    #             batch_size=self.batch_size,
    #             shuffle=False,
    #             num_workers=self.num_workers,
    #             pin_memory=True,
    #             persistent_workers=True,
    #         )
