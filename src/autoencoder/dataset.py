import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import psutil
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from utils.argparse import file_path
from utils.logger import logger


class MRIMemoryDataset(Dataset):
    def __init__(
        self,
        data_file_path: Path,
        subject_list: np.ndarray,
        exclude: list[int] = [],
    ):
        """Create a dataset from the selected subjects in the subject list

        Args:
            data_file_path (Path): Data h5 file path.
            subject_list (np.ndarray): ist of all the subjects to include.
            exclude (list[int], optional): list of features to exclude from
            training. Defaults to [].
        """
        archive = h5py.File(data_file_path, "r")

        indexes = archive.get("index")[()]
        # indexes of the data we want to load
        (selection, *_) = np.where(np.isin(indexes, subject_list))

        # load the data in memory. The total file is *only* 3.1GB so it should be
        # doable on most systems. Lets check anyway...
        file_size = os.path.getsize(data_file_path)
        available_memory = psutil.virtual_memory().available
        if available_memory - file_size < 0:
            logger.warning(
                "Data file requires %s bytes of memory but %s was available",
                format(file_size, ","),
                format(available_memory, ","),
            )

        self.data = archive.get("data")[selection]
        # delete excluded features
        self.data = np.delete(self.data, exclude, axis=1)

        archive.close()

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.data[index]

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
        subject_list: np.ndarray,
        exclude: list[int] = [],
    ):
        """Create a dataset from the selected subjects in the subject list

        Args:
            data_file_path (Path): Data h5 file path.
            subject_list (np.ndarray): ist of all the subjects to include.
            exclude (list[int], optional): list of features to exclude from
            training. Defaults to [].
        """
        logger.warning(
            "This dataset is very slow compared to MRIMemoryDataset, only use this if you don't have enough memory. "
            + "You can enable MRIMemoryDataset by using --in_memory in the console"
        )

        self.data_file_path = data_file_path
        self.exclude = exclude

        with h5py.File(self.data_file_path, "r") as archive:
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
        data = np.delete(data, self.exclude)

        return data

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
    ):
        """Collection of train and validation data sets.

        Args:
            data_dir (Path): Path to the data directory.
            data_file_name (str): file name of the H5 file.
            header_file_name (str): file name of the CSV file.
            batch_size (int, optional): training batch size. Defaults to 265.
            subject_list_train (list[int], optional): subjects to include in
            training. Defaults to [11, 12, 13, 14].
            subject_list_val (list[int], optional): subject(s) to include in
            validation. Defaults to [15].
            in_memory (bool): Whether to load the entire dataset in memory.
            Defaults to False.
        """
        super(MRIDataModule, self).__init__()

        self.data_file = data_file
        self.batch_size = batch_size
        self.subject_list_train = np.array(subject_list_train)
        self.subject_list_val = np.array(subject_list_val)
        self.in_memory = in_memory

        # Only assign 2 workers if Python is running on Windows (nt).
        self.num_workers = 2 if os.name == "nt" else os.cpu_count()

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
            help="input batch size for training (default: 64)",
        )
        parser.add_argument(
            "--in_memory",
            action="store_true",
            help="load the entire dataset into memory",
        )

        return parent_parser

    def setup(self, stage: Optional[str]) -> None:
        DatasetClass = MRIMemoryDataset if self.in_memory else MRIDataset

        self.train_set = DatasetClass(
            self.data_file,
            self.subject_list_train,
        )
        self.val_set = DatasetClass(
            self.data_file,
            self.subject_list_val,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
