import os
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import psutil
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from utils.logger import logger


class MRISelectorSubjDataset(Dataset):
    def __init__(
        self,
        data_file_path: Path,
        header_file_path: Path,
        subject_list: np.ndarray,
        exclude: list[int] = [],
    ):
        """Create a dataset from the selected subjects in the subject list

        Args:
            data_file_path (Path): Data h5 file path.
            header_file_path (Path): Header csv file path.
            subject_list (np.ndarray): ist of all the subjects to include.
            exclude (list[int], optional): list of features to exclude from training. Defaults to [].
        """

        # load the header
        header = pd.read_csv(header_file_path, index_col=0)
        header = header[header["1"].isin(subject_list)]
        # indexes of the data we want to load
        indexes = header["0"].to_numpy(dtype=np.uint32)

        # load the data in memory. The total file is *only* 3GB so it should be
        # doable on most systems. Lets check anyway...
        file_size = os.path.getsize(data_file_path)
        available_memory = psutil.virtual_memory().available
        if available_memory - file_size < 0:
            logger.warning(
                "Data file requires %s bytes of memory but %s was available",
                format(file_size, ","),
                format(available_memory, ","),
            )

        archive = h5py.File(data_file_path, "r")
        self.data = archive.get("data1")[
            indexes,
        ]
        # delete excluded features
        self.data = np.delete(self.data, exclude, axis=1)

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data"""
        return self.data[index]


class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        data_file_name: str,
        header_file_name: str,
        batch_size: int = 265,
        subject_list_train: list[int] = [11, 12, 13, 14],
        subject_list_val: list[int] = [15],
    ):
        """Collection of train and validation data sets.

        Args:
            data_dir (Path): Path to the data directory.
            data_file_name (str): file name of the H5 file.
            header_file_name (str): file name of the CSV file.
            batch_size (int, optional): training batch size. Defaults to 265.
            subject_list_train (list[int], optional): subjects to include in training. Defaults to [11, 12, 13, 14].
            subject_list_val (list[int], optional): subject(s) to include in validation. Defaults to [15].
        """
        super(MRIDataModule, self).__init__()
        self.save_hyperparameters()

        self.data_file_path = Path(data_dir, data_file_name)
        self.header_file_path = Path(data_dir, header_file_name)
        self.batch_size = batch_size
        self.subject_list_train = np.array(subject_list_train)
        self.subject_list_val = np.array(subject_list_val)

    def setup(self, stage: Optional[str]) -> None:

        self.train_set = MRISelectorSubjDataset(
            self.data_file_path,
            self.header_file_path,
            self.subject_list_train,
        )
        self.val_set = MRISelectorSubjDataset(
            self.data_file_path,
            self.header_file_path,
            self.subject_list_val,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
        )
