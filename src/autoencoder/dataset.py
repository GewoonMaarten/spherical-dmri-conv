import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import psutil
from torch.utils.data import Dataset

from utils.logger import logger


class MRISelectorSubjDataset(Dataset):
    """MRI dataset to select features from."""

    def __init__(self, root_dir, dataf, headerf, subj_list, exclude=[]):
        """
        batch_size & shuffle are defined with 'DataLoader' in pytorch

        Args:
            root_dir (string): Directory with the data and header files
            data (string): Data h5 file
            header (string): Header csv file
            subj_list (list): list of all the subjects to include
            exclude (list): list of features to exclude from training
        """

        header_path = Path(root_dir, headerf)
        data_path = Path(root_dir, dataf)

        # load the header
        header = pd.read_csv(header_path, index_col=0)
        header = header[header["1"].isin(subj_list)]
        # indexes of the data we want to load
        indexes = header["0"].to_numpy(dtype=np.uint32)

        # load the data in memory. The total file is *only* 3GB so it should be
        # doable on most systems. Lets check anyway...
        file_size = os.path.getsize(data_path)
        available_memory = psutil.virtual_memory().available
        if available_memory - file_size < 0:
            logger.warning(
                "Data file requires %s bytes of memory but %s was available",
                format(file_size, ","),
                format(available_memory, ","),
            )

        archive = h5py.File(data_path, "r")
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
