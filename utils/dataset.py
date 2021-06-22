from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MRISelectorSubjDataset(Dataset):
    """MRI dataset to select features from."""

    def __init__(self, root_dir, dataf, headerf, subj_list):
        """
        batch_size & shuffle are defined with 'DataLoader' in pytorch 

        Args:
            root_dir (string): Directory with the data and header files
            data (string): Data h5 file
            header (string): Header csv file
            subj_list (list): list of all the subjects to include
        """
        super().__init__()

        header_path = Path(root_dir, headerf)
        data_path = Path(root_dir, dataf)

        # load the header
        header = pd.read_csv(header_path, index_col=0).to_numpy()
        self.ind = header[np.isin(header[:, 1], subj_list), 0]
        self.indexes = np.arange(len(self.ind))

        # load the data in memory. The file is *only* 3GB so it should be
        # doable on most systems.
        archive = h5py.File(data_path, 'r')
        self.data = archive.get('data1')[:]

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.ind)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Find list of IDs
        list_IDs_temp = self.ind[self.indexes[index]]
        return self.data[list_IDs_temp, :]
