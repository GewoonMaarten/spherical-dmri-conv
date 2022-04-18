from pathlib import Path
from typing import List, Literal, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import DataLoader, Dataset

from autoencoder.spherical.transform import group_te_ti_b_values


class DiffusionMRIDataset(Dataset):
    def __init__(
        self,
        parameters_file_path: Path,
        data_file_path: Path,
        subject_list: np.ndarray,
        tissue: Literal["wb", "gm", "wm", "csf"],
        include_parameters: List[int] = None,
        exclude_parameters: List[int] = None,
        batch_size: int = 1,
        return_target: bool = False,
        use_spherical_data: bool = False,
    ) -> None:
        """Diffusion MRI dataset. Loads voxel data from HDF5 file fast.

        Args:
            parameters_file_path: HDF5 file path that contains all parameters from the MRI data
            data_file_path: HDF5 file path containing voxel data
            subject_list: list of subjects to create the dataset with.
            tissue: The tissue to return. Can be the following values: ``wb``, ``gm``, ``wm``, and ``csf``.
                where:
                    - ``wb`` = Whole Brain
                    - ``gm`` = Grey Matter
                    - ``wm`` = White Matter
                    - ``csf`` = Cerebral Spinal Fluid
                These tissue types should be created beforehand with MRTrix3 ``5ttgen`` tool.
            include_parameters: parameters to *only* include in the dataset. Defaults to None.
            exclude_parameters: parameters to exclude from the dataset. Defaults to None.
            batch_size: batch size. Defaults to 0.
            return_target : return target data with all parameters included. Useful for loss calculation when recreating
                the dataset from a subset of parameters. Defaults to False.
            use_spherical_data: Used to transform the data for models that require spherical data. Defaults to False.
        """
        super(DiffusionMRIDataset).__init__()

        assert tissue in ("wb", "gm", "wm", "csf"), f"Unknown tissue value: {tissue}"
        assert include_parameters is None or exclude_parameters is None, "Only specify include or exclude, not both."

        self._parameters_file_path = parameters_file_path
        self._data_file_path = data_file_path
        self._subject_list = subject_list
        self._tissue = tissue
        self._include_parameters = include_parameters
        self._exclude_parameters = exclude_parameters
        self._batch_size = batch_size
        self._return_target = return_target
        self._use_spherical_data = use_spherical_data

        with h5py.File(self._parameters_file_path, "r", libver="latest") as archive:
            self._parameters = archive["parameters"][...]

        self._selected_parameters = np.arange(self._parameters.shape[0])
        if self._include_parameters is not None:
            self._selected_parameters = self._selected_parameters[self._include_parameters]
            # self._parameters = self._parameters[self._selected_parameters]
        elif self._exclude_parameters is not None:
            self._selected_parameters = np.delete(self._selected_parameters, self._exclude_parameters)
            # self._parameters = self._parameters[self._selected_parameters]

        self.num_batches = 0
        with h5py.File(self._data_file_path, "r") as archive:
            tissue_id = archive.get("masks").attrs[self._tissue]
            tissue_mask = np.isin(archive.get("masks"), tissue_id)
            index_mask = np.isin(archive.get("index"), self._subject_list)
            (self.selection, *_) = np.where(tissue_mask & index_mask)

            self.dim = len(self.selection)
            self.num_batches = self.dim // self._batch_size
            self.num_batches += 1 if self.dim % self._batch_size > 0 else 0

    def get_subject_id_by_batch_id(self, batch_id: int) -> int:
        with h5py.File(self._data_file_path, "r") as archive:
            return archive.get("index")[batch_id * self._batch_size]

    def get_metadata_by_subject_id(self, subject_id: int):
        with h5py.File(self._data_file_path, "r") as archive:
            metadata_id = archive["data"].attrs[str(subject_id)]
            return {
                "max_data": archive["data"].attrs["max_data"][metadata_id],
                "lstsq_coefficient": archive["data"].attrs["lstsq_coefficient"][metadata_id],
                "tissue": self._tissue,
            }

    def __len__(self):
        return self.num_batches

    def __getitem__(self, index):
        """Generates one sample of data"""
        with h5py.File(self._data_file_path, "r") as archive:
            batch_start = self._batch_size * index
            batch_end = min(batch_start + self._batch_size, self.dim)

            data = archive["data"][self.selection[batch_start:batch_end]]

            if self._use_spherical_data:
                data_filtered = np.zeros_like(data)
                data_filtered[:, self._selected_parameters] = data[:, self._selected_parameters]

                data_filtered = group_te_ti_b_values(self._parameters, data_filtered)[1]
                data = group_te_ti_b_values(self._parameters, data)[1]
                data = torch.flatten(torch.from_numpy(data).float(), start_dim=1)
            else:
                data_filtered = data[:, self._selected_parameters]

            if self._return_target:
                return {"target": data, "sample": data_filtered}
            else:
                return {"sample": data_filtered}


@DATAMODULE_REGISTRY
class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        parameters_file_path: str,
        data_file_path: str,
        train_subject_ids: List[int],
        validate_subject_ids: List[int],
        include_parameters: str = None,
        exclude_parameters: str = None,
        return_target: bool = False,
        use_spherical_data: bool = False,
        batch_size: int = 0,
        num_workers: int = 0,
    ):
        super(MRIDataModule, self).__init__()

        self._parameters_file_path = Path(parameters_file_path)
        self._data_file_path = Path(data_file_path)
        self._train_subject_ids = train_subject_ids
        self._validate_subject_ids = validate_subject_ids
        self._batch_size = batch_size
        self._include_parameters = include_parameters
        self._exclude_parameters = exclude_parameters
        self._return_target = return_target
        self._num_workers = num_workers
        self._use_spherical_data = use_spherical_data

        if self._exclude_parameters is not None:
            self._exclude_parameters = np.loadtxt(self._exclude_parameters, dtype=np.int32)
        elif self._include_parameters is not None:
            self._include_parameters = np.loadtxt(self._include_parameters, dtype=np.int32)

        self._data_loader_args = dict(
            batch_size=None,
            batch_sampler=None,
            num_workers=self._num_workers,
            persistent_workers=True if self._num_workers > 0 else False,
            pin_memory=True,
        )

    def setup(self, stage: Optional[str]) -> None:
        common_args = dict(
            include_parameters=self._include_parameters,
            exclude_parameters=self._exclude_parameters,
            batch_size=self._batch_size,
            return_target=self._return_target,
            use_spherical_data=self._use_spherical_data,
        )

        # Train on the whole brain
        if stage == "fit" or stage == "validate" or stage is None:
            self.train_dataset = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_path, self._train_subject_ids, "wb", **common_args
            )
            self.val_dataset = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_path, self._validate_subject_ids, "wb", **common_args
            )

        # Test on individual tissues
        if stage == "test" or stage is None:
            self.test_dataset_wb = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_path, self._validate_subject_ids, "wb", **common_args
            )
            self.test_dataset_csf = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_path, self._validate_subject_ids, "csf", **common_args
            )
            self.test_dataset_gm = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_path, self._validate_subject_ids, "gm", **common_args
            )
            self.test_dataset_wm = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_path, self._validate_subject_ids, "wm", **common_args
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self._data_loader_args)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self._data_loader_args)

    def test_dataloader(self) -> DataLoader:
        return [
            DataLoader(self.test_dataset_csf, **self._data_loader_args),
            DataLoader(self.test_dataset_wb, **self._data_loader_args),
            DataLoader(self.test_dataset_wm, **self._data_loader_args),
            DataLoader(self.test_dataset_gm, **self._data_loader_args),
        ]
