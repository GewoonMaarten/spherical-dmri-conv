import itertools
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import DataLoader, Dataset


class Transformer(ABC, object):
    @abstractmethod
    def pre_compute(self, **kwargs):
        pass


class SphericalTransformer(Transformer):
    def __init__(self, l_max: int = 0, symmetric: bool = True, inversion_n_iters: int = 1000) -> None:
        self._l_max = l_max
        self._symmetric = 2 if symmetric else 1
        self._inversion_n_iters = inversion_n_iters

    def pre_compute(self, **kwargs):
        self._parameters = kwargs["parameters"]

        # Fit the spherical harmonics on the gradients.
        # y = sh_basis_real(torch.from_numpy(self._parameters[:, :3]), self._l_max)
        # self._y_inv = gram_schmidt_sh_inv(y, self._l_max, n_iters=self._inversion_n_iters)

        self._b_s = np.unique(self._parameters[:, 3])
        self._ti_s = np.unique(self._parameters[:, 4]) if self._parameters.shape[1] > 4 else np.array([1])
        self._te_s = np.unique(self._parameters[:, 5]) if self._parameters.shape[1] > 5 else np.array([1])

        self._b_n = self._b_s.shape[0]
        self._ti_n = self._ti_s.shape[0]
        self._te_n = self._te_s.shape[0]

        l_sizes = dict()
        for l in range(0, self._l_max + 1, self._symmetric):
            o = 2 * l + 1
            l_sizes[l] = o

        self._filters = list()
        for (ti_idx, ti), (te_idx, te), b in itertools.product(enumerate(self._ti_s), enumerate(self._te_s), self._b_s):
            if b == 0:
                continue
            filter = self._parameters[:, 3] == b
            if self._parameters.shape[1] > 4:
                filter = filter & (self._parameters[:, 4] == ti) & (self._parameters[:, 5] == te)

            num_voxels = filter.sum()  # number of voxels left after filtering
            if num_voxels == 0:
                continue

            # Get the maximum l where the amount of coefficients is smaller or equal to the amount of voxels.
            l = [l for l, o in l_sizes.items() if o <= num_voxels][-1]
            l_size = np.sum([2 * s + 1 for s in range(0, l + 1, 2)])

            self._filters.append((ti_idx, te_idx, b, l, l_size, filter))

    def __call__(self, **kwargs):
        data = kwargs["data"]

        grouped_data = np.empty((data.shape[0], 90, 3))
        for idx, (_, _, _, _, _, filter) in enumerate(self._filters):
            grouped_data[:, :, idx] = data[:, filter]
        # sh_coefficients_b_idx = dict()
        # sh_coefficients = dict()
        # for l in range(0, self._l_max + 1, self._symmetric):
        #     o = 2 * l + 1
        #     sh_coefficients_b_idx[l] = 0
        #     sh_coefficients[l] = torch.zeros((data.shape[0], self._ti_n, self._te_n, self._b_n, o))

        # prev_b = [self._b_s[0]]
        # for ti_idx, te_idx, b, l, l_size, filter in self._filters:
        #     # If we've visited all b values, we reset the counter
        #     if b in prev_b:
        #         sh_coefficients_b_idx = {k: 0 for k in sh_coefficients_b_idx}
        #         prev_b = [b]
        #     else:
        #         prev_b.append(b)

        #     data_filtered = torch.from_numpy(data[:, filter])
        #     y_inv_filtered = self._y_inv[:l_size, filter]

        #     sh_coefficient = torch.einsum("np,lp->nl", data_filtered, y_inv_filtered)
        #     # print(sh_coefficient.shape, sh_coefficients[l].shape)

        #     # Extract even covariants.
        #     s = 0
        #     for l in range(0, l + 1, self._symmetric):
        #         o = 2 * l + 1
        #         sh_coefficients[l][:, ti_idx, te_idx, sh_coefficients_b_idx[l]] = sh_coefficient[
        #             :, torch.arange(s, s + o)
        #         ]

        #         s += o
        #         sh_coefficients_b_idx[l] += 1

        return grouped_data


class DiffusionMRIDataset(Dataset):
    tissues = ("wb", "gm", "wm", "csf")

    def __init__(
        self,
        parameters_file_path: Path,
        data_file_path: Path,
        subject_list,
        tissue: str,
        include_parameters: List[int] = None,
        exclude_parameters: List[int] = None,
        batch_size: int = 1,
        return_target: bool = False,
        transform: SphericalTransformer = None,
    ) -> None:
        """Diffusion MRI dataset. Loads voxel data from HDF5 file fast.

        Args:
            parameters_file_path (Path): HDF5 file path that contains all parameters from the MRI data
            data_file_paths (List[Path]): list of HDF5 file paths containing voxel data
            tissue (str): The tissue to return. Can be the following values: `wb`, `gm`, `wm`, and `csf`.
                where:
                    - `wb`  = Whole Brain
                    - `cgm` = Grey Matter
                    - `wm`  = White Matter
                    - `csf` = Cerebral Spinal Fluid
                These tissue types should be created beforehand with MRTrix3 `5ttgen` tool.
            include_parameters (List[int], optional): parameters to *only* include in the dataset. Defaults to None.
            exclude_parameters (List[int], optional): parameters to exclude from the dataset. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 0.
            return_target (bool, optional): return target data with all parameters included. Useful for loss calculation
                when recreating the dataset from a subset of parameters. Defaults to False.
            transform (Union[SphericalTransformer, DelimitTransformer], optional): Used to transform the data for models
                that require different data. Defaults to None.
        """
        super(DiffusionMRIDataset).__init__()

        assert tissue in self.tissues, f"Unknown tissue value: {tissue}"
        assert include_parameters is None or exclude_parameters is None, "Only specify include or exclude, not both."

        self._parameters_file_path = parameters_file_path
        self._data_file_path = data_file_path
        self._subject_list = subject_list
        self._tissue = tissue
        self._include_parameters = include_parameters
        self._exclude_parameters = exclude_parameters
        self._batch_size = batch_size
        self._return_target = return_target
        self._transform = transform

        with h5py.File(self._parameters_file_path, "r", libver="latest") as archive:
            self._parameters = archive["parameters"][...]

        self._selected_parameters = np.arange(self._parameters.shape[0])
        if self._include_parameters is not None:
            self._selected_parameters = self._selected_parameters[self._include_parameters]
            # self._parameters = self._parameters[self._selected_parameters]
        elif self._exclude_parameters is not None:
            self._selected_parameters = np.delete(self._selected_parameters, self._exclude_parameters)
            # self._parameters = self._parameters[self._selected_parameters]

        if self._transform is not None:
            self._transform.pre_compute(parameters=self._parameters, batch_size=self._batch_size)

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

            if self._transform is not None:
                data_filtered = np.zeros_like(data)
                data_filtered[:, self._selected_parameters] = data[:, self._selected_parameters]
                data_filtered = self._transform(data=data_filtered, scheme=self._selected_parameters)
                data = self._transform(data=data, scheme=self._selected_parameters)
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
        transform: SphericalTransformer = None,
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
        self._transform = transform

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
            transform=self._transform,
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
