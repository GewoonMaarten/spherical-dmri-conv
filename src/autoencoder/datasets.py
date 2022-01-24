import copy
import itertools
import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import ChainDataset, DataLoader, IterableDataset

from autoencoder.logger import logger
from autoencoder.spherical.harmonics import gram_schmidt_sh_inv, sh_basis_real


class Transformer(ABC, object):
    @abstractmethod
    def pre_compute(self, **kwargs):
        pass


class DelimitTransformer(Transformer):
    def pre_compute(self, **kwargs):
        self._parameters = kwargs["parameters"]

    def __call__(self, **kwargs):
        data = kwargs["data"]
        scheme = kwargs["scheme"]

        b0_filter = scheme[:, 3] != 0.0

        scheme = scheme[b0_filter]
        data = data[b0_filter]

        b_s, b_counts = np.unique(scheme[:, 3], return_counts=True)
        ti_s = list()
        te_s = list()

        b_n = b_s.shape[0]
        ti_n = 1
        te_n = 1

        max_gradients = b_counts[0]

        if scheme.shape[1] > 4:
            ti_s = np.unique(scheme[:, 4])
            te_s = np.unique(scheme[:, 5])
            ti_n = ti_s.shape[0]
            te_n = te_s.shape[0]

            gradient_filter = (scheme[:, 4] == scheme[:, 4][0]) & (scheme[:, 5] == scheme[:, 5][0])
            max_gradients = np.max([data[(scheme[:, 3] == b) & gradient_filter].shape[0] for b in b_s])

        new_data = torch.zeros(ti_n, te_n, b_n * max_gradients, 1, 1, 1)
        for (ti_idx, ti), (te_idx, te), (b_idx, b) in itertools.product(
            enumerate(ti_s), enumerate(te_s), enumerate(b_s),
        ):
            filter_scheme = scheme[:, 3] == b
            if scheme.shape[1] > 4:
                filter_scheme = filter_scheme & (scheme[:, 4] == ti) & (scheme[:, 5] == te)

            if not np.any(filter_scheme):
                continue

            data_filtered = torch.from_numpy(data[filter_scheme])
            n_data_gradients = data_filtered.shape[0]
            new_data[
                ti_idx, te_idx, (b_idx * max_gradients) : (b_idx * max_gradients + n_data_gradients), 0, 0, 0
            ] = data_filtered

        return torch.flatten(new_data, start_dim=1, end_dim=3)


class SphericalTransformer(Transformer):
    def __init__(self, l_max: int = 0, symmetric: bool = True, inversion_n_iters: int = 1000) -> None:
        self._l_max = l_max
        self._symmetric = 2 if symmetric else 1
        self._inversion_n_iters = inversion_n_iters

    def pre_compute(self, **kwargs):
        self._parameters = kwargs["parameters"]

        # Fit the spherical harmonics on the gradients.
        y = sh_basis_real(torch.from_numpy(self._parameters[:, :3]), self._l_max)
        self._y_inv = gram_schmidt_sh_inv(y, self._l_max, n_iters=self._inversion_n_iters)

        self._b_s = np.unique(self._parameters[:, 3])
        self._ti_s = np.unique(self._parameters[:, 4]) if self._parameters.shape[1] > 4 else [1]
        self._te_s = np.unique(self._parameters[:, 5]) if self._parameters.shape[1] > 5 else [1]

        self._b_n = self._b_s.shape[0]
        self._ti_n = self._ti_s.shape[0]
        self._te_n = self._te_s.shape[0]

        l_sizes = dict()
        for l in range(0, self._l_max + 1, self._symmetric):
            o = 2 * l + 1
            l_sizes[l] = o

        self._filters = list()
        for (ti_idx, ti), (te_idx, te), b in itertools.product(enumerate(self._ti_s), enumerate(self._te_s), self._b_s):
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

        sh_coefficients_b_idx = dict()
        sh_coefficients = dict()
        for l in range(0, self._l_max + 1, self._symmetric):
            o = 2 * l + 1
            sh_coefficients_b_idx[l] = 0
            sh_coefficients[l] = torch.zeros((data.shape[0], self._ti_n, self._te_n, self._b_n, o))

        prev_b = [self._b_s[0]]
        for ti_idx, te_idx, b, l, l_size, filter in self._filters:
            # If we've visited all b values, we reset the counter
            if b in prev_b:
                sh_coefficients_b_idx = {k: 0 for k in sh_coefficients_b_idx}
                prev_b = [b]
            else:
                prev_b.append(b)

            data_filtered = torch.from_numpy(data[:, filter])
            y_inv_filtered = self._y_inv[:l_size, filter]

            sh_coefficient = torch.einsum("np,lp->nl", data_filtered, y_inv_filtered)
            # print(sh_coefficient.shape, sh_coefficients[l].shape)

            # Extract even covariants.
            s = 0
            for l in range(0, l + 1, self._symmetric):
                o = 2 * l + 1
                sh_coefficients[l][:, ti_idx, te_idx, sh_coefficients_b_idx[l]] = sh_coefficient[
                    :, torch.arange(s, s + o)
                ]

                s += o
                sh_coefficients_b_idx[l] += 1

        return sh_coefficients


class DiffusionMRIDataset(IterableDataset):
    tissues = ("cgm", "scgm", "wm", "csf", "pt")

    def __init__(
        self,
        parameters_file_path: Path,
        data_file_paths: List[Path],
        tissue: str,
        include_parameters: List[int] = None,
        exclude_parameters: List[int] = None,
        batch_size: int = 0,
        return_target: bool = False,
        transform: Union[SphericalTransformer, DelimitTransformer] = None,
    ) -> None:
        """Diffusion MRI dataset. Loads voxel data from HDF5 file fast.

        Args:
            parameters_file_path (Path): HDF5 file path that contains all parameters from the MRI data
            data_file_paths (List[Path]): list of HDF5 file paths containing voxel data
            tissue (str): The tissue to return. Can be the following values: `cgm`, `scgm`, `wm`, `csf`, and `pt`.
                where:
                    - `cgm`  = Cortical Grey Matter
                    - `scgm` = Sub-Cortical Grey Matter
                    - `wm`   = White Matter
                    - `pt`   = Pathological Tissue
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
        self._data_file_paths = data_file_paths
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
            self._parameters = self._parameters[self._selected_parameters]
        elif self._exclude_parameters is not None:
            self._selected_parameters = np.delete(self._selected_parameters, self._exclude_parameters)
            self._parameters = self._parameters[self._selected_parameters]

        if self._transform is not None:
            self._transform.pre_compute(parameters=self._parameters, batch_size=self._batch_size)

        self._metadata = list()  # stores metadata for each file

        self._total_batches = 0
        for p in self._data_file_paths:
            with h5py.File(p, "r", libver="latest") as archive:
                self._total_batches += self._dataset_size(archive, self._tissue)
                self._metadata.append(
                    (
                        self._total_batches,
                        {
                            "max_data": archive.attrs["max_data"],
                            "lstq_coefficient": archive.attrs["lstsq_coefficient"],
                            "id": archive.attrs["id"],
                        },
                    )
                )
        logger.info("created dataset with %d total batches", self._total_batches)

    def get_metadata(self, batch_id: int):
        for batch_range, metadata in self._metadata:
            if batch_id <= batch_range:
                return metadata

        # TODO: it should never reach here, but sometimes it does. The self._total_batches is not quite right.
        return self._metadata[-1][1]

    def _dataset_size(self, archive: h5py.File, tissue: str) -> int:
        dim = archive[tissue].shape[0]
        num_batches = dim

        if self._batch_size > 0:
            num_batches = dim // self._batch_size
            num_batches += 1 if dim % self._batch_size > 0 else 0

        return num_batches

    def _iter_tissue(self, archive: h5py.File, tissue: str):
        dim = archive[tissue].shape[0]

        if self._batch_size > 0:
            num_batches = dim // self._batch_size
            num_batches += 1 if dim % self._batch_size > 0 else 0
            for i in range(num_batches):
                batch_start = self._batch_size * i
                batch_end = min(batch_start + self._batch_size, dim)
                batch = archive[tissue][batch_start:batch_end]
                sample = batch[:, self._selected_parameters]

                if self._transform is not None:
                    sample = self._transform(data=sample, scheme=self._selected_parameters)

                yield {
                    "sample": sample,
                    "target": batch if self._return_target else None,
                }
        else:
            for i in range(dim):
                item = archive[tissue][i]
                sample = item[None, self._selected_parameters]

                if self._transform is not None:
                    sample = self._transform(data=sample, scheme=self._selected_parameters)

                yield {
                    "sample": sample,
                    "target": item[None, :] if self._return_target else None,
                }

    def _configure_worker(self, worker_info):
        if worker_info is not None:
            per_worker = math.ceil(len(self._data_file_paths) / worker_info.num_workers)
            worker_id = worker_info.id

            work_start = worker_id * per_worker
            work_end = min(work_start + per_worker, len(self._data_file_paths))

            self._data_file_paths = self._data_file_paths[work_start:work_end]

    # def __len__(self):
    #     return self._total_batches

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._configure_worker(worker_info)

        iters = list()
        for p in self._data_file_paths:
            archive = h5py.File(p, "r", libver="latest")
            iters.append(self._iter_tissue(archive, self._tissue))

        return itertools.chain.from_iterable(iters)

    def __getstate__(self):
        """Return state values to be pickled."""
        return None

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        pass


@DATAMODULE_REGISTRY
class MRIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        parameters_file_path: str,
        data_file_paths: List[str],
        include_parameters: str = None,
        exclude_parameters: str = None,
        return_target: bool = False,
        transform: Union[SphericalTransformer, DelimitTransformer] = None,
        batch_size: int = 0,
        num_workers: int = 0,
    ):
        super(MRIDataModule, self).__init__()

        self._parameters_file_path = Path(parameters_file_path)
        self._data_file_paths = [Path(p) for p in data_file_paths]
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
            self.train_dataset = ChainDataset(
                [
                    DiffusionMRIDataset(self._parameters_file_path, self._data_file_paths[:-1], tissue, **common_args)
                    for tissue in ("csf", "cgm", "scgm", "wm")
                ]
            )
            self.val_dataset = ChainDataset(
                [
                    DiffusionMRIDataset(self._parameters_file_path, self._data_file_paths[-2:], tissue, **common_args)
                    for tissue in ("csf", "cgm", "scgm", "wm")
                ]
            )
        # Test on individual tissues
        if stage == "test" or stage is None:
            self.test_dataset_csf = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_paths[-1:], "csf", **common_args
            )
            self.test_dataset_cgm = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_paths[-1:], "cgm", **common_args
            )
            self.test_dataset_scgm = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_paths[-1:], "scgm", **common_args
            )
            self.test_dataset_wm = DiffusionMRIDataset(
                self._parameters_file_path, self._data_file_paths[-1:], "wm", **common_args
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self._data_loader_args)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self._data_loader_args)

    def test_dataloader(self) -> DataLoader:
        return [
            DataLoader(self.test_dataset_csf, **self._data_loader_args),
            DataLoader(self.test_dataset_scgm, **self._data_loader_args),
            DataLoader(self.test_dataset_cgm, **self._data_loader_args),
            DataLoader(self.test_dataset_wm, **self._data_loader_args),
            DataLoader(self.val_dataset, **self._data_loader_args),
        ]
