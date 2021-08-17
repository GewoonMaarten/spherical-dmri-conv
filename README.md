# Rotation Equivariant MUDI

> Sub-sampling on multi-dimensional diffusion MRI with rotation equivariance.

[![Made with Pytorch](https://img.shields.io/badge/MADE%20WITH-pytorch-red?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

## Notebooks

- `concrete_autoencoder_pytorch.ipynb`.
  This notebook is used to train various concrete autoencoders on de multi-dimensional diffusion MR scans.
- `data_preprocessing.ipynb`.
  This notebook is used to turn the multi-dimensional diffusion MR scans into usable data for the concrete autoencoder.
- `model-analysis.ipynb`.
  This notebook is used to evaluate the accuracies of the learned models.

## Running the notebooks

### Conda/Miniconda

It is recommended that you use Conda or Miniconda.
An environment file is provided with the necessary dependencies.
You can install it with:

```console
conda env create -f environment.yml
```

Or if you already have a Conda environment:

```console
conda env update -n my_env -f environment.yml
```

Install the project with:

```console
python -m pip install -e .
```

### Training

`trainer.py` can be used as a CLI for training models.

```console
python trainer.py --help
```

Example:

```console
python trainer.py \
    --data_file ./data/data.hdf5 \
    --header_file ./data/header.csv \
    --input_output_size 1344 \
    --latent_size 500 \
    --gpus=1 \
    --max_epochs 10
```

The output can be viewed with Tensorboard:

```console
tensorboard --logdir=./lightning_logs/
```

Or MLFlow:

```console
mlflow ui
```

## Acknowledgements

The dataset was kindly provided by [Centre for the Developing Brain](http://cmic.cs.ucl.ac.uk/cdmri20/challenge.html).
