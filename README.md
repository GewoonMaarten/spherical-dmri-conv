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

### Environment variables and .env file

Create a `.env` file in the root directory of this project.
`utils.env.py` will load file and set the variables.
If the file is not present it will use the system environment variables.

The following environment variables can be declared:

<table>
    <thead>
        <tr>
            <th>Name</th>
            <th>Description</th>
            <th>Default</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><code>DATA_PATH</code></td>
            <td>
                The path to the MUDI data. The directory structure is expected to be the same as provided by
                <a href="http://cmic.cs.ucl.ac.uk/cdmri20/challenge.html">Centre for the Developing Brain</a>.
            </td>
            <td></td>
        </tr>
        <tr>
            <td><code>LOGGING_LEVEL</code></td>
            <td>
                <table>
                    <thead>
                        <tr>
                            <th>Level</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>CRITICAL</td>
                            <td>50</td>
                        </tr>
                        <tr>
                            <td>ERROR</td>
                            <td>40</td>
                        </tr>
                        <tr>
                            <td>WARNING</td>
                            <td>30</td>
                        </tr>
                        <tr>
                            <td>INFO</td>
                            <td>20</td>
                        </tr>
                        <tr>
                            <td>DEBUG</td>
                            <td>10</td>
                        </tr>
                        <tr>
                            <td>NOTSET</td>
                            <td>0</td>
                        </tr>
                    </tbody>
                </table>
                It is recommend to set this <code>&gt;=20</code> when training the models for real.
            </td>
            <td>30</td>
        </tr>
    </tbody>

</table>

Example:

```
DATA_PATH=/home/user/MUDI
LOGGING_LEVEL=10
```

## Acknowledgements

The dataset was kindly provided by [Centre for the Developing Brain](http://cmic.cs.ucl.ac.uk/cdmri20/challenge.html).
