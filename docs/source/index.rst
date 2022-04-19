.. Concrete Autoencoder documentation master file, created by
   sphinx-quickstart on Tue Mar  1 16:16:35 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Concrete Autoencoder documentation!
================================================
Sub-sampling on multi-dimensional diffusion MRI with rotation equivariance.
  
.. figure:: ./_static/connectome.jpg
   
   The human connectome. Credit: Human Connectome Project


.. toctree::
   :maxdepth: 2
   :glob:


   autoencoder

Getting started
================

Installing the project
----------------------

.. code-block:: bash

   python -m pip install -e .

This installs all dependencies to run the project. Install the development dependencies with:

.. code-block:: bash

   python -m pip install -e .[dev]

Running the project
-------------------

`PyTorch Lightning CLI`_ is used to generate a command line interface which you can use to run the project.
This CLI is automatically generated based on the parameters required by each model and dataset.
The values for the parameters can be provided via config files or command line arguments. 
Several config files are provided in the `configs` folder:

 - ``trainer.yaml``: this config file contains basic trainer parameters, like which GPU to use. 
   This config file should be included in every training session.
 - ``mudi_data.yaml``: contains the configuration for the MUDI dataset.
   Used if you want to train with the MUDI dataset.
 - ``mudi_concrete_autoencder.yaml``: config for the Concrete autoencoder with MUDI specific parameters.
 - ``mudi_fcn_decoder.yaml``: config for the Fully Connected Decoder with MUDI specific parameters.
 - ``mudi_spherical_decoder.yaml``: config for the Spherical Decoder with MUDI specific parameters.
 - ``hcp_data.yaml``: contains the configuration for the HCP dataset.
   Used if you want to train with the HCP dataset.
 - ``hcp_concrete_autoencder.yaml``: config for the Concrete autoencoder with HCP specific parameters.
 - ``hcp_fcn_decoder.yaml``: config for the Fully Connected Decoder with HCP specific parameters.
 - ``hcp_spherical_decoder.yaml``: config for the Spherical Decoder with HCP specific parameters.

Multiple configs can be chained:

.. code-block:: bash

   python -m autoencoder \
      --config ./configs/trainer.yaml \
      --config ./configs/hcp_data.yaml \
      --config ./configs/hcp_fcn_decoder.yaml

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Bibliography
============

.. bibliography::

.. _Pytorch Lightning CLI: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html