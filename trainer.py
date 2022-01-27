import os

import mlflow
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY, LightningCLI
import copy
import autoencoder.datasets
import autoencoder.models
from autoencoder.logger import set_log_level

if __name__ == "__main__":
    set_log_level(10)

    MODEL_REGISTRY.register_classes(autoencoder.models, pl.LightningModule, override=True)
    DATAMODULE_REGISTRY.register_classes(autoencoder.datasets, pl.LightningDataModule, override=True)

    cli = LightningCLI(run=False, save_config_overwrite=True)

    experiment_name = cli.model.__class__.__name__

    if "MLFLOW_ENDPOINT_URL" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT_URL"])

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    mlflow.log_params(cli.config["model"]["init_args"])
    mlflow.log_params(cli.config["data"]["init_args"])

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

    if experiment_name == "ConcreteAutoencoder":
        latent_features_file_path = "logs/latent_features.txt"
        np.savetxt(
            latent_features_file_path, np.array(cli.model.encoder.latent_features, dtype=int), fmt="%d",
        )
        mlflow.log_artifact(latent_features_file_path)
