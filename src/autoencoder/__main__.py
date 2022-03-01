import os

import mlflow
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY, MODEL_REGISTRY, LightningCLI

import autoencoder.datasets
import autoencoder.models
from autoencoder.logger import set_log_level
import argparse


class kwargs_append_action(argparse.Action):
    """argparse action to split an argument into KEY=VALUE form on append to a dictionary."""

    def __call__(self, parser, args, values, option_string=None):
        try:
            d = dict(map(lambda x: x.split("="), values))
        except ValueError as ex:
            raise argparse.ArgumentError(self, f'Could not parse argument "{values}" as k1=v1 k2=v2 ... format')
        setattr(args, self.dest, d)


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--tags",
            dest="tags",
            nargs="*",
            required=False,
            action=kwargs_append_action,
            metavar="KEY=VALUE",
            help="Add key/value tags. Used to tag MLflow runs. May appear multiple times. Aggregate in dict",
        )


if __name__ == "__main__":
    set_log_level(10)

    MODEL_REGISTRY.register_classes(autoencoder.models, pl.LightningModule, override=True)
    DATAMODULE_REGISTRY.register_classes(autoencoder.datasets, pl.LightningDataModule, override=True)

    cli = MyLightningCLI(run=False, save_config_overwrite=True)

    experiment_name = cli.model.__class__.__name__

    if "MLFLOW_ENDPOINT_URL" in os.environ:
        mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT_URL"])

    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    mlflow.log_params(cli.config["model"]["init_args"])
    mlflow.log_params(cli.config["data"]["init_args"])
    if "tags" in cli.config.keys():
        mlflow.set_tags(cli.config["tags"])

    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)

    scripted_pytorch_model = torch.jit.script(cli.model)
    mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

    if experiment_name == "ConcreteAutoencoder":
        latent_features_file_path = "logs/latent_features.txt"
        np.savetxt(
            latent_features_file_path,
            np.array(cli.model.encoder.latent_features, dtype=int),
            fmt="%d",
        )
        mlflow.log_artifact(latent_features_file_path)
    else:
        latent_features_file_path = "logs/latent_features.txt"
        np.savetxt(latent_features_file_path, cli.datamodule.val_dataset._selected_parameters, fmt="%d")
        mlflow.log_artifact(latent_features_file_path)
