import os
from argparse import ArgumentParser, Namespace

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from autoencoder.concrete_autoencoder import ConcreteAutoencoder
from autoencoder.dataset import MRIDataModule
from autoencoder.logger import logger, set_log_level


def trainer(args: Namespace) -> None:
    """Take command line arguments to train a model.

    Args:
        args (Namespace): arguments from argparse
    """
    experiment_name = "concrete_autoencoder"

    set_log_level(args.verbose)
    is_verbose = args.verbose < 30

    logger.info("Start training with params: %s", str(args.__dict__))

    model = ConcreteAutoencoder(
        args.input_output_size,
        args.latent_size,
        args.decoder_hidden_layers,
        learning_rate=args.learning_rate,
        lambda_reg=args.lambda_reg,
    )

    if args.checkpoint is not None:
        logger.info("Loading from checkpoint")
        model = model.load_from_checkpoint(
            str(args.checkpoint), hparams_file=str(args.hparams)
        )

    dm = MRIDataModule(
        data_file=args.data_file,
        batch_size=args.batch_size,
        in_memory=args.in_memory,
    )

    plugins = []
    if args.accelerator == "ddp":
        plugins = [
            DDPPlugin(find_unused_parameters=False, gradient_as_bucket_view=True)
        ]

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            EarlyStopping(
                monitor="mean_max",
                mode="max",
                patience=float("inf"),
                stopping_threshold=0.998,
                verbose=is_verbose,
            ),
            ModelCheckpoint(
                monitor="mean_max",
                mode="max",
                save_top_k=1,
                verbose=is_verbose,
            ),
        ],
        checkpoint_callback=True,
        logger=TensorBoardLogger("logs", name=experiment_name),
        plugins=plugins,
    )

    mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT_URL"])
    mlflow.set_experiment(experiment_name)
    mlflow.pytorch.autolog()
    mlflow.log_params(vars(args))

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Concrete Autoencoder trainer", usage="%(prog)s [options]"
    )

    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        choices=[0, 10, 20, 30, 40, 50],
        default=20,
        metavar="XX",
        help="verbosity level (default: 10)",
    )

    parser = pl.Trainer.add_argparse_args(parent_parser=parser)
    parser = ConcreteAutoencoder.add_model_specific_args(parent_parser=parser)
    parser = MRIDataModule.add_model_specific_args(parent_parser=parser)

    args = parser.parse_args()
    trainer(args)
