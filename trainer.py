from argparse import ArgumentParser, Namespace

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

import utils.logger as logger
from autoencoder.concrete_autoencoder import ConcreteAutoencoder
from autoencoder.dataset import MRIDataModule


def trainer(args: Namespace) -> None:
    """Take command line arguments to train a model.

    Args:
        args (Namespace): arguments from argparse
    """
    mlflow.pytorch.autolog()
    mlflow.log_params(vars(args))

    logger.init_logger(logger.LOGGER_NAME, args.verbose)
    is_verbose = args.verbose < 30

    early_stopping = EarlyStopping(
        monitor="mean_max",
        mode="max",
        patience=float("inf"),
        stopping_threshold=0.998,
        verbose=is_verbose,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=is_verbose,
    )
    lr_logger = LearningRateMonitor()

    model = ConcreteAutoencoder(
        args.input_output_size,
        args.latent_size,
        args.decoder_hidden_layers,
        args.learning_rate,
    )
    dm = MRIDataModule(
        data_file=args.data_file,
        header_file=args.header_file,
        batch_size=args.batch_size,
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[lr_logger, early_stopping, checkpoint_callback],
        checkpoint_callback=True,
    )

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