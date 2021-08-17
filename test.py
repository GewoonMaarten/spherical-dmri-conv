from pathlib import Path

import mlflow
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from autoencoder.concrete_autoencoder import ConcreteAutoencoder
from autoencoder.dataset import MRIDataModule, MRISelectorSubjDataset

ROOT_PATH = Path().cwd()
root_dir = Path(ROOT_PATH, "data")

mlflow.pytorch.autolog()

stopping_callback = EarlyStopping(
    monitor="mean_max",
    stopping_threshold=0.998,
    patience=float("inf"),
    verbose=True,
    mode="max",
)
model_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./models",
    filename="mudi_checkpoint-epoch{epoch:02d}-val_loss{val_loss:.2f}",
    auto_insert_metric_name=False,
)

logger = TensorBoardLogger("tb_logs", name="my_model")

model = ConcreteAutoencoder(1344, 500, 1)
print(model)

dm = MRIDataModule(
    root_dir,
    "data_fake.hdf5",
    "header_fake.csv",
    subject_list_train=np.array([11, 12, 13, 14]),
    subject_list_val=np.array([15]),
)
dm.setup(stage="fit")

trainer = pl.Trainer(
    auto_lr_find=True,
    gpus=-1,
    max_epochs=10,
    logger=logger,
    callbacks=[model_callback, stopping_callback],
)
# Auto log all MLflow entities
mlflow.pytorch.autolog()

# Train the model
with mlflow.start_run() as run:
    trainer.fit(model, dm)
