import torch
import os, glob, sys, argparse, yaml
from types import SimpleNamespace
sys.path.append('/home/opc/data/ml-cfd/FlowModel')
import flowmodel.data.modules as datamodules
from model_def import LitModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
import wandb
os.environ['WANDB_START_METHOD'] = 'thread'
pl.seed_everything(42)

### Config ###
with open('config.yaml','r') as fl:
    defaults = yaml.load(fl,yaml.FullLoader)
wandb.init(config=SimpleNamespace(**defaults))
config = wandb.config

### Read data ###
dm = getattr(datamodules,config.get('data_module'))(**config)

model = LitModel(
    dm.irreps_io[0], dm.irreps_io[1],
    **config
)
wandb_logger = WandbLogger(project=config.get('project'), log_model=True)
lr_monitor = LearningRateMonitor()
strategy = DDPPlugin(find_unused_parameters=False) if config.get('gpus') != 1 else None
trainer = pl.Trainer(
    gpus=config.get('gpus'), strategy=strategy, precision=16,
    logger=wandb_logger, callbacks=[lr_monitor],
    max_epochs=config.get('epochs'), log_every_n_steps=10
)
wandb_logger.watch(model)
trainer.fit(model, dm)
trainer.test(model, datamodule=dm)