import torch
from train_funcs import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

### Model params ###
defaults = dict(
    num_nodes = 5000,
    rollout = 1,
    rc = .025,
    latent_layers = 4,
    latent_scalars = 8,
    latent_vectors = 8,
    epochs = 400,
)

wandb.init(config=defaults)
config = wandb.config

case = '../OpenFoam/cylinder2D_base'
zones = ['internal','cylinder','inlet','outlet','top','bottom']
data_fields = ['p','U']
ts = torch.arange(3,6.001,step=0.05)
train_loader, val_loader = build_dataset(case, zones, ts, config.rc, 
                                         num_nodes=config.num_nodes, rollout=config.rollout,
                                         data_fields=data_fields, train_split=0.9)
model = build_model(len(zones)+1, 1,
                    config.latent_layers, config.latent_scalars, config.latent_vectors,
                    1, 1)

wandb_logger = WandbLogger(project='flow-model', log_model=True)
trainer = pl.Trainer(gpus=1, logger=wandb_logger, max_epochs=config.epochs)
wandb_logger.watch(model)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)