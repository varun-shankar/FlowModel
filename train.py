import torch
from types import SimpleNamespace
from data import OFDataModule
from model_def import LitModel, build_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
import wandb
import os
import glob
os.environ['WANDB_START_METHOD'] = 'thread'
pl.seed_everything(42)

### Config ###
config = dict(
    num_nodes = 5000,
    dt = .01,
    rollout = 1,
    test_rollout = 10,
    rc = .04,
    latent_layers = 4,
    latent_scalars = 16,
    latent_vectors = 16,
    batch_size = 1,
    lr = 0.002 * 8,
    epochs = 200,
)
load_checkpoint = False
wandb_logger = WandbLogger(project='flow-model', log_model='all', job_type='train', config=config)
config = SimpleNamespace(**config)

### Read data ###
case = '../OpenFoam/cylinder2D_base'
zones = ['internal','cylinder','inlet','outlet','top','bottom']
data_fields = ['p','U']
ts = torch.arange(3,6.001,step=config.dt)
test_ts = torch.arange(6,6.101,step=config.dt)
dm = OFDataModule(
    case, zones, ts, config.rc, 
    num_nodes=config.num_nodes, rollout=config.rollout,
    data_fields=data_fields, train_split=0.9,
    test_ts=test_ts, test_rollout=config.test_rollout,
    shuffle=True, batch_size=config.batch_size
)


### Lightning setup ###
if load_checkpoint:
    latest = max(glob.glob('checkpoints/*'), key=os.path.getctime)
    print('Loading '+latest)
    model = LitModel.load_from_checkpoint(latest)
else:
    model = build_model(
        len(zones)+1, 1,
        config.latent_layers, config.latent_scalars, config.latent_vectors,
        1, 1,
        config.lr
    )

### Train ###
checkpoint_callback = ModelCheckpoint(monitor='val_loss', dirpath='checkpoints/', filename='best')
trainer = pl.Trainer(
    gpus=-1, strategy=DDPPlugin(find_unused_parameters=False), precision=16, 
    logger=wandb_logger, callbacks=[checkpoint_callback],
    max_epochs=config.epochs
)
wandb_logger.watch(model)
trainer.fit(model, dm)

trainer.test(model, datamodule=dm)