import torch
from types import SimpleNamespace
from data import OFDataModule
from model_def import LitModel, build_model
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
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
    lr = 0.00001 * 8,
    epochs = 60,
)

wandb_logger = WandbLogger(project='flow-model', log_model='all', job_type='eval', config=config)
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
    test_ts=test_ts, test_rollout=config.test_rollout
)

# model = LitModel.load_from_checkpoint('checkpoints/best-v20.ckpt')
# trainer = pl.Trainer()
# trainer.test(model, datamodule=dm)