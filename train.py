import os, glob, sys, yaml, argparse
from types import SimpleNamespace
import flowmodel.data.modules as datamodules
from flowmodel.nn.model import LitModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
pl.seed_everything(42)

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config',
                        help='Configuration file', type=str)
    args = parser.parse_args()
    with open(args.config,'r') as fl:
        config = yaml.load(fl,yaml.FullLoader)
    return config

### Config ###
config = parse_command_line()

## Read data ###
dm = getattr(datamodules,config.get('data_module'))(**config)

### Lightning setup ###
if config.get('job_type') == 'train':
    model = LitModel(
        dm.irreps_io[0], dm.irreps_io[1], dm.loss_fn,
        **config
    )
    ckpt = None
    run_id = wandb.util.generate_id()
elif config.get('job_type') == 'retrain':
    if config.get('load_id', None) is None:
        ckpt = max(glob.glob('checkpoints/*'), key=os.path.getctime)
    else:
        ckpt = max(glob.glob('checkpoints/run-'+config.get('load_id')+'*'), key=os.path.getctime)
    print('Loading '+ckpt)
    model = LitModel.load_from_checkpoint(ckpt, loss_fn=dm.loss_fn, **config)
    ckpt = None
    run_id = wandb.util.generate_id()
elif config.get('job_type') == 'resume' or 'eval':
    if config.get('load_id', None) is None:
        ckpt = max(glob.glob('checkpoints/*'), key=os.path.getctime)
    else:
        ckpt = max(glob.glob('checkpoints/run-'+config.get('load_id')+'*'), key=os.path.getctime)
    print('Loading '+ckpt)
    model = LitModel.load_from_checkpoint(ckpt, loss_fn=dm.loss_fn)
    import re
    run_id = re.search('run-(.*)-best', ckpt).group(1)
else:
    print('Unknown job type')

### Train ###
wandb_logger = WandbLogger(project=config.get('project'), log_model='all', 
    job_type=config.get('job_type'), id=run_id, settings=wandb.Settings(start_method="fork"), 
    config=SimpleNamespace(**config))
if config.get('job_type') != 'eval':
    wandb_logger.watch(model)
    checkpoint_callback = ModelCheckpoint(monitor=config.get('monitor','val_loss'), 
        dirpath='checkpoints/', filename='run-'+run_id+'-best')
    lr_monitor = LearningRateMonitor()
    strategy = 'ddp_find_unused_parameters_false' if config.get('gpus') != 1 else None
    trainer = pl.Trainer(
        gpus=config.get('gpus'), strategy=strategy, precision=16,
        logger=wandb_logger, callbacks=[checkpoint_callback,lr_monitor],
        max_epochs=config.get('epochs'), log_every_n_steps=10,
        resume_from_checkpoint=ckpt#, accumulate_grad_batches=2
    )
    trainer.fit(model, dm)
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        trainer = pl.Trainer(gpus=1, logger=wandb_logger, limit_test_batches=5)
        trainer.test(model, datamodule=dm)
else:
    dm.setup('fit')
    trainer = pl.Trainer(gpus=config.get('gpus'), logger=wandb_logger, limit_test_batches=5)
    trainer.test(model, datamodule=dm)