import torch
from train_funcs import *
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

### Config ###
config = dict(
    num_nodes = 5000,
    rollout = 1,
    rc = .03,
    latent_layers = 4,
    latent_scalars = 32,
    latent_vectors = 8,
    epochs = 400,
)

### Read data ###
case = '../OpenFoam/cylinder2D_base'
zones = ['internal','cylinder','inlet','outlet','top','bottom']
data_fields=['p','U']
ts = torch.arange(3,6.001,step=0.05)
train_loader, val_loader = build_dataset(case, zones, ts, config.rc, 
                                         num_nodes=config.num_nodes, rollout=config.rollout,
                                         data_fields=data_fields, train_split=0.9)


### Lightning setup ###
wandb_logger = WandbLogger(project='e3nn-opt', log_model=True)
load_checkpoint = False
if load_checkpoint:
    checkpoint_reference = 'vshankar/e3nn-opt/model-3nspch19:v0'
    run = wandb.init(project='e3nn-opt')
    artifact = run.use_artifact(checkpoint_reference, type='model')
    artifact_dir = artifact.download()
    model = LitModel.load_from_checkpoint(artifact_dir+'/model.ckpt')
else:
    model = build_model(len(zones)+1, 1,
                    config.latent_layers, config.latent_scalars, config.latent_vectors,
                    1, 1)
    wandb_logger.experiment.config.update(config)

### Train ###
trainer = pl.Trainer(gpus=1, logger=wandb_logger, max_epochs=config.epochs)
wandb_logger.watch(model)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

### Equivariance check ###
from model_def import rot_data
from e3nn import o3

data = dataset[-1]
pred = model(data)
torch.save((data,pred),'data.pt')

rot = o3.rand_matrix()
rot_data, D_in, D_out = rot_data(data, irreps_in, irreps_out, rot)

pred_after = pred @ D_out.T
pred_before = model(rot_data)

print(torch.allclose(pred_before, pred_after, rtol=1e-3, atol=1e-3))
print(torch.max(torch.abs(pred_before-pred_after)))