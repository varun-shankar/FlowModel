import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
import fluidfoam as ff
from load_OF import load_case
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

### Read data ###
case = '../OpenFoam/cylinder2D_base'

dt = 0.05
ts = torch.arange(3,6.001,step=dt)
p, b, v = load_case(case,['p','U'],ts,['internal','cylinder'])

# Sample internal nodes
num_nodes = 5000
idx = torch.randperm(p[0].shape[0])[:num_nodes]
p[0] = p[0][idx,:]; b[0] = b[0][idx]; v[0] = v[0][:,idx,:] 

pos = torch.cat(p,dim=0)
b1hot = F.one_hot(torch.cat(b,dim=0))
fields = torch.cat(v,dim=1)
features = torch.cat([b1hot.unsqueeze(0).repeat(len(ts),1,1),fields],dim=-1)

# Generate graph
rc = .03
edge_index = radius_graph(pos, r=rc)

### Generate dataset/loaders ###
dataset = [Data(x=features[i,:,:], edge_index=edge_index, pos=pos, rc=rc, dt=dt, y=fields[i+1,:,:]) for i in range(len(ts)-1)]
# random.shuffle(dataset)
train_data = dataset[:int((len(dataset)+1)*.9)] 
val_data = dataset[int((len(dataset)+1)*.9):]
train_loader = DataLoader(train_data, num_workers=8)
val_loader = DataLoader(val_data, num_workers=8)

### Model definition ###
from model_def import LitModel
irreps_in = "3x0e + 1o"
irreps_out = "0e + 1o"
irreps_latent = "16x0e + 16x1o"

### Lightning setup ###
wandb_logger = WandbLogger(project="e3nn-opt", log_model=True)
load_checkpoint = False
if load_checkpoint:
    checkpoint_reference = "vshankar/e3nn-opt/model-3nspch19:v0"
    run = wandb.init(project="e3nn-opt")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    model = LitModel.load_from_checkpoint(artifact_dir+"/model.ckpt")
else:
    model = LitModel(irreps_in, irreps_latent, irreps_out)
    wandb_logger.experiment.config.update({"R_c": rc, "Num_nodes": num_nodes})

### Train ###
trainer = pl.Trainer(gpus=1, logger=wandb_logger, max_epochs=500)
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