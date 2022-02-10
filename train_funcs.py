import torch
import random
import numpy as np
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
from load_OF import load_case
from model_def import LitModel
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

def build_dataset(case, zones, ts, rc,
                  num_nodes=-1, 
                  rollout=1,
                  data_fields=['p','U'],
                  knn=False,
                  train_split=0.9):
    ### Read data ###
    p, b, v = load_case(case, data_fields, ts, zones)

    # Sample internal nodes
    if num_nodes != -1:
        n_bounds = sum([len(b[i+1]) for i in range(len(b)-1)])
        idx = torch.randperm(p[0].shape[0])[:num_nodes-n_bounds]
        p[0] = p[0][idx,:]; b[0] = b[0][idx]; v[0] = v[0][:,idx,:]

    pos = torch.cat(p,dim=0)
    b1hot = F.one_hot(torch.cat(b,dim=0))
    fields = torch.cat(v,dim=1)
    features = torch.cat([b1hot.unsqueeze(0).repeat(len(ts),1,1),fields],dim=-1)

    # Generate graph
    if knn:
        print('Generating KNN graph with k=',rc)
        edge_index = knn_graph(pos, k=rc)
    else:
        print('Generating radius graph with r=',rc)
        edge_index = radius_graph(pos, r=rc, max_num_neighbors=32)

    ### Generate dataset/loaders ###
    dataset = [Data(x=features[i,:,:], y=fields[i+1:i+1+rollout,:,:], 
                    dts=torch.diff(ts[i:i+1+rollout]),
                    pos=pos, edge_index=edge_index, rc=rc) for i in range(len(ts)-1)]
    # random.shuffle(dataset)
    train_data = dataset[:int((len(dataset)+1)*train_split)] 
    val_data = dataset[int((len(dataset)+1)*train_split):]
    train_loader = DataLoader(train_data, num_workers=8)
    val_loader = DataLoader(val_data, num_workers=8)

    return train_loader, val_loader

def build_model(in_scalars, in_vectors,
                latent_layers, latent_scalars, latent_vectors,
                out_scalars, out_vectors):
    ### Model definition ###
    irreps_in = f'{in_scalars:g}'+'x0e + '+f'{in_vectors:g}'+'x1o'
    irreps_out = f'{out_scalars:g}'+'x0e + '+f'{out_vectors:g}'+'x1o'
    irreps_latent = f'{latent_scalars:g}'+'x0e + '+f'{latent_vectors:g}'+'x1o'

    return LitModel(irreps_in, irreps_latent, irreps_out, latent_layers)