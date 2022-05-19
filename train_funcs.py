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

def gen_dataset(case, zones, ts, rc,
                num_nodes=-1, 
                rollout=1,
                data_fields=['p','U'],
                knn=False):
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
        edge_index = knn_graph(pos, k=rc)
    else:
        edge_index = radius_graph(pos, r=rc, max_num_neighbors=32)

    ### Generate dataset ###
    dataset = [Data(x=features[i,:,:], y=fields[i+1:i+1+rollout,:,:], 
                    dts=torch.diff(ts[i:i+1+rollout]),
                    pos=pos, edge_index=edge_index, rc=rc) for i in range(len(ts)-rollout)]

    return dataset

def build_dataloaders(case, zones, ts, rc,
                      num_nodes=-1, 
                      rollout=1,
                      data_fields=['p','U'],
                      knn=False,
                      train_split=0.9, shuffle=False,
                      test_ts=[], test_rollout=1):
    dataset = gen_dataset(case, zones, ts, rc, num_nodes, 
                          rollout, data_fields, knn)
    if shuffle:
        random.shuffle(dataset)
    train_data = dataset[:int((len(dataset)+1)*train_split)] 
    val_data = dataset[int((len(dataset)+1)*train_split):]
    train_loader = DataLoader(train_data, num_workers=8)
    val_loader = DataLoader(val_data, num_workers=8)

    if len(test_ts) > 0:
        test_data = gen_dataset(case, zones, test_ts, rc, num_nodes, 
                                rollout, data_fields, knn)
        test_rollout_data = gen_dataset(case, zones, test_ts, rc, num_nodes, 
                                        test_rollout, data_fields, knn)
        test_loader = DataLoader(test_data, num_workers=8)
        test_rollout_loader = DataLoader(test_rollout_data, num_workers=8)
        return train_loader, val_loader, test_loader, test_rollout_loader
    else:
        return train_loader, val_loader