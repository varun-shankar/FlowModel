import torch
from typing import Optional
import random
import numpy as np
from torch_cluster import knn_graph, radius_graph
from load_OF import load_case
import torch.nn.functional as F
from torch_geometric.data import Data as pygData
from torch_geometric.utils import degree
from e3nn.math import soft_one_hot_linspace
from e3nn import o3
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

class Data(pygData):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'y':
            return 1
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

    def rotate(self, rot):
        irreps_in = o3.Irreps(self.irreps_io[0][0]).simplify()
        irreps_out = o3.Irreps(self.irreps_io[0][1]).simplify()
        D_in = irreps_in.D_from_matrix(rot).type_as(self.x)
        D_out = irreps_out.D_from_matrix(rot).type_as(self.x)
        self.x = self.x @ D_in.T
        self.pos = self.pos @ rot.T
        self.y = self.y @ D_out.T
        return self

    def embed(self, num_basis=10):
        edge_src, edge_dst = self.edge_index
        deg = degree(edge_dst, self.num_nodes).type_as(self.x)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.norm = deg_inv_sqrt[edge_src] * deg_inv_sqrt[edge_dst]

        self.edge_vec = self.pos[edge_dst] - self.pos[edge_src]
        rc = float(self.rc.max()) if torch.is_tensor(self.rc) else self.rc
        self.emb = soft_one_hot_linspace(self.edge_vec.norm(dim=1), 0.0, rc, num_basis, 
            basis='smooth_finite', cutoff=True).mul(num_basis**0.5)

class OFDataModule(pl.LightningDataModule):
    def __init__(self, case, zones, ts, rc,
                       num_nodes=-1, 
                       rollout=1,
                       data_fields=['p','U'], data_irreps='0e+1o',
                       knn=False,
                       train_split=0.9, random_split=False,
                       test_ts=[], test_rollout=0,
                       shuffle=False, batch_size=1):
        super().__init__()
        self.case = case
        self.zones = zones
        self.ts = ts
        self.rc = rc
        self.num_nodes = num_nodes
        self.rollout = rollout
        self.data_fields = data_fields
        self.data_irreps = data_irreps
        self.knn = knn
        self.train_split = train_split
        self.random_split = random_split
        self.test_ts = test_ts
        self.test_rollout = test_rollout
        self.shuffle = shuffle
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        all_ts = torch.cat([self.ts,self.test_ts],dim=0)

        ### Read data ###
        p, b, v = load_case(self.case, self.data_fields, all_ts, self.zones)

        # Sample internal nodes
        if self.num_nodes != -1:
            n_bounds = sum([len(b[i+1]) for i in range(len(b)-1)])
            idx = torch.randperm(p[0].shape[0])[:self.num_nodes-n_bounds]
            p[0] = p[0][idx,:]; b[0] = b[0][idx]; v[0] = v[0][:,idx,:]

        pos = torch.cat(p,dim=0)
        b1hot = F.one_hot(torch.cat(b,dim=0))
        fields = torch.cat(v,dim=1)
        features = torch.cat([b1hot.unsqueeze(0).repeat(len(all_ts),1,1),fields],dim=-1)
        irreps_io = [f'{len(self.zones):g}'+'x0e+'+self.data_irreps,self.data_irreps]

        # Generate graph
        if self.knn:
            edge_index = knn_graph(pos, k=self.rc)
        else:
            edge_index = radius_graph(pos, r=self.rc, max_num_neighbors=32)
        print('Avg neighbors = ', edge_index.shape[1]/pos.shape[0])

        ### Generate dataset ###
        dataset = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.rollout,:,:], irreps_io=irreps_io,
                        dts=torch.diff(all_ts[i:i+1+self.rollout]),
                        pos=pos, edge_index=edge_index, rc=self.rc) for i in range(len(self.ts)-self.rollout)]
        if self.random_split:
            random.shuffle(dataset)
        self.train_data = dataset[:int((len(dataset)+1)*self.train_split)] 
        self.val_data = dataset[int((len(dataset)+1)*self.train_split):]

        # Test
        testset = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.rollout,:,:], irreps_io=irreps_io,
                        dts=torch.diff(all_ts[i:i+1+self.rollout]),
                        pos=pos, edge_index=edge_index, rc=self.rc) for i in range(
                            len(self.ts),len(self.ts)+len(self.test_ts)-self.rollout)]
        testset_rollout = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.test_rollout,:,:], irreps_io=irreps_io,
                        dts=torch.diff(all_ts[i:i+1+self.test_rollout]),
                        pos=pos, edge_index=edge_index, rc=self.rc) for i in range(
                            len(self.ts),len(self.ts)+len(self.test_ts)-self.test_rollout)]
        self.test_data = testset
        self.test_data_rollout = testset_rollout

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return [DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8),
                DataLoader(self.test_data_rollout, num_workers=8)]