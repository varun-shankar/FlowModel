import torch
import random
import numpy as np
import copy
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh
from torch_geometric.utils import degree
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

# class MP(torch.nn.Module):
#     def __init__(self, irreps_input, irreps_output, 
#                        irreps_sh=o3.Irreps.spherical_harmonics(lmax=2), num_basis=10, fch=16):
#         super(MP, self).__init__()

#         self.irreps_input = o3.Irreps(irreps_input)
#         self.irreps_output = o3.Irreps(irreps_output)
#         self.irreps_sh = o3.Irreps(irreps_sh)
#         self.tp = o3.FullyConnectedTensorProduct(2*self.irreps_input, self.irreps_sh, self.irreps_output, shared_weights=False)
#         self.num_basis = num_basis
#         self.fc = nn.FullyConnectedNet([self.num_basis, fch, self.tp.weight_numel], torch.relu)

#     def forward(self, x, pos, edge_index, rc):

#         num_nodes = pos.shape[0]
#         edge_src, edge_dst = edge_index
#         num_neighbors = len(edge_src) / num_nodes
#         edge_vec = pos[edge_dst] - pos[edge_src]
#         sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
#         rc = rc[0].item() if torch.is_tensor(rc) else rc
#         emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, rc, self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)
#         return scatter(self.tp(torch.cat([x[edge_src],x[edge_dst]],dim=1), sh, self.fc(emb)), edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)

# class dudt(torch.nn.Module):
#     def __init__(self, num_latent):
#         super(dudt, self).__init__()

#         self.conv1 = MP([(num_latent,(0,1)),(num_latent,(1,-1))], 
#                         [(2*num_latent,(0,1)),(num_latent,(1,-1))])
#         self.g1 = nn.Gate([(num_latent,(0,1))], [torch.tanh], 
#                           [(num_latent,(0,1))], [torch.tanh], [(num_latent,(1,-1))])
#         self.conv2 = MP([(num_latent,(0,1)),(num_latent,(1,-1))], 
#                         [(2*num_latent,(0,1)),(num_latent,(1,-1))])
#         self.g2 = nn.Gate([(num_latent,(0,1))], [torch.tanh], 
#                           [(num_latent,(0,1))], [torch.tanh], [(num_latent,(1,-1))])
#         self.conv3 = MP([(num_latent,(0,1)),(num_latent,(1,-1))], 
#                         [(2*num_latent,(0,1)),(num_latent,(1,-1))])
#         self.g3 = nn.Gate([(num_latent,(0,1))], [torch.tanh], 
#                           [(num_latent,(0,1))], [torch.tanh], [(num_latent,(1,-1))])
#         self.conv4 = MP([(num_latent,(0,1)),(num_latent,(1,-1))], 
#                         [(num_latent,(0,1)),(num_latent,(1,-1))])

#     def forward(self, x, data):

#         h = x
#         h = self.conv1(h, data.pos, data.edge_index, data.rc)
#         h = self.g1(h)
#         h = self.conv2(h, data.pos, data.edge_index, data.rc)
#         h = self.g2(h)
#         h = self.conv3(h, data.pos, data.edge_index, data.rc)
#         h = self.g3(h)
#         h = self.conv4(h, data.pos, data.edge_index, data.rc)
        
#         return h


class NLMP(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=torch.tanh,
                       irreps_sh=o3.Irreps.spherical_harmonics(lmax=2), num_basis=10, fch=16):
        super(NLMP, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify()
        self.gate = nn.Gate(irreps_scalars, [act for _, ir in irreps_scalars],
                            irreps_gates, [act for _, ir in irreps_gates], irreps_gated)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.tp = o3.FullyConnectedTensorProduct(2*self.irreps_input, self.irreps_sh, self.gate.irreps_in, shared_weights=False)
        self.num_basis = num_basis
        self.fc = nn.FullyConnectedNet([self.num_basis, fch, self.tp.weight_numel], torch.relu)

    def forward(self, x, pos, edge_index, rc):

        num_nodes = pos.shape[0]
        edge_src, edge_dst = edge_index
        num_neighbors = len(edge_src) / num_nodes
        edge_vec = pos[edge_dst] - pos[edge_src]
        sh = o3.spherical_harmonics(self.irreps_sh, edge_vec, normalize=True, normalization='component')
        rc = rc[0].item() if torch.is_tensor(rc) else rc
        emb = soft_one_hot_linspace(edge_vec.norm(dim=1), 0.0, rc, self.num_basis, basis='smooth_finite', cutoff=True).mul(self.num_basis**0.5)
        edge_ftr = self.gate(self.tp(torch.cat([x[edge_src],x[edge_dst]],dim=1), sh, self.fc(emb)))
        return scatter(edge_ftr, edge_dst, dim=0, dim_size=num_nodes).div(num_neighbors**0.5)


class dudt(torch.nn.Module):
    def __init__(self, irreps_latent):
        super(dudt, self).__init__()

        self.conv1 = NLMP(irreps_latent, irreps_latent)
        self.conv2 = NLMP(irreps_latent, irreps_latent)
        self.conv3 = NLMP(irreps_latent, irreps_latent)
        self.conv4 = NLMP(irreps_latent, irreps_latent)

    def forward(self, x, data):

        h = x
        h = self.conv1(h, data.pos, data.edge_index, data.rc)
        h = self.conv2(h, data.pos, data.edge_index, data.rc)
        h = self.conv3(h, data.pos, data.edge_index, data.rc)
        h = self.conv4(h, data.pos, data.edge_index, data.rc)
        
        return h


class LitModel(pl.LightningModule):
    def __init__(self, irreps_in, irreps_latent, irreps_out):
        super().__init__()
        self.save_hyperparameters()
        
        self.enc = o3.Linear(irreps_in, irreps_latent)
        self.f = dudt(irreps_latent)
        self.dec = o3.Linear(irreps_latent, irreps_out)

    def forward(self, data):

        h = data.x
        h = self.enc(h)
        hs = []
        for i in range(data.y.shape[0]):
            h = h + self.f(h, data)*data.dt
            hs.append(h)
        hs = torch.stack(hs)

        hs = self.dec(hs)

        return hs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = F.mse_loss(y_hat, data.y)
        self.log('train_loss', loss, batch_size=data.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = F.mse_loss(y_hat, data.y)
        self.log('val_loss', loss, batch_size=data.num_graphs)
        return loss


def rot_data(data, irreps_in, irreps_out, rot):
    D_in = o3.Irreps(irreps_in).D_from_matrix(rot)
    D_out = o3.Irreps(irreps_out).D_from_matrix(rot)
    rot_data = copy.deepcopy(data)
    rot_data.x = data.x @ D_in.T
    rot_data.pos = data.pos @ rot.T
    rot_data.y = data.y @ D_out.T
    return rot_data, D_in, D_out