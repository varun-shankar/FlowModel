import torch
from torch_scatter import scatter
from e3nn import o3, nn
from torch.nn import ReLU, Tanh, SiLU, Sigmoid, ELU, Linear
from .utils import *
from torch_geometric.nn import GCNConv, EdgeConv
from torchdyn.core import NeuralODE

## Message Passing Layers ##
class Eq_NLMP(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=torch.tanh,
                       irreps_sh=o3.Irreps.spherical_harmonics(lmax=2), num_basis=10, fch=16):
        super(Eq_NLMP, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify()
        self.gate = nn.Gate(irreps_scalars, [act for _, ir in irreps_scalars],
                            irreps_gates, [act for _, ir in irreps_gates], irreps_gated)
        self.irreps_sh = o3.Irreps(irreps_sh)
        # self.tp = o3.FullyConnectedTensorProduct(2*self.irreps_input, self.irreps_sh, self.gate.irreps_in, shared_weights=False)
        self.tp = o3.FullyConnectedTensorProduct(2*self.irreps_input, self.irreps_sh, self.irreps_output, shared_weights=False)
        self.tp2 = o3.FullyConnectedTensorProduct(self.irreps_output, self.irreps_sh, self.gate.irreps_in, shared_weights=False)
        self.num_basis = num_basis
        self.fc = nn.FullyConnectedNet([self.num_basis, fch, self.tp.weight_numel], torch.relu)
        self.fc2 = nn.FullyConnectedNet([self.num_basis, fch, self.tp2.weight_numel], torch.relu)

    def forward(self, x, data):
        
        edge_src, edge_dst = data.edge_index
        sh = o3.spherical_harmonics(self.irreps_sh, data.edge_vec, normalize=True, normalization='component')
        # edge_ftr = self.gate(self.tp(torch.cat([x[edge_src],x[edge_dst]],dim=1), sh, self.fc(data.emb))) * data.norm.view(-1, 1)
        edge_ftr = self.tp(torch.cat([x[edge_src],x[edge_dst]],dim=1), sh, self.fc(data.emb))
        edge_ftr = self.gate(self.tp2(edge_ftr, sh, self.fc2(data.emb))) * data.norm.view(-1, 1)
        
        return scatter(edge_ftr, edge_dst, dim=0, dim_size=data.num_nodes)

class Eq_NLMP2(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=ReLU(),
                       irreps_sh=o3.Irreps.spherical_harmonics(lmax=2),
                       edge_basis=10, fch=16, **kwargs):
        super(Eq_NLMP2, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify()
        
        self.gate = nn.Gate(irreps_scalars, [act for _, ir in irreps_scalars],
                            irreps_gates, [act for _, ir in irreps_gates], irreps_gated)
        self.irreps_sh = o3.Irreps(irreps_sh)
        self.tp = o3.FullyConnectedTensorProduct(3*self.irreps_input, self.irreps_sh, self.gate.irreps_in, shared_weights=False)
        self.tp2 = o3.FullyConnectedTensorProduct(self.irreps_output, self.irreps_sh, self.irreps_output, shared_weights=False)
        self.fc = nn.FullyConnectedNet([edge_basis, fch, self.tp.weight_numel], torch.relu)
        self.fc2 = nn.FullyConnectedNet([edge_basis, fch, self.tp2.weight_numel], torch.relu)

        self.lin = torch.nn.Sequential(
            o3GatedLinear(self.irreps_input+self.irreps_output, self.irreps_output),
            o3.Linear(self.irreps_output, self.irreps_output)
        )

    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        sh = o3.spherical_harmonics(self.irreps_sh, data.edge_vec, normalize=True, normalization='component')
        tmp = self.gate(self.tp(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1), sh, self.fc(data.emb)))
        data.he += self.tp2(tmp, sh, self.fc2(data.emb))
        node_ftr = scatter(data.he*data.norm.view(-1, 1), edge_dst, dim=0, dim_size=data.num_nodes)
        data.hn += self.lin(torch.cat([data.hn,node_ftr],dim=1))
        
        return data

class Eq_NLMP3(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output,
                       irreps_sh=o3.Irreps.spherical_harmonics(lmax=2),
                       edge_basis=16, fch=16, **kwargs):
        super(Eq_NLMP3, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        hfac = 4

        self.irreps_sh = o3.Irreps(irreps_sh)
        self.edge_val = torch.nn.Sequential(
            o3GatedLinear(3*self.irreps_input, (hfac*self.irreps_output).sort().irreps.simplify(), **kwargs),
            o3.Linear((hfac*self.irreps_output).sort().irreps.simplify(), self.irreps_sh)
        )

        self.tp = o3.FullyConnectedTensorProduct(self.irreps_sh, self.irreps_sh, self.irreps_output, shared_weights=False)
        self.fc = nn.FullyConnectedNet([edge_basis, fch, self.tp.weight_numel], torch.relu)

        self.edge_upd = torch.nn.Sequential(
            o3GatedLinear(self.tp.irreps_out+2*self.irreps_input, (hfac*self.irreps_output).sort().irreps.simplify(), **kwargs),
            o3.Linear((hfac*self.irreps_output).sort().irreps.simplify(), self.irreps_output)
        )
        self.node_lin = torch.nn.Sequential(
            o3GatedLinear(self.irreps_input+self.irreps_output, (hfac*self.irreps_output).sort().irreps.simplify(), **kwargs),
            o3.Linear((hfac*self.irreps_output).sort().irreps.simplify(), self.irreps_output)
        )

    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        v = self.edge_val(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        
        data.he += self.edge_upd(torch.cat([self.tp(v, data.fe, self.fc(data.fes)),data.hn[edge_src],data.hn[edge_dst]],dim=1))
        node_ftr = scatter(data.he*data.norm.view(-1, 1), edge_dst, dim=0, dim_size=data.num_nodes)
        data.hn += self.node_lin(torch.cat([data.hn,node_ftr],dim=1))
        
        return data

class nEq_NLMP3(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output,
                       irreps_sh=o3.Irreps.spherical_harmonics(lmax=2),
                       edge_basis=16, fch=16, act=ReLU(), **kwargs):
        super(nEq_NLMP3, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input).dim
        self.irreps_output = o3.Irreps(irreps_output).dim
        hfac = 4

        self.irreps_sh = o3.Irreps(irreps_sh).dim
        self.edge_val = torch.nn.Sequential(
            Linear(3*self.irreps_input, (hfac*self.irreps_output)), act,
            Linear((hfac*self.irreps_output), self.irreps_sh)
        )

        self.tp = Linear(2*self.irreps_sh+edge_basis, self.irreps_output)

        self.edge_upd = torch.nn.Sequential(
            Linear(self.irreps_output+2*self.irreps_input, (hfac*self.irreps_output)), act,
            Linear((hfac*self.irreps_output), self.irreps_output)
        )
        self.node_lin = torch.nn.Sequential(
            Linear(self.irreps_input+self.irreps_output, (hfac*self.irreps_output)), act,
            Linear((hfac*self.irreps_output), self.irreps_output)
        )

    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        v = self.edge_val(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
        
        data.he += self.edge_upd(torch.cat([self.tp(torch.cat([v, data.fe, data.fes],dim=1)),data.hn[edge_src],data.hn[edge_dst]],dim=1))
        node_ftr = scatter(data.he*data.norm.view(-1, 1), edge_dst, dim=0, dim_size=data.num_nodes)
        data.hn += self.node_lin(torch.cat([data.hn,node_ftr],dim=1))
        
        return data

class nEq_NLMP2(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=ReLU(),
                       edge_basis=10, fch=256, **kwargs):
        super(nEq_NLMP2, self).__init__()

        self.irreps_input = irreps_input.dim
        self.irreps_output = irreps_output.dim

        self.lin1 = torch.nn.Sequential(
            Linear(3*self.irreps_input+edge_basis, fch), act,
            Linear(fch, self.irreps_output)
        )
        self.lin2 = torch.nn.Sequential(
            Linear(self.irreps_input+self.irreps_output, fch), act,
            Linear(fch, self.irreps_output)
        )

    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        data.he += self.lin1(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst],data.emb],dim=1))
        node_ftr = scatter(data.he*data.norm.view(-1, 1), edge_dst, dim=0, dim_size=data.num_nodes)
        data.hn += self.lin2(torch.cat([data.hn,node_ftr],dim=1))
        
        return data

class nEq_NLMP_iso(torch.nn.Module):
    def __init__(self, features_input, features_output, act=torch.tanh,
                       num_basis=10, fch=16):
        super(nEq_NLMP_iso, self).__init__()

        self.features_input = features_input
        self.features_output = features_output
        self.act = act
        self.num_basis = num_basis
        self.fc = torch.nn.Sequential(Linear(self.num_basis, fch), ReLU(), 
            Linear(fch, 2*features_input*features_output))

    def forward(self, x, data):
        
        edge_src, edge_dst = data.edge_index
        weights = self.fc(data.emb).reshape(-1, self.features_output, 2*self.features_input)
        edge_ftr = self.act(weights @ torch.cat([x[edge_src],x[edge_dst]],dim=1).unsqueeze(-1)).squeeze() * data.norm.view(-1, 1)

        return scatter(edge_ftr, edge_dst, dim=0, dim_size=data.num_nodes)

class nEq_NLMP_aniso(torch.nn.Module):
    def __init__(self, features_input, features_output, act=torch.tanh,
                       fch=16):
        super(nEq_NLMP_aniso, self).__init__()

        self.features_input = features_input
        self.features_output = features_output
        self.act = act
        self.fc = torch.nn.Sequential(Linear(3, fch), ReLU(), 
            Linear(fch, 2*features_input*features_output))

    def forward(self, x, data):
        
        edge_src, edge_dst = data.edge_index
        weights = self.fc(data.edge_vec).reshape(-1, self.features_output, 2*self.features_input)
        edge_ftr = self.act(weights @ torch.cat([x[edge_src],x[edge_dst]],dim=1).unsqueeze(-1)).squeeze() * data.norm.view(-1, 1)

        return scatter(edge_ftr, edge_dst, dim=0, dim_size=data.num_nodes)

class GCN(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=torch.tanh):
        super(GCN, self).__init__()

        self.irreps_input = irreps_input.dim
        self.irreps_output = irreps_output.dim
        self.act = act
        self.f = GCNConv(self.irreps_input, self.irreps_output)

    def forward(self, data):
        data.hn = self.act(self.f(data.hn, data.edge_index))+0*data.he.mean()
        return data 

class Edge(torch.nn.Module):
    def __init__(self, features_input, features_output, act=torch.tanh,
                       fch=64):
        super(Edge, self).__init__()

        self.features_input = features_input
        self.features_output = features_output
        self.act = act
        self.fc = torch.nn.Sequential(Linear(2*features_input, fch), ReLU(), 
            Linear(fch, features_output))
        self.f = EdgeConv(self.fc, aggr='add')

    def forward(self, x, data):
        
        return self.act(self.f(x, data.edge_index))