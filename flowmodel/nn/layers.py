import torch
from torch_scatter import scatter
from e3nn import o3, nn
from torch.nn import ReLU, Tanh, Linear
import math
from torch_geometric.nn import GCNConv, EdgeConv

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
    def __init__(self, irreps_input, irreps_output, act=torch.tanh,
                       irreps_sh=o3.Irreps.spherical_harmonics(lmax=2),
                       edge_basis=10, fch=16):
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
        self.edge_basis = edge_basis
        self.fc = nn.FullyConnectedNet([self.edge_basis, fch, self.tp.weight_numel], torch.relu)
        self.fc2 = nn.FullyConnectedNet([self.edge_basis, fch, self.tp2.weight_numel], torch.relu)

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


class nEq_NLMP2(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=torch.tanh,
                       edge_basis=10, fch=16):
        super(nEq_NLMP2, self).__init__()

        self.irreps_input = irreps_input
        self.irreps_output = irreps_output

        self.lin1 = torch.nn.Sequential(
            Linear(3*self.irreps_input, self.irreps_output), Tanh(),
            Linear(self.irreps_output, self.irreps_output)
        )
        self.lin2 = torch.nn.Sequential(
            Linear(self.irreps_input+self.irreps_output, self.irreps_output), Tanh(),
            Linear(self.irreps_output, self.irreps_output)
        )



    def forward(self, data):
        
        edge_src, edge_dst = data.edge_index
        data.he += self.lin1(torch.cat([data.he,data.hn[edge_src],data.hn[edge_dst]],dim=1))
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
    def __init__(self, features_input, features_output, act=torch.tanh):
        super(GCN, self).__init__()

        self.features_input = features_input
        self.features_output = features_output
        self.act = act
        self.f = GCNConv(features_input, features_output)

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


class dudt(torch.nn.Module):
    def __init__(self, layer_type, irreps_latent, num_layers, **kwargs):
        super(dudt, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(layer_type(irreps_latent, irreps_latent, **kwargs))

    # def forward(self, x, data):

    #     h = x
    #     for i in range(len(self.layers)):
    #         h = self.layers[i](h, data)

    #     return h
    def forward(self, data):

        for i in range(len(self.layers)):
            self.layers[i](data)

        return data

class Encoder(torch.nn.Module):
    def __init__(self, irreps_in, irreps_latent, model_type='equivariant'):
        super(Encoder, self).__init__()

        if model_type=='equivariant' or model_type=='reservoir':
            self.node_enc = torch.nn.Sequential(
                o3GatedLinear(irreps_in, irreps_latent),
                o3.Linear(irreps_latent, irreps_latent)#, LayerNorm()
            )
            self.edge_enc = torch.nn.Sequential(
                o3GatedLinear('0e+1o', irreps_latent),
                o3.Linear(irreps_latent, irreps_latent)#, LayerNorm()
            )
        else:
            self.node_enc = torch.nn.Sequential(
                Linear(irreps_in, irreps_latent), Tanh(),
                Linear(irreps_latent, irreps_latent)
            )
            self.edge_enc = torch.nn.Sequential(
                Linear(4, irreps_latent), Tanh(),
                Linear(irreps_latent, irreps_latent)
            )

    def forward(self, data):
        data.embed()
        data.hn = torch.cat([data.emb_node,data.hn],dim=-1)
        data.he = torch.cat([data.edge_vec.norm(dim=1,keepdim=True),data.edge_vec],dim=-1)
        data.hn = self.node_enc(data.hn)
        data.he = self.edge_enc(data.he)
        return data

class Decoder(torch.nn.Module):
    def __init__(self, irreps_latent, irreps_out, model_type='equivariant'):
        super(Decoder, self).__init__()

        if model_type=='equivariant' or model_type=='reservoir':
            self.node_dec = torch.nn.Sequential(
                o3GatedLinear(irreps_latent, irreps_latent),
                o3.Linear(irreps_latent, irreps_out)
            )
        else:
            self.node_dec = torch.nn.Sequential(
                Linear(irreps_latent, irreps_latent), Tanh(),
                Linear(irreps_latent, irreps_out)
            )

    def forward(self, data):
        data.hn = self.node_dec(data.hn)
        return data

class o3GatedLinear(torch.nn.Module):
    def __init__(self, irreps_input, irreps_output, act=torch.tanh):
        super(o3GatedLinear, self).__init__()

        self.irreps_input = o3.Irreps(irreps_input)
        self.irreps_output = o3.Irreps(irreps_output)
        irreps_scalars = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_output if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, '0e') for mul, _ in irreps_gated]).simplify()
        self.gate = nn.Gate(irreps_scalars, [act for _, ir in irreps_scalars],
                            irreps_gates, [act for _, ir in irreps_gates], irreps_gated)
        self.lin = o3.Linear(self.irreps_input, self.gate.irreps_in)

    def forward(self, x):

        return self.gate(self.lin(x))

class LayerNorm(torch.nn.Module):
    def __init__(self):
        super(LayerNorm, self).__init__()
    
    def forward(self, x):
        x = x - x.mean(0, keepdim=True)
        x = x * (x.pow(2).mean() + 1e-12).pow(-0.5)
        return x

def build_model(model_type, irreps_in, irreps_out,
                latent_layers, latent_scalars, latent_vectors, latent_tensors=0, node_basis=1, **kwargs):
    if model_type == 'equivariant':
        # irreps_in = f'{in_scalars:g}'+'x0e + '+f'{in_vectors:g}'+'x1o'
        # irreps_out = f'{out_scalars:g}'+'x0e + '+f'{out_vectors:g}'+'x1o'
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_latent = f'{latent_scalars:g}'+'x0e + '+ \
            f'{latent_vectors:g}'+'x1o + '+ \
            f'{latent_tensors:g}'+'x2e'
        enc = Encoder(irreps_in, irreps_latent, model_type)
        f = dudt(Eq_NLMP2, irreps_latent, latent_layers)
        dec = Decoder(irreps_latent, irreps_out, model_type)
    elif model_type == 'reservoir':
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_latent = f'{latent_scalars:g}'+'x0e + '+ \
            f'{latent_vectors:g}'+'x1o + '+ \
            f'{latent_tensors:g}'+'x2e'
        enc = Encoder(irreps_in, irreps_latent, model_type)
        f = dudt(Eq_NLMP2, irreps_latent, latent_layers)
        dec = Decoder(irreps_latent, irreps_out, model_type)
        for p in enc.parameters():
            p.requires_grad = False
        for i in range(len(f.layers)-1):
            for p in f.layers[i].parameters():
                p.requires_grad = False
    elif model_type == 'non-equivariant':
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Encoder(irreps_in, irreps_latent, model_type)
        f = dudt(nEq_NLMP2, irreps_latent, latent_layers)
        dec = Decoder(irreps_latent, irreps_out, model_type)
    elif model_type == 'non-equivariant isotropic':
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Linear(irreps_in, irreps_latent)
        f = dudt(nEq_NLMP_iso, irreps_latent, latent_layers)
        dec = Linear(irreps_latent, irreps_out)
    elif model_type == 'non-equivariant anisotropic':
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Linear(irreps_in, irreps_latent)
        f = dudt(nEq_NLMP_aniso, irreps_latent, latent_layers)
        dec = Linear(irreps_latent, irreps_out)
    elif model_type == 'GCN':
        irreps_in = o3.Irreps(f'{node_basis:g}'+'x0e')+o3.Irreps(irreps_in)
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Encoder(irreps_in, irreps_latent, model_type)
        f = dudt(GCN, irreps_latent, latent_layers)
        dec = Decoder(irreps_latent, irreps_out, model_type)
    elif model_type == 'Edge':
        irreps_in = o3.Irreps(irreps_in).dim
        irreps_out = o3.Irreps(irreps_out).dim
        irreps_latent = latent_scalars + 3*latent_vectors + 5*latent_tensors
        enc = Linear(irreps_in, irreps_latent)
        f = dudt(Edge, irreps_latent, latent_layers)
        dec = Linear(irreps_latent, irreps_out)
    else:
        print('Unknown model type: ', model_type)
    return enc, f, dec