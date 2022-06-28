import torch
from e3nn import o3
from .layers import *
from .utils import *

## Subnets ##
class Latent(torch.nn.Module):
    def __init__(self, layer_type, irreps_latent, num_layers, **kwargs):
        super(Latent, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(NormLayer())
        for i in range(num_layers):
            self.layers.append(layer_type(irreps_latent, irreps_latent, **kwargs))
            self.layers.append(NormLayer())

    def forward(self, data):

        for i in range(len(self.layers)):
            self.layers[i](data)

        return data

class Encoder(torch.nn.Module):
    def __init__(self, irreps_in, irreps_latent, model_type='equivariant', **kwargs):
        super(Encoder, self).__init__()

        if model_type=='equivariant' or model_type=='reservoir':
            self.node_enc = torch.nn.Sequential(
                o3GatedLinear(irreps_in, irreps_latent, **kwargs),
                o3.Linear(irreps_latent, irreps_latent)
            )
            self.edge_enc = torch.nn.Sequential(
                o3GatedLinear('0e+1o', irreps_latent, **kwargs),
                o3.Linear(irreps_latent, irreps_latent)
            )
        else:
            self.node_enc = torch.nn.Sequential(
                Linear(irreps_in, irreps_latent), kwargs.get('act', SiLU()),
                Linear(irreps_latent, irreps_latent)
            )
            self.edge_enc = torch.nn.Sequential(
                Linear(4, irreps_latent), kwargs.get('act', SiLU()),
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
    def __init__(self, irreps_latent, irreps_out, model_type='equivariant', **kwargs):
        super(Decoder, self).__init__()

        if model_type=='equivariant' or model_type=='reservoir':
            self.node_dec = torch.nn.Sequential(
                o3GatedLinear(irreps_latent, irreps_latent, **kwargs),
                o3.Linear(irreps_latent, irreps_out)
            )
        else:
            self.node_dec = torch.nn.Sequential(
                Linear(irreps_latent, irreps_latent), kwargs.get('act', SiLU()),
                Linear(irreps_latent, irreps_out)
            )

    def forward(self, data):
        data.hn = self.node_dec(data.hn)
        return data

class dudt(torch.nn.Module):
    def __init__(self, enc, f, dec):
        super(dudt, self).__init__()

        self.enc = enc
        self.f = f
        self.dec = dec
        self.data = None

    def forward(self, t, u):
        self.data.hn = u
        self.enc(self.data)
        self.f(self.data)
        self.dec(self.data)
        return self.data.hn

class NODE(torch.nn.Module):
    def __init__(self, enc, f, dec, solver='naive_euler',
        sensitivity='interpolated_adjoint', **kwargs):
        super(NODE, self).__init__()

        self.dudt = dudt(enc, f, dec)
        self.solver = solver
        if solver != 'naive_euler':
            self.ode = NeuralODE(self.dudt, sensitivity=sensitivity, solver=solver)

    def forward(self, u, t_span):
        if self.solver == 'naive_euler':
            t = t_span; sol = [u]; up = u
            for i in range(len(t)-1):
                up = up + self.dudt(up)*(t[i+1]-t[i])
                sol.append(up)
            sol = torch.stack(sol)
        else:
            t, sol = self.ode(u, t_span)
        return t, sol