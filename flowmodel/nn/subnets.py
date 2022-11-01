import torch
from e3nn import o3
from e3nn.nn import NormActivation
from .layers import *
from .utils import *
from ..data.modules import Data

## Subnets ##
class GraphPool(torch.nn.Module):
    def __init__(self, cf, irreps_latent, irreps_sh, **kwargs):
        super(GraphPool, self).__init__()
        
        self.f0 = torch.nn.Sequential(
            Eq_NLMP3(irreps_latent, irreps_latent, **kwargs), NormLayer(),
            Eq_NLMP3(irreps_latent, irreps_latent, **kwargs), NormLayer()
        )
        self.cf = cf
        self.lin = o3.Linear(irreps_latent, irreps_sh)
        self.tp = o3.FullyConnectedTensorProduct(irreps_sh, irreps_sh, '0e')
        self.edge_enc = torch.nn.Sequential(
            o3GatedLinear(irreps_sh, irreps_latent, **kwargs),
            o3.Linear(irreps_latent, irreps_latent)
        )

    def forward(self, data, r):

        data = self.f0(data)

        score = self.lin(data.hn)
        score = self.tp(score, score)

        idx = torch.topk(score.flatten(),int(data.num_nodes/self.cf),dim=0).indices
        gate = torch.sigmoid(score[idx])

        # idx = torch.randperm(data.num_nodes)[:int(data.num_nodes/self.cf)]
        subdata = Data(hn=data.hn[idx]*gate, pos=data.pos[idx], batch=data.batch[idx], idx=idx)
        subdata = subdata.resample_edges(r)
        subdata.he = self.edge_enc(subdata.fe)

        return subdata

class vWrap(torch.nn.Module):
    def __init__(self, layer_type, irreps_input, irreps_output, *args, 
                num_levels=1, skip_mp_levels=[], **kwargs):
        super(vWrap, self).__init__()

        self.num_levels = num_levels
        self.layer_type = layer_type
        self.layers = torch.nn.ModuleList()
        for i in range(num_levels):
            if i in skip_mp_levels:
                l = torch.nn.Identity()
            else:
                l = layer_type(irreps_input, irreps_output, *args, **kwargs)
            self.layers.append(l)
        # self.down = torch.nn.ModuleList()
        # for i in range(num_levels-1):
        #     self.down.append(o3.Linear(2*irreps_output, irreps_output))
        self.up = torch.nn.ModuleList()
        for i in range(num_levels-1):
            self.up.append(o3.Linear(2*irreps_output, irreps_output))
        
    def forward(self, data_list):

        for i in range(self.num_levels):
            data_list[i] = self.layers[i](data_list[i])

        # for i in range(self.num_levels-1):
        #     data_list[i+1].hn += self.down[i](
        #         torch.cat([data_list[i].hn[data_list[i+1].idx],data_list[i+1].hn],dim=1))

        for i in reversed(range(self.num_levels-1)):
            input = 0*data_list[i].hn; input[data_list[i+1].idx] = data_list[i+1].hn
            data_list[i].hn += self.up[i](
                torch.cat([input,data_list[i].hn],dim=1))

        return data_list

class vNormLayer(torch.nn.Module):
    def __init__(self):
        super(vNormLayer, self).__init__()
        self.NL = NormLayer()
    def forward(self, data_list):

        for i in range(len(data_list)):
            data_list[i] = self.NL(data_list[i])

        return data_list

class Latent(torch.nn.Module):
    def __init__(self, layer_type, irreps_latent, num_layers, **kwargs):
        super(Latent, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.layers.append(vNormLayer())
        for i in range(num_layers):
            sml = kwargs.get('skip_mp_levels_all', None)
            skip_mp_levels = [] if sml is None else sml[i]
            self.layers.append(vWrap(layer_type, 
                irreps_latent, irreps_latent, skip_mp_levels=skip_mp_levels, **kwargs))
            self.layers.append(vNormLayer())

    def forward(self, data):

        for i in range(len(self.layers)):
            data = self.layers[i](data)

        return data

class Encoder(torch.nn.Module):
    def __init__(self, irreps_in, irreps_latent, model_type='equivariant', **kwargs):
        super(Encoder, self).__init__()
        self.irreps_latent = irreps_latent
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        if model_type=='equivariant':
            self.node_enc = torch.nn.Sequential(#NormLayer(),
                o3GatedLinear(irreps_in, irreps_latent, **kwargs),
                o3.Linear(irreps_latent, irreps_latent)
            )
            self.edge_enc = torch.nn.Sequential(
                o3GatedLinear(self.irreps_sh, irreps_latent, **kwargs),
                o3.Linear(irreps_latent, irreps_latent)
            )
        else:
            self.node_enc = torch.nn.Sequential(
                Linear(irreps_in.dim, irreps_latent.dim), kwargs.get('act', ReLU()),
                Linear(irreps_latent.dim, irreps_latent.dim)
            )
            self.edge_enc = torch.nn.Sequential(
                Linear(self.irreps_sh.dim, irreps_latent.dim), kwargs.get('act', ReLU()),
                Linear(irreps_latent.dim, irreps_latent.dim)
            )

    def forward(self, data):
        # data.embed()
        data.hn = torch.cat([data.fn,data.hn],dim=-1)
        data.he = data.fe#torch.cat([data.edge_vec.norm(dim=1,keepdim=True),data.edge_vec],dim=-1)
        data.hn = self.node_enc(data.hn)
        data.he = self.edge_enc(data.he)
        return data

class Decoder(torch.nn.Module):
    def __init__(self, irreps_latent, irreps_out, model_type='equivariant', **kwargs):
        super(Decoder, self).__init__()

        if model_type=='equivariant':
            self.node_dec = torch.nn.Sequential(
                o3GatedLinear(irreps_latent, irreps_latent, **kwargs),
                o3.Linear(irreps_latent, irreps_out)
            )
        else:
            self.node_dec = torch.nn.Sequential(
                Linear(irreps_latent.dim, irreps_latent.dim), kwargs.get('act', ReLU()),
                Linear(irreps_latent.dim, irreps_out.dim)
            )

    def forward(self, data):
        data.hn = self.node_dec(data.hn)
        return data

class dudt(torch.nn.Module):
    def __init__(self, enc, f, dec, num_levels=1, cf=[], **kwargs):
        super(dudt, self).__init__()

        self.num_levels = num_levels
        self.enc = enc
        self.gps = torch.nn.ModuleList()
        for i in range(num_levels-1): 
            self.gps.append(GraphPool(cf[i], enc.irreps_latent, enc.irreps_sh))
        self.f = f
        self.dec = dec
        self.data = None

    def forward(self, t, u):
        self.data.hn = u

        self.data = self.data.resample_edges(self.data.rc[0][0])
        self.data = self.enc(self.data)

        self.data = [self.data]
        for i in range(self.num_levels-1):
            self.data.append(self.gps[i](self.data[i], self.data[0].rc[0][i+1]))

        data = self.f(self.data); data[0].data1_idx.append(data[1].idx)
        self.data = data[0]

        self.data = self.dec(self.data)
        return self.data.hn

class NODE(torch.nn.Module):
    def __init__(self, enc, f, dec, solver='naive_euler',
        sensitivity='autograd', **kwargs):
        super(NODE, self).__init__()

        self.dudt = dudt(enc, f, dec, **kwargs)
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