import torch
import copy
from torchmetrics import MeanSquaredError
from torch_scatter import scatter
from torch.nn import Sequential as Seq, Linear, ReLU, Tanh
from e3nn import o3, nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
import pytorch_lightning as pl

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

    def forward(self, x, data):
        
        edge_src, edge_dst = data.edge_index
        sh = o3.spherical_harmonics(self.irreps_sh, data.edge_vec, normalize=True, normalization='component')
        edge_ftr = self.gate(self.tp(torch.cat([x[edge_src],x[edge_dst]],dim=1), sh, self.fc(data.emb))) * data.norm.view(-1, 1)

        return scatter(edge_ftr, edge_dst, dim=0, dim_size=data.num_nodes)

class dudt(torch.nn.Module):
    def __init__(self, irreps_latent, num_layers):
        super(dudt, self).__init__()

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(NLMP(irreps_latent, irreps_latent))

    def forward(self, x, data):

        h = x
        for i in range(len(self.layers)):
            h = self.layers[i](h, data)

        return h


class LitModel(pl.LightningModule):
    def __init__(self, irreps_in, irreps_latent, irreps_out, latent_layers=4, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.enc = o3.Linear(irreps_in, irreps_latent)
        self.f = dudt(irreps_latent, latent_layers)
        self.dec = o3.Linear(irreps_latent, irreps_out)
        self.metric = MeanSquaredError()
        self.lr = lr

    def forward(self, data):

        data.embed()
        h = data.x
        h = self.enc(h)
        hs = []
        for i in range(data.y.shape[0]):
            h = h + self.f(h, data)*data.dts[i]
            hs.append(h)
        hs = torch.stack(hs)
        hs = self.dec(hs)

        return hs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_sch1 = CosineAnnealingWarmRestarts(optimizer, T_0=20)
        lr_sch2 = StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [lr_sch1, lr_sch2]

    def training_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = self.metric(y_hat, data.y)
        self.log('train_loss', loss, batch_size=data.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = self.metric(y_hat, data.y)
        self.log('val_loss', loss, batch_size=data.num_graphs)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            data = batch
            y_hat = self(data)
            loss = self.metric(y_hat, data.y)
            self.log('test_loss', loss, batch_size=data.num_graphs)

            rot = o3.rand_matrix().type_as(data.x)
            rot_data, _, _ = self.rotate_data(data, '7x0e+1x1o', '1x0e+1x1o', rot)
            y_hat_rot = self(rot_data)
            rot_loss = self.metric(y_hat_rot, rot_data.y)
            self.log('test_rot_loss', rot_loss, batch_size=data.num_graphs)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'data.pt')
        else:
            data = batch
            y_hat = self(data)
            loss = self.metric(y_hat, data.y)
            self.log('test_rollout_loss', loss, batch_size=data.num_graphs)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'data_rollout.pt')

        return loss

    def rotate_data(self, data, irreps_in, irreps_out, rot):
        D_in = o3.Irreps(irreps_in).D_from_matrix(rot).type_as(data.x)
        D_out = o3.Irreps(irreps_out).D_from_matrix(rot).type_as(data.x)
        rot_data = copy.deepcopy(data)
        rot_data.x = data.x @ D_in.T
        rot_data.pos = data.pos @ rot.T
        rot_data.y = data.y @ D_out.T
        return rot_data, D_in, D_out

def build_model(in_scalars, in_vectors,
                latent_layers, latent_scalars, latent_vectors,
                out_scalars, out_vectors,
                lr):

    irreps_in = f'{in_scalars:g}'+'x0e + '+f'{in_vectors:g}'+'x1o'
    irreps_out = f'{out_scalars:g}'+'x0e + '+f'{out_vectors:g}'+'x1o'
    irreps_latent = f'{latent_scalars:g}'+'x0e + '+f'{latent_vectors:g}'+'x1o'

    return LitModel(irreps_in, irreps_latent, irreps_out, latent_layers, lr)