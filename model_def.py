import torch
import copy, sys
sys.path.append('/home/opc/data/ml-cfd/FlowModel')
from flowmodel.nn.layers import build_model
from torchmetrics import MeanSquaredError
from torch_scatter import scatter
from e3nn import o3, nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR
import pytorch_lightning as pl


class LitModel(pl.LightningModule):
    def __init__(self, irreps_in, irreps_out,
                latent_layers=4, latent_scalars=8, latent_vectors=8, latent_tensors=0,
                model_type='equivariant', noise_var=0,
                lr=1e-3, epochs=None, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        self.enc, self.f, self.dec = build_model(model_type, 
            irreps_in, irreps_out,
            latent_layers, latent_scalars, latent_vectors, latent_tensors)
        self.metric = MeanSquaredError()
        self.val_metric = MeanSquaredError()
        self.lr = lr
        self.epochs = epochs
        self.noise_var = noise_var

    def forward(self, data):

        _ = data.rotate(o3.rand_matrix().type_as(data.x)) if self.model_type != 'equivariant' and self.training else 0
        # _ = data.rotate(o3.rand_matrix().type_as(data.x)) if self.training else 0
        data.embed()
        h = data.x
        h = self.enc(h)
        if self.model_type == 'equivariant':
            h = h + self.f.layers[0].irreps_input.randn(h.shape[0],-1).type_as(h)*(self.noise_var**.5) if self.training else h
        else:
            h = h + torch.randn(h.shape).type_as(h)*(self.noise_var**.5) if self.training else h
        hs = []
        for i in range(data.y.shape[0]):
            h = h + self.f(h, data)*data.dts[i]
            hs.append(h)
        hs = torch.stack(hs)
        hs = self.dec(hs)

        return hs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_sch = OneCycleLR(optimizer, self.lr, self.epochs, pct_start=0.2)
        # lr_sch1 = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
        # lr_sch2 = StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [lr_sch]

    def training_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = self.metric(y_hat, data.y)
        self.log('train_loss', loss, batch_size=data.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = self.val_metric(y_hat, data.y)
        self.log('val_loss', loss, batch_size=data.num_graphs)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            data = batch
            y_hat = self(data)
            loss = torch.nn.functional.mse_loss(y_hat, data.y)
            self.log('test_loss', loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)

            rot = o3.rand_matrix().type_as(data.x)
            # rot_data, _, _ = self.rotate_data(data, '7x0e+1x1o', '1x0e+1x1o', rot)
            rot_data, D_out = copy.deepcopy(data).rotate(rot)
            y_hat_rot = self(rot_data)
            rot_loss = torch.nn.functional.mse_loss(y_hat_rot, rot_data.y)
            eq_loss = torch.nn.functional.mse_loss(y_hat_rot, y_hat @ D_out.T)
            self.log('test_rot_loss', rot_loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)
            self.log('eq_loss', eq_loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'data.pt')
        else:
            data = batch
            y_hat = self(data)
            loss = torch.nn.functional.mse_loss(y_hat, data.y)
            self.log('test_rollout_loss', loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'data_rollout.pt')

        return loss

    # def rotate_data(self, data, irreps_in, irreps_out, rot):
    #     D_in = o3.Irreps(irreps_in).D_from_matrix(rot).type_as(data.x)
    #     D_out = o3.Irreps(irreps_out).D_from_matrix(rot).type_as(data.x)
    #     rot_data = copy.deepcopy(data)
    #     rot_data.x = data.x @ D_in.T
    #     rot_data.pos = data.pos @ rot.T
    #     rot_data.y = data.y @ D_out.T
    #     return rot_data, D_in, D_out