import torch
import copy, sys
from .build_model import build_model
from e3nn import o3
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR, OneCycleLR
import pytorch_lightning as pl
from torchmetrics import Metric

class RunningVar(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("var", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        self.var = self.var + torch.mean((preds - target)**2, dim=0, keepdim=True)
        self.total += 1

    def compute(self):
        return self.var / self.total

class LitModel(pl.LightningModule):
    def __init__(self, irreps_in, irreps_out, loss_fn,
                latent_layers=4, latent_scalars=8, latent_vectors=8, latent_tensors=0,
                model_type='equivariant', noise_var=0, noise_fac=0, data_aug=False,
                lr=1e-3, epochs=None, lr_sch=False, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_type = model_type
        self.node = build_model(model_type, 
            irreps_in, irreps_out,
            latent_layers, latent_scalars, latent_vectors, latent_tensors, **kwargs)
        self.loss_fn = loss_fn
        self.lr = lr
        self.lr_sch = lr_sch
        self.epochs = epochs
        self.noise_var = noise_var
        self.noise_fac = noise_fac
        self.var = RunningVar()
        self.data_aug = data_aug

    def forward(self, data):
        data.data1_idx = []
        _ = data.rotate(o3.rand_matrix()) if self.data_aug and self.training else 0

        xi = data.x
        if self.training:
            xi = xi + o3.Irreps(data.irreps_io[0][0]).randn(xi.shape[0],-1).type_as(xi) * \
            (self.noise_var)**.5
        
        self.node.dudt.data = data
        t, yhs = self.node(xi, data.ts[0,:])

        return yhs[1:,:,:]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if self.lr_sch:
            lr_sch = OneCycleLR(optimizer, self.lr, self.epochs, pct_start=0.1)
            # lr_sch1 = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
            # lr_sch2 = StepLR(optimizer, step_size=20, gamma=0.1)
            return [optimizer], [lr_sch]
        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        data = batch#.subsample()
        y_hat = self(data)
        loss = self.loss_fn(y_hat, data)
        self.var.update(y_hat[-1], data.y.transpose(0,1)[-1])
        self.log('train_loss', loss, batch_size=data.num_graphs)
        return loss
    
    def training_epoch_end(self, outputs):
        frac = self.noise_fac
        self.noise_var = self.noise_var * (1-frac) + self.var.compute() * frac 
        self.var.reset()

    def validation_step(self, batch, batch_idx):
        data = batch
        y_hat = self(data)
        loss = self.loss_fn(y_hat, data)
        # loss = self.metric(io.CartesianTensor('ij=ji').to_cartesian(y_hat), io.CartesianTensor('ij=ji').to_cartesian(data.y))
        self.log('val_loss', loss, batch_size=data.num_graphs)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            data = batch

            rot_data = copy.deepcopy(data)
            rot = o3.rand_matrix()
            rot_data, D_out = rot_data.rotate(rot)

            y_hat = self(data)
            y_hat_rot = self(rot_data)

            loss = self.loss_fn(y_hat, data)
            # loss = self.metric(io.CartesianTensor('ij=ji').to_cartesian(y_hat), io.CartesianTensor('ij=ji').to_cartesian(data.y))
            self.log('test_loss', loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)

            rot_loss = self.loss_fn(y_hat_rot, rot_data)
            # rot_loss = self.metric(io.CartesianTensor('ij=ji').to_cartesian(y_hat_rot), io.CartesianTensor('ij=ji').to_cartesian(rot_data.y))
            eq_loss = torch.nn.functional.mse_loss(y_hat @ D_out.T, y_hat_rot)
            self.log('test_rot_loss', rot_loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)
            self.log('eq_loss', eq_loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'data.pt')
        else:
            data = batch
            y_hat = self(data)
            loss = self.loss_fn(y_hat, data)
            self.log('test_rollout_loss', loss, batch_size=data.num_graphs,
                add_dataloader_idx=False)

            if batch_idx == 0 and self.global_rank == 0:
                torch.save((data,y_hat),'data_rollout.pt')

        return loss
