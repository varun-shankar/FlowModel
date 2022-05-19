import torch
from typing import Optional
import random
import numpy as np
from torch_cluster import knn_graph, radius_graph
from .load_OF import load_case, load_kaggle, load_nek
from .load_SU2 import load_SU2
import glob, re
import torch.nn.functional as F
from torch_geometric.data import Data as pygData
from torch_geometric.utils import degree
from e3nn.math import soft_one_hot_linspace
from e3nn import o3, io
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

###### Data Class ###################################################################################

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
        return self, D_out

    def embed(self, num_basis=10):
        edge_src, edge_dst = self.edge_index
        deg = degree(edge_dst, self.num_nodes).type_as(self.x)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.norm = deg_inv_sqrt[edge_src] * deg_inv_sqrt[edge_dst]

        self.edge_vec = self.pos[edge_dst] - self.pos[edge_src]
        rc = float(self.rc.max()) if torch.is_tensor(self.rc) else self.rc
        self.emb = soft_one_hot_linspace(self.edge_vec.norm(dim=1), 0.0, rc, num_basis, 
            basis='fourier', cutoff=True).mul(num_basis**0.5)


###### OpenFoam #####################################################################################

class OFDataModule(pl.LightningDataModule):
    def __init__(self, case, zones, ts, rc,
                       num_nodes=-1, 
                       rollout=1,
                       data_fields=['p','U'], data_irreps=['0e+1o','0e+1o'],
                       knn=False,
                       train_split=0.9, random_split=False,
                       test_ts=[], test_rollout=0,
                       shuffle=True, batch_size=1, **kwargs):
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
        self.irreps_io = self.data_irreps#[f'{len(self.zones):g}'+'x0e+'+self.data_irreps[0],self.data_irreps[1]]
        self.__dict__.update(kwargs)

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit':
            self.ts = self.ts if torch.is_tensor(self.ts) else torch.arange(self.ts[0],
                self.ts[1]+1e-8,step=self.dt)
            self.test_ts = self.test_ts if torch.is_tensor(self.test_ts) else torch.arange(self.test_ts[0],
                self.test_ts[1]+1e-8,step=self.dt)
            all_ts = torch.cat([self.ts,self.test_ts],dim=0)

            ### Read data ###
            p, b, v = load_case(self.case, self.data_fields, all_ts, self.zones)

            # Sample internal nodes
            if self.num_nodes != -1:
                n_bounds = sum([len(b[i+1]) for i in range(len(b)-1)])
                torch.manual_seed(42)
                idx = torch.randperm(p[0].shape[0])[:self.num_nodes-n_bounds]
                p[0] = p[0][idx,:]; b[0] = b[0][idx]; v[0] = v[0][:,idx,:]

            pos = torch.cat(p,dim=0)
            b1hot = F.one_hot(torch.cat(b,dim=0)).float()
            v = torch.cat(v,dim=1)
            # v = torch.nn.LayerNorm(v.shape[1:],elementwise_affine=False)(v)
            # v[:,:,0:1] = (v[:,:,0:1]-torch.mean(v[:,:,0:1],dim=(0,1),keepdim=True))/ \
            #     torch.std(v[:,:,0:1],dim=(0,1),keepdim=True)
            # v[:,:,1:] = (v[:,:,1:]-torch.mean(v[:,:,1:],dim=(0,1),keepdim=True))/ \
            #     torch.std(v[:,:,1:],dim=(0,1,2),keepdim=True)
            # v = (v - torch.mean(v,dim=(0,1),keepdim=True))/ \
            #     (torch.amax(v,dim=(0,1),keepdim=True)-torch.amin(v,dim=(0,1),keepdim=True))
            um = v[:,:,1:].norm(dim=-1).mean()
            v[:,:,:1] /= um**2
            v[:,:,1:] /= um
            fields = v
            # v = torch.cat([v[:,:,0:1],v[:,:,1:].norm(dim=-1,keepdim=True),v[:,:,1:]],dim=-1)
            features = v#torch.cat([b1hot.unsqueeze(0).repeat(len(all_ts),1,1),v],dim=-1)

            # Generate graph
            if self.knn:
                edge_index = knn_graph(pos, k=self.rc)
            else:
                edge_index = radius_graph(pos, r=self.rc, max_num_neighbors=32)
            print('Avg neighbors = ', edge_index.shape[1]/pos.shape[0])

            ### Generate dataset ###
            dataset = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.rollout,:,:], irreps_io=self.irreps_io,
                            dts=torch.diff(all_ts[i:i+1+self.rollout]), emb_node=b1hot,
                            pos=pos, edge_index=edge_index, rc=self.rc) for i in range(len(self.ts)-self.rollout)]
            if self.random_split:
                random.shuffle(dataset)
            self.train_data = dataset[:int((len(dataset)+1)*self.train_split)] 
            self.val_data = dataset[int((len(dataset)+1)*self.train_split):]

            # Test
            testset = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.rollout,:,:], irreps_io=self.irreps_io,
                            dts=torch.diff(all_ts[i:i+1+self.rollout]), emb_node=b1hot,
                            pos=pos, edge_index=edge_index, rc=self.rc) for i in range(
                                len(self.ts),len(self.ts)+len(self.test_ts)-self.rollout)]
            testset_rollout = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.test_rollout,:,:], irreps_io=self.irreps_io,
                            dts=torch.diff(all_ts[i:i+1+self.test_rollout]), emb_node=b1hot,
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


###### Kaggle #######################################################################################

class KaggleDataModule(pl.LightningDataModule):
    def __init__(self, turb_model, geometry, rc,
                       dir = '../../kaggle_data/',
                       num_nodes=-1, 
                       data_fields=['wallDistance','p','U'], data_irreps='2x0e+1o',
                       label='tau', label_irrep=io.CartesianTensor('ij=ji'),
                       knn=False,
                       train_split=0.8, random_split=False,
                       shuffle=True, batch_size=1, **kwargs):
        super().__init__()
        self.turb_model = turb_model
        self.geometry = geometry
        self.rc = rc
        self.dir = dir
        self.num_nodes = num_nodes
        self.data_fields = data_fields
        self.data_irreps = data_irreps
        self.label = label
        self.label_irrep = label_irrep
        self.knn = knn
        self.train_split = train_split
        self.random_split = random_split
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.irreps_io = [self.data_irreps,self.label_irrep.__repr__()]

    def setup(self, stage: Optional[str] = None):
        
        if stage == 'fit':
            ### Read data ###
            self.cases = []
            self.test_cases = []
            for g in self.geometry:
                c = glob.glob(self.dir+self.turb_model+'/'+self.turb_model+'_'+g+'*_p.npy')
                c = [i.replace(self.dir+self.turb_model+'/'+self.turb_model+'_','').replace('_p.npy','') for i in c]
                self.test_cases.append(c[0])#c.pop(0))
                self.cases.extend(c)

            ### Generate dataset ###
            dataset = []
            for c in self.cases:
                p, v, l = load_kaggle(self.dir,self.turb_model,c,fields=self.data_fields,label=self.label)
                if self.num_nodes != -1:
                    torch.manual_seed(42)
                    idx = torch.randperm(p.shape[0])[:self.num_nodes]
                    p = p[idx,:]; v = v[idx,:]; l = torch.index_select(l,0,idx)
                pos = p
                
                emb_node = v[:,0:1]; v = v[:,1:]
                # v[:,0:2] = (v[:,0:2]-torch.mean(v[:,0:2],dim=(0,),keepdim=True))/ \
                #     torch.std(v[:,0:2],dim=(0,),keepdim=True)
                # v[:,2:] = (v[:,2:]-torch.mean(v[:,2:],dim=(0,),keepdim=True))/ \
                #     torch.std(v[:,2:],dim=(0,1),keepdim=True)
                um = v[:,1:].norm(dim=-1).mean()
                v[:,:1] /= um**2
                v[:,1:] /= um
                v = torch.cat([v[:,:1],v[:,1:].norm(dim=-1,keepdim=True),v[:,1:]],dim=-1)
                features = v
                l /= um**2
                fields = self.label_irrep.from_cartesian(l)
                # fields = (fields-torch.mean(fields,dim=(0,),keepdim=True))/ \
                #     torch.std(fields,dim=(0,1),keepdim=True)

                rc = self.rc if isinstance(self.rc, float) else self.rc[self.geometry.index(re.sub(r'_.*','',c))]
                if self.knn:
                    edge_index = knn_graph(pos, k=rc)
                else:
                    edge_index = radius_graph(pos, r=rc, loop=True, max_num_neighbors=32)
                print('Avg neighbors = ', edge_index.shape[1]/pos.shape[0])

                data = Data(x=features, y=fields.unsqueeze(0), irreps_io=self.irreps_io,
                            dts=torch.ones(1), emb_node=emb_node,
                            pos=pos, edge_index=edge_index, rc=rc)
                dataset.append(data)

            if self.random_split:
                random.shuffle(dataset)
            self.train_data = dataset[:int((len(dataset)+1)*self.train_split)] 
            self.val_data = dataset[int((len(dataset)+1)*self.train_split):]

            # Test
            testset = []
            for c in self.test_cases:
                p, v, l = load_kaggle(self.dir,self.turb_model,c,fields=self.data_fields,label=self.label)
                if self.num_nodes != -1:
                    idx = torch.randperm(p.shape[0])[:self.num_nodes]
                    p = p[idx,:]; v = v[idx,:]; l = torch.index_select(l,0,idx)
                pos = p

                emb_node = v[:,0:1]; v = v[:,1:]
                um = v[:,1:].norm(dim=-1).mean()
                v[:,:1] /= um**2
                v[:,1:] /= um
                v = torch.cat([v[:,:1],v[:,1:].norm(dim=-1,keepdim=True),v[:,1:]],dim=-1)
                features = v
                l /= um**2
                fields = self.label_irrep.from_cartesian(l)

                rc = self.rc if isinstance(self.rc, float) else self.rc[self.geometry.index(re.sub(r'_.*','',c))]
                if self.knn:
                    edge_index = knn_graph(pos, k=rc)
                else:
                    edge_index = radius_graph(pos, r=rc, loop=True, max_num_neighbors=32)
                print('Avg neighbors = ', edge_index.shape[1]/pos.shape[0])

                data = Data(x=features, y=fields.unsqueeze(0), irreps_io=self.irreps_io,
                            dts=torch.ones(1), emb_node=emb_node,
                            pos=pos, edge_index=edge_index, rc=rc)
                testset.append(data)

            self.test_data = testset

            print('Cases: ', self.cases)
            print('Test Cases: ', self.test_cases)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)


###### NEK ##########################################################################################

class NekDataModule(pl.LightningDataModule):
    def __init__(self, ts, rc,
                       dir='/home/opc/data/nekData/',
                       nstep=4, num_nodes=5000,
                       prefetched=True,
                       rollout=1,
                       data_fields=['p','U'], data_irreps=['0e+1o','0e+1o'],
                       knn=False,
                       train_split=0.9, random_split=False,
                       test_ts=[], test_rollout=0,
                       shuffle=True, batch_size=1, **kwargs):
        super().__init__()
        self.ts = ts
        self.rc = rc
        self.dir = dir
        self.nstep = nstep
        self.num_nodes = num_nodes
        self.prefetched = prefetched
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
        self.irreps_io = self.data_irreps
        self.__dict__.update(kwargs)

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit':

            ### Read data ###
            if self.prefetched:
                pos, v, all_ts = torch.load(self.dir+'nekData.pt')
            else:
                pos, v, all_ts = load_nek(self.dir, step=self.nstep, 
                    tsteps=self.ts+self.test_ts, num_nodes=self.num_nodes)
                torch.save((pos, v, all_ts), self.dir+'nekData.pt')

            nts = self.ts; ntts = self.test_ts
            self.ts = all_ts[:nts]; self.test_ts = all_ts[nts:nts+ntts]
            # v[:,:,0:1] = (v[:,:,0:1]-torch.mean(v[:,:,0:1],dim=(0,1),keepdim=True))/ \
            #     torch.std(v[:,:,0:1],dim=(0,1),keepdim=True)
            # v[:,:,1:] = (v[:,:,1:]-torch.mean(v[:,:,1:],dim=(0,1),keepdim=True))/ \
            #     torch.std(v[:,:,1:],dim=(0,1,2),keepdim=True)
            # v = torch.cat([v[:,:,0:1],v[:,:,1:].norm(dim=-1,keepdim=True),v[:,:,1:]],dim=-1)
            # um = v[:,:,1:].norm(dim=-1).mean()
            # v[:,:,:1] /= um**2
            # v[:,:,1:] /= um
            fields = v
            features = v
            emb_node = torch.ones(pos.shape[0],1)

            # Generate graph
            if self.knn:
                edge_index = knn_graph(pos, k=self.rc)
            else:
                edge_index = radius_graph(pos, r=self.rc, max_num_neighbors=32)
            print('Avg neighbors = ', edge_index.shape[1]/pos.shape[0])

            ### Generate dataset ###
            dataset = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.rollout,:,:], irreps_io=self.irreps_io,
                            dts=torch.diff(all_ts[i:i+1+self.rollout]), emb_node=emb_node,
                            pos=pos, edge_index=edge_index, rc=self.rc) for i in range(len(self.ts)-self.rollout)]
            if self.random_split:
                random.shuffle(dataset)
            self.train_data = dataset[:int((len(dataset)+1)*self.train_split)] 
            self.val_data = dataset[int((len(dataset)+1)*self.train_split):]

            # Test
            testset = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.rollout,:,:], irreps_io=self.irreps_io,
                            dts=torch.diff(all_ts[i:i+1+self.rollout]), emb_node=emb_node,
                            pos=pos, edge_index=edge_index, rc=self.rc) for i in range(
                                len(self.ts),len(self.ts)+len(self.test_ts)-self.rollout)]
            testset_rollout = [Data(x=features[i,:,:], y=fields[i+1:i+1+self.test_rollout,:,:], irreps_io=self.irreps_io,
                            dts=torch.diff(all_ts[i:i+1+self.test_rollout]), emb_node=emb_node,
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


###### SU2 #####################################################################################

class SU2DataModule(pl.LightningDataModule):
    def __init__(self, markers, rc,
                       dir='/home/opc/data/ml-cfd/SU2/platform_bump/data/',
                       test_dir='/home/opc/data/ml-cfd/SU2/platform_bump/test_data/',
                       num_data=200, num_test=10, num_nodes=-1, 
                       data_fields=['Pressure','Velocity'], data_irreps=['1o','0e+2x1o'],
                       knn=False,
                       train_split=0.9, random_split=False,
                       shuffle=True, batch_size=1, **kwargs):
        super().__init__()
        self.markers = markers
        self.rc = rc
        self.dir = dir
        self.test_dir = test_dir
        self.num_data = num_data
        self.num_test = num_test
        self.num_nodes = num_nodes
        self.data_fields = data_fields
        self.data_irreps = data_irreps
        self.knn = knn
        self.train_split = train_split
        self.random_split = random_split
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.irreps_io = self.data_irreps
        self.__dict__.update(kwargs)

    def setup(self, stage: Optional[str] = None):

        if stage == 'fit':
            dataset = []
            neighbors = []
            for i in range(self.num_data):
                p, b, v, s = load_SU2(self.dir+f'mesh_{i}.su2', 
                    self.dir+f'sens_{i}.vtk', 
                    self.dir+f'flow_{i}.vtk', self.markers, self.data_fields)

                # Sample internal nodes
                if self.num_nodes != -1:
                    n_bounds = sum(b!=0)
                    torch.manual_seed(42)
                    idx = torch.cat([(b!=0).nonzero().flatten(),
                        (b==0).nonzero().flatten()[
                            torch.randperm(sum(b==0))[:self.num_nodes-n_bounds]]],dim=0)
                    p = p[idx,:]; b = b[idx]; v = v[idx,:]

                pos = p
                b1hot = F.one_hot(b.long()).float()
                um = v[:,1:].norm(dim=-1).mean()
                v[:,:1] /= um**2
                v[:,1:] /= um
                sens = torch.zeros(p.shape[0],3)
                sens[b==1,:] = s #sens calc on first marker
                v = torch.cat([v,sens], dim=-1)

                # Generate graph
                if self.knn:
                    edge_index = knn_graph(pos, k=self.rc)
                else:
                    edge_index = radius_graph(pos, r=self.rc, max_num_neighbors=32)
                neighbors.append(edge_index.shape[1]/pos.shape[0])

                data = Data(x=p-p.mean(dim=0), y=v.unsqueeze(0), irreps_io=self.irreps_io,
                            dts=torch.ones(1), emb_node=b1hot,
                            pos=pos, edge_index=edge_index, rc=self.rc)
                dataset.append(data)
            print('Avg neighbors = ', sum(neighbors)/len(neighbors))
            if self.random_split:
                random.shuffle(dataset)
            self.train_data = dataset[:int((len(dataset)+1)*self.train_split)] 
            self.val_data = dataset[int((len(dataset)+1)*self.train_split):]

            # Test
            testset = []
            neighbors = []
            for i in range(self.num_test):
                p, b, v, s = load_SU2(self.test_dir+f'mesh_{i}.su2', 
                    self.test_dir+f'sens_{i}.vtk', 
                    self.test_dir+f'flow_{i}.vtk', self.markers, self.data_fields)

                # Sample internal nodes
                if self.num_nodes != -1:
                    n_bounds = sum(b!=0)
                    torch.manual_seed(42)
                    idx = torch.cat([(b!=0).nonzero().flatten(),
                        (b==0).nonzero().flatten()[
                            torch.randperm(sum(b==0))[:self.num_nodes-n_bounds]]],dim=0)
                    p = p[idx,:]; b = b[idx]; v = v[idx,:]

                pos = p
                b1hot = F.one_hot(b.long()).float()
                um = v[:,1:].norm(dim=-1).mean()
                v[:,:1] /= um**2
                v[:,1:] /= um
                sens = torch.zeros(p.shape[0],3)
                sens[b==1,:] = s #sens calc on first marker
                v = torch.cat([v,sens], dim=-1)

                # Generate graph
                if self.knn:
                    edge_index = knn_graph(pos, k=self.rc)
                else:
                    edge_index = radius_graph(pos, r=self.rc, max_num_neighbors=32)
                neighbors.append(edge_index.shape[1]/pos.shape[0])

                data = Data(x=p-p.mean(dim=0), y=v.unsqueeze(0), irreps_io=self.irreps_io,
                            dts=torch.ones(1), emb_node=b1hot,
                            pos=pos, edge_index=edge_index, rc=self.rc)
                testset.append(data)
            print('Avg neighbors = ', sum(neighbors)/len(neighbors))
            self.test_data = testset

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)

    def loss_fn(self, y_hat, data, α=0.8):
        l1 = F.mse_loss(y_hat[:,:,:-3], data.y[:,:,:-3])
        l2 = F.mse_loss(y_hat[:,data.emb_node[:,1].bool(),-3:], 
                  data.y[:,data.emb_node[:,1].bool(),-3:])

        return (1-α)*l1 + α*l2