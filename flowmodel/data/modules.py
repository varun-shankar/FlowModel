import torch
from typing import Optional
import random
from torch_cluster import knn_graph, radius_graph
import glob, re, os
import torch.nn.functional as F
from torch_geometric.data import Data as pygData
from torch_geometric.data import Batch as pygBatch
from torch_geometric.utils import degree
from e3nn.math import soft_one_hot_linspace
from e3nn import o3, io
from torch_geometric.loader import DataLoader, NeighborSampler, RandomNodeSampler
from torch_geometric.loader import GraphSAINTRandomWalkSampler as RWSampler
import pytorch_lightning as pl

###### Data Class ###################################################################################

class Data(pygData):
    # def __cat_dim__(self, key, value, *args, **kwargs):
    #     if key == 'y':
    #         return 1
    #     else:
    #         return super().__cat_dim__(key, value, *args, **kwargs)

    def rotate(self, rot):
        irreps_in = o3.Irreps(self.irreps_io[0][0]).simplify()
        irreps_out = o3.Irreps(self.irreps_io[0][1]).simplify()
        D_in = irreps_in.D_from_matrix(rot).type_as(self.x)
        D_out = irreps_out.D_from_matrix(rot).type_as(self.x)
        self.x = self.x @ D_in.T
        self.pos = self.pos @ rot.type_as(self.x).T
        self.y = self.y @ D_out.T
        return self, D_out

    def embed(self, num_basis=10):
        edge_src, edge_dst = self.edge_index
        deg = degree(edge_dst, self.num_nodes).type_as(self.x)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        self.norm = deg_inv_sqrt[edge_src] * deg_inv_sqrt[edge_dst]
        self.norm = self.norm * self.edge_norm if 'edge_norm' in self else self.norm

        self.edge_vec = self.pos[edge_dst] - self.pos[edge_src]
        rc = float(self.rc.max()) if torch.is_tensor(self.rc) else self.rc
        self.emb = soft_one_hot_linspace(self.edge_vec.norm(dim=1), 0.0, rc, num_basis, 
            basis='fourier', cutoff=True).mul(num_basis**0.5)

def check_sampled_data(dataset):
    nn = 0; ne = 0
    for i in range(len(dataset)):
        nn += dataset[i].num_nodes
        ne += dataset[i].edge_index.size(1)
    print(f'Sampled data: {nn/len(dataset):.2f} nodes, {ne/len(dataset):.2f} edges')

class RWSampled_Dataset(RWSampler):
    def __init__(self, data, **kwargs):
        self.finished_loading = False
        super().__init__(data, **kwargs)
        self.finished_loading = True
        self.g = torch.Generator(); self.g.seed()
        _ = check_sampled_data(self) if int(os.environ.get('LOCAL_RANK', 0)) == 0 else 0
    def __getitem__(self, idx):
        if self.finished_loading:
            sample_ids = (self.data.batch==idx).nonzero().flatten()
            start = sample_ids[torch.randint(0, sample_ids.size(0), 
                (self.__batch_size__,), generator=self.g, dtype=torch.long)]
            n_id = self.adj.random_walk(start.flatten(), self.walk_length).view(-1)
            adj, _ = self.adj.saint_subgraph(n_id)
            data = RWSampler.__collate__(self, [(n_id, adj)])
            data.irreps_io = data.irreps_io[0]
            return data
        else:
            return RWSampler.__getitem__(self, idx)

class Cluster_Dataset(NeighborSampler):
    def __init__(self, data, batch_size=1, num_steps=1, **kwargs):
        self.data = data
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.N = data.num_nodes
        self.E = data.edge_index.size(1)
        super().__init__(data.edge_index, **kwargs)
        self.sample_coverage = 0
        self.g = torch.Generator(); self.g.seed()
        _ = check_sampled_data(self) if int(os.environ.get('LOCAL_RANK', 0)) == 0 else 0
    def __len__(self):
        return self.num_steps
    def __getitem__(self, idx):
        sample_ids = (self.data.batch==idx).nonzero().flatten()
        n_id = self.sample(sample_ids[torch.randint(0, sample_ids.size(0), 
            (self.batch_size,), generator=self.g, dtype=torch.long)])[1]
        adj, _ = self.adj_t.saint_subgraph(n_id)
        data = RWSampler.__collate__(self, [(n_id, adj)])
        data.irreps_io = data.irreps_io[0]
        return data

###### OpenFoam #####################################################################################

class OFDataModule(pl.LightningDataModule):
    def __init__(self, case, zones, ts, rc,
                       num_nodes=-1, sample_graph=False,
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
        self.sample_graph = sample_graph
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
        from .load_OF import load_OF
        if stage == 'fit':
            self.ts = self.ts if torch.is_tensor(self.ts) else torch.arange(self.ts[0],
                self.ts[1]+1e-8,step=self.dt)
            self.test_ts = self.test_ts if torch.is_tensor(self.test_ts) else torch.arange(self.test_ts[0],
                self.test_ts[1]+1e-8,step=self.dt)
            all_ts = torch.cat([self.ts,self.test_ts],dim=0)

            ### Read data ###
            p, b, v = load_OF(self.case, self.data_fields, all_ts, self.zones)

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
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print(f'Avg neighbors: {edge_index.shape[1]/pos.shape[0]:.2f}')

            ### Generate dataset ###
            dataset = [Data(x=features[i,:,:], emb_node=b1hot,
                            y=fields[i+1:i+1+self.rollout,:,:].transpose(0,1), 
                            irreps_io=self.irreps_io,
                            ts=(all_ts[i:i+1+self.rollout].unsqueeze(0)), 
                            pos=pos, edge_index=edge_index, rc=self.rc
                            ) for i in range(len(self.ts)-self.rollout)]
            if self.random_split:
                random.shuffle(dataset)
            self.train_data = dataset[:int((len(dataset)+1)*self.train_split)] 
            self.val_data = dataset[int((len(dataset)+1)*self.train_split):]

            if self.sample_graph != None:
                self.train_data = pygBatch.from_data_list(self.train_data)
                if self.sample_graph == 'random walk':
                    self.train_data = RWSampled_Dataset(self.train_data, 
                        batch_size=getattr(self,'seed_num',1000), num_steps=self.train_data.num_graphs,
                        walk_length=getattr(self,'hops',getattr(self,'latent_layers',10)), 
                        sample_coverage=getattr(self,'sample_coverage',0),
                        save_dir='.')
                elif self.sample_graph == 'cluster':
                    self.train_data = Cluster_Dataset(self.train_data, 
                        batch_size=getattr(self,'seed_num',1), num_steps=self.train_data.num_graphs,
                        sizes=-1*torch.ones(getattr(self,'hops',getattr(self,'latent_layers',10))))
                else:
                    print('Unknown sampling method')

            # Test
            testset = [Data(x=features[i,:,:], emb_node=b1hot,
                            y=fields[i+1:i+1+self.rollout,:,:].transpose(0,1), 
                            irreps_io=self.irreps_io,
                            ts=(all_ts[i:i+1+self.rollout].unsqueeze(0)), 
                            pos=pos, edge_index=edge_index, rc=self.rc
                            ) for i in range(
                                len(self.ts),len(self.ts)+len(self.test_ts)-self.rollout)]
            testset_rollout = [Data(x=features[i,:,:], emb_node=b1hot,
                                y=fields[i+1:i+1+self.test_rollout,:,:].transpose(0,1), 
                                irreps_io=self.irreps_io,
                                ts=(all_ts[i:i+1+self.test_rollout].unsqueeze(0)), 
                                pos=pos, edge_index=edge_index, rc=self.rc
                                ) for i in range(
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

    def loss_fn(self, y_hat, data):
        return torch.sum((y_hat - data.y.transpose(0,1))**2 * data.node_norm.unsqueeze(-1)) if \
            'node_norm' in data else torch.mean((y_hat - data.y.transpose(0,1))**2)


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
        from .load_kaggle import load_kaggle
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
                if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                    print(f'Avg neighbors: {edge_index.shape[1]/pos.shape[0]:.2f}')

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
        from .load_nek import load_nek
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
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print(f'Avg neighbors: {edge_index.shape[1]/pos.shape[0]:.2f}')

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
                       alpha=0.5,
                       train_split=0.9, random_split=True,
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
        self.alpha = alpha
        self.train_split = train_split
        self.random_split = random_split
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.irreps_io = self.data_irreps
        self.__dict__.update(kwargs)

    def setup(self, stage: Optional[str] = None):
        from .load_SU2 import load_SU2
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
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print(f'Avg neighbors: {sum(neighbors)/len(neighbors):.2f}')

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

    def loss_fn(self, y_hat, data):
        l1 = F.mse_loss(y_hat[:,:,:-3], data.y[:,:,:-3])
        l2 = F.mse_loss(y_hat[:,data.emb_node[:,1].bool(),-3:], 
                  data.y[:,data.emb_node[:,1].bool(),-3:])

        return (1-self.alpha)*l1 + self.alpha*l2