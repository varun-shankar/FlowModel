import torch
import numpy as np
import fluidfoam as ff
import sys, traceback
import math

class Suppressor(object):
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = self
    def __exit__(self, type, value, traceback):
        sys.stdout = self.stdout
        if type is not None:
            raise
    def write(self, x): pass

def read_field_snap(dir,t,f,b,numpts):
    tstr = f'{t:g}'
    v = torch.tensor(ff.readfield(dir,tstr,f,boundary=b)).float()
    v = v.t() if v.dim() == 2 else v.unsqueeze(-1)
    v = v.repeat(numpts,1) if v.shape[0] == 1 else v
    return v

def read_mesh_and_field(dir,b,fields,ts):
    p = torch.tensor(np.array(ff.readmesh(dir,boundary=b))).t().float()
    numpts = p.shape[0]
    v = torch.stack([torch.cat([read_field_snap(dir,t,f,b,numpts) for f in fields],dim=-1) for t in ts])
    return p, v, numpts

def load_case(dir,fields=[],ts=[0.],bounds=['internal'],verbose=False):
    bounds.remove('internal') if 'internal' in bounds else False
    bounds.insert(0,'internal')
    p = []
    b_ind = []
    v = []

    i = 0
    for b in bounds:
        b = None if b == 'internal' else b
        if verbose:
            pb, vb, numb = read_mesh_and_field(dir,b,fields,ts)
        else:
            with Suppressor():
                pb, vb, numb = read_mesh_and_field(dir,b,fields,ts)
        p.append(pb)
        b_ind.append((i*torch.ones(numb)).long())
        v.append(vb)
        i+=1

    return p, b_ind, v

def load_kaggle(dir,model,case,fields=[],label=[]):
    p = torch.stack([torch.Tensor(np.load(dir+model+'/'+model+'_'+case+'_Cx.npy')),
                     torch.Tensor(np.load(dir+model+'/'+model+'_'+case+'_Cy.npy')),
                     torch.Tensor(np.load(dir+model+'/'+model+'_'+case+'_Cz.npy'))],
                     dim=-1)

    v = []
    for f in fields:
        if f =='U':
            vf = torch.stack([torch.Tensor(np.load(dir+model+'/'+model+'_'+case+'_'+f+'x.npy')),
                              torch.Tensor(np.load(dir+model+'/'+model+'_'+case+'_'+f+'y.npy')),
                              torch.Tensor(np.load(dir+model+'/'+model+'_'+case+'_'+f+'z.npy'))],
                              dim=-1)
            # vf = torch.cat([vf.norm(dim=1,keepdim=True),vf],dim=-1)
        else:
            vf = torch.Tensor(np.load(dir+model+'/'+model+'_'+case+'_'+f+'.npy')).unsqueeze(-1)
        v.append(vf)
    v = torch.cat(v, dim=-1)

    l = torch.Tensor(np.load(dir+'labels/'+case+'_'+label+'.npy'))

    return p, v, l


def load_nek(dir, step=4, tsteps=100, num_nodes=5000):
    nx= 81; ny= 41; nz= 61
    # x,y,z = np.mgrid[0:nx:step,0:ny:step,0:nz:step]
    # inds = np.ravel_multi_index([x.flatten(),y.flatten(),z.flatten()],(nx,ny,nz),order='F')

    # xlim = 2*math.pi; ylim = 1; zlim = math.pi
    # xs = np.linspace(0,xlim,num=x.shape[0])
    # ys = np.linspace(0,ylim,num=x.shape[1])
    # zs = np.linspace(0,zlim,num=x.shape[2])

    pos = []
    pos_inds = torch.randperm(nx*ny*nz)[:num_nodes]+1#[]inds+1
    v = []
    ts = []

    meshfile = dir+'xyz000001.dat'
    with open(meshfile) as fp:
        for i,line in enumerate(fp):
            # if i>0:
            #     input = np.asarray(line.split(), dtype=float)[1:4]
            #     if (any(np.isclose(xs,input[0],rtol=0,atol=1e-16)) and 
            #             any(np.isclose(ys,input[1],rtol=0,atol=1e-16)) and 
            #             any(np.isclose(zs,input[2],rtol=0,atol=1e-16))):
            #         pos.append(torch.Tensor(input))
            #         pos_inds.append(i)
            if i in pos_inds:
                pos.append(torch.Tensor(np.asarray(line.split(), dtype=float))[1:4])

    for t in range(tsteps):
        vt = []
        timefile = dir+f'uvw{t+2:06d}.dat'
        with open(timefile) as fp:
            for i,line in enumerate(fp):
                if i in pos_inds:
                    input = torch.Tensor(np.asarray(line.split(), dtype=float))
                    vt.append(input[[4,1,2,3]])
                    tval = input[0]
        v.append(torch.stack(vt))
        ts.append(tval)

    pos = torch.stack(pos)
    v = torch.stack(v)
    ts = torch.stack(ts)
    torch.save((pos,v,ts),dir+'nekData.pt')
    return pos, v, ts

