import torch
import numpy as np
import fluidfoam as ff

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

def load_case(dir,fields=[],ts=[0.],bounds=['internal']):
    bounds.remove('internal') if 'internal' in bounds else False
    bounds.insert(0,'internal')
    p = []
    b_ind = []
    v = []

    i = 0
    for b in bounds:
        b = None if b == 'internal' else b
        pb, vb, numb = read_mesh_and_field(dir,b,fields,ts)
        p.append(pb)
        b_ind.append((i*torch.ones(numb)).long())
        v.append(vb)
        i+=1

    return p, b_ind, v