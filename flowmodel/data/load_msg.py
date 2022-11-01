import numpy as np
import torch

def load_case(dir, Re, ts, data_fields):
    lx = 8
    ly = 1
    nx = 512
    ny = int(nx/8)
    x = np.linspace(0.0,lx,nx+1)
    y = np.linspace(0.0,ly,ny+1)
    X, Y = np.meshgrid(x, y, indexing='ij')
    X, Y = X[::2,::2], Y[::2,::2]
    p = torch.stack([torch.tensor(X.flatten()),torch.tensor(Y.flatten()),torch.tensor(0*X.flatten())],dim=-1)

    b = X*0; b[:,[0,-1]]=1; b[[0,-1],:]=1
    b = torch.tensor(b.flatten()).long()

    v = []
    for i in ts:
        out = np.load(dir+'Re_'+str(Re)+'/data_512_64/data_'+str(i.item())+'.npz')
        v.append(torch.stack([torch.tensor(out[df][::2,::2].flatten()) for df in data_fields],dim=-1))
    
    v = torch.stack(v)
    return p.float(), b, v.float()