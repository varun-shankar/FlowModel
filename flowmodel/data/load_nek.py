import torch
import numpy as np

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
