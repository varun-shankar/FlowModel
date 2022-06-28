import torch
import numpy as np

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