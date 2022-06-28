import numpy as np
import os, sys
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import torch
sys.path.append('/home/opc/data/ml-cfd/FlowModel')
from flowmodel.data.modules import Data
from scipy.interpolate import griddata
from e3nn import o3, io
from scipy.ndimage import gaussian_filter

data, pred = torch.load('data.pt', map_location=torch.device('cpu'))
print(torch.nn.functional.mse_loss(data.y,pred).item())

label_irrep = io.CartesianTensor('ij=ji')
print(torch.nn.functional.mse_loss(label_irrep.to_cartesian(data.y),label_irrep.to_cartesian(pred)).item())

pos = data.pos[:,0:2]
x = np.sum(data.x.detach().numpy()[:,-3:]**2,axis=-1)**.5
y = label_irrep.to_cartesian(data.y).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).detach().numpy()/2
p = label_irrep.to_cartesian(pred).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1).detach().numpy()/2

fig, axs = plt.subplots(4, figsize=(8,14))
ext = [pos[:,0].min().item(),pos[:,0].max().item(),
    pos[:,1].min().item(),pos[:,1].max().item()]
step = (ext[1] - ext[0])/1e3
grid_x, grid_y = np.mgrid[ext[0]:ext[1]:step, ext[2]:ext[3]:step]

u0 = np.flipud(griddata(pos, x, (grid_x, grid_y), method='cubic').T)
ut = np.flipud(griddata(pos, y[0,:], (grid_x, grid_y), method='cubic').T)
up = np.flipud(griddata(pos, p[0,:], (grid_x, grid_y), method='cubic').T)

up = gaussian_filter(up, sigma=7)

ims=[]
ims.append(axs[0].imshow(u0, extent=ext, vmin=0, vmax=x.max()))
ims.append(axs[1].imshow(ut, extent=ext, vmin=0, vmax=p.max()))
ims.append(axs[2].imshow(up, extent=ext, vmin=0, vmax=p.max()))
err = (p-y)**2
ims.append(axs[3].imshow((ut-up)**2, extent=ext, vmin=0, vmax=err.max()))

for i in range(4):
    fig.colorbar(ims[i], ax=axs[i])
    axs[i].xaxis.set_visible(False)
    axs[i].yaxis.set_visible(False)
axs[0].scatter(pos[:,0],pos[:,1], s=.1)

plt.tight_layout()
plt.savefig('plotting/kaggle.png')