import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import torch
from flowmodel.data.modules import Data
from scipy.interpolate import griddata

data, pred = torch.load('data_rollout.pt', map_location=torch.device('cpu'))
data.y = data.y.transpose(0,1)
print(torch.nn.functional.mse_loss(data.y,pred).item())
inds = torch.isclose(data.pos[:,2],data.pos[:,2].mean(),rtol=1e-2)
pos = data.pos[inds,:]
pos = pos[:,0:2]

x = np.sum(data.x.detach().numpy()[inds,-1:]**2,axis=-1)**.5
y = np.sum(data.y.detach().numpy()[:,inds,-1:]**2,axis=-1)**.5
p = np.sum(pred.detach().numpy()[:,inds,-1:]**2,axis=-1)**.5

fig, axs = plt.subplots(5, figsize=(19,18))
ext = [pos[:,0].min().item(),pos[:,0].max().item(),
    pos[:,1].min().item(),pos[:,1].max().item()]
step = (ext[1] - ext[0])/1e3
grid_x, grid_y = np.mgrid[ext[0]:ext[1]:step, ext[2]:ext[3]:step]
ims = [axs[i].imshow(np.random.random(grid_x.T.shape), extent=ext, vmin=0, vmax=y.max()) for i in range(4)]
ims.append(axs[4].imshow(np.random.random(grid_x.T.shape), extent=ext, vmin=0, vmax=((y-p)**2).max(), cmap='Greys_r'))
for i in range(5):
    fig.colorbar(ims[i], ax=axs[i])
axs[0].scatter(pos[:,0],pos[:,1],s=.5,c='k')
ims.append(axs[3].scatter(pos[:,0],pos[:,1],s=.5,c='k'))

axs[0].set_ylabel(r'$u_0$')
axs[1].set_ylabel(r'$u$')
axs[2].set_ylabel(r'$\hat u$')
axs[3].set_ylabel(r'$\hat u$')
axs[4].set_ylabel(r'$||u-\hat u||$')

plt.tight_layout()

def animate(i):
    
    u0 = griddata(pos, x, (grid_x, grid_y), method='cubic').T
    ut = griddata(pos, y[i,:], (grid_x, grid_y), method='cubic').T
    up = griddata(pos, p[i,:], (grid_x, grid_y), method='cubic').T

    ims[0].set_array(u0)
    ims[1].set_array(ut)
    ims[2].set_array(up)
    ims[3].set_array(up)
    ims[4].set_array((ut-up)**2)
    mask = np.zeros(data.pos.shape[0]); mask[data.data1_idx[i]] = 1; mask = mask[inds]
    ims[5].set_color(np.tile(mask,(3,1)).T)
    return ims

anim = animation.FuncAnimation(fig, animate,
                               frames=pred.shape[0], interval=50, blit=True) 
anim.save('plotting/movie.gif', writer='ffmpeg', fps=24)