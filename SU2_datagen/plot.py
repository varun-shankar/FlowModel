import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ml = pd.read_csv('out_ml.csv',header=None).replace(0,np.nan).to_numpy()
gd = pd.read_csv('out_gd.csv',header=None).to_numpy()
bo = pd.read_csv('out_bo.csv',header=None).to_numpy()
ml3 = pd.read_csv('out_ml_3.csv',header=None).replace(0,np.nan).to_numpy()
gd3 = pd.read_csv('out_gd_3.csv',header=None).to_numpy()
bo3 = pd.read_csv('out_bo_3.csv',header=None).to_numpy()

fig, ax = plt.subplots(2,1,figsize=(7,6))

ax[0].plot(ml[:,2],ml[:,1],'b-',label='ML+adjoint')
ax[0].plot(gd[:,2],gd[:,1],'b--',label='adjoint')
ax[0].plot(bo[:,2],bo[:,1],'b:',label='Bayesian')
ax[0].plot(ml3[:,2],ml3[:,1],'r-')
ax[0].plot(gd3[:,2],gd3[:,1],'r--')
ax[0].plot(bo3[:,2],bo3[:,1],'r:')
ax[0].set_ylabel('bump location')
ax[0].set_xlabel('time (s)')

# ax[1] = ax1.twinx()
ax[1].scatter(ml[:,2],ml[:,0],color='b',label='ML+adjoint')
ax[1].scatter(gd[:,2],gd[:,0],color='b',marker='*',label='adjoint')
ax[1].scatter(bo[:,2],bo[:,0],color='b',marker='+',label='Bayesian')
ax[1].scatter(ml3[:,2],ml3[:,0],color='r')
ax[1].scatter(gd3[:,2],gd3[:,0],color='r',marker='*')
ax[1].scatter(bo3[:,2],bo3[:,0],color='r',marker='+')
ax[1].set_ylabel('drag coefficient')

ax[0].legend(); ax[1].legend()
fig.tight_layout()
plt.savefig('fig.png')