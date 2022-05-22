import SU2
import copy, glob, os, sys, time
import pandas as pd
import geometry
import torch
import meshio
import shutil
import numpy as np
from dragonfly import minimise_function

config_filename = 'adjoint.cfg'

SU2_config = SU2.io.Config(config_filename)
SU2_config["HISTORY_OUTPUT"].append("AERO_COEFF")
state = SU2.io.State()

SU2_config.NUMBER_PART = 1
SU2_config.NZONES = 1

state.find_files(SU2_config)

SU2_config.OPT_OBJECTIVE= {
        'DRAG': {
            'SCALE': 1,
            'OBJTYPE': 'DEFAULT',
            'VALUE': 0,
            'MARKER': 'front, bump, back'
        },
    }

def run_direct(SU2_config):
    
    konfig = copy.deepcopy(SU2_config)

    state = SU2.io.State()
    state.find_files(konfig)

    infod = SU2.run.direct(konfig)
    state.update(infod)

    konfig['MESH_FILENAME'] = 'mesh.su2'
    #Get Functions
    drag = SU2.eval.func('DRAG',konfig, state)
    
    return state, drag

rc = 0.5
num_nodes = 64
bump_rad = 0.2

## experiment with these initial conditions
# bump_center = torch.Tensor([5.5])
bump_center = torch.FloatTensor(1).uniform_(3, 5)

def loss(dv):
    geometry.build_shape(dv,bump_rad,num_nodes,.3)
    state, drag = run_direct(SU2_config)
    return -drag

domain = [[3.0, 5.0]]
maxiters = 2 # 100

# TODO - get time after each iteration

tic = time.perf_counter()
loss_min, dv_min, history = minimise_function(loss, domain, maxiters)
tt = tic - time.perf_counter()

print("minimum loss", loss_min)
print("bump center", dv_min)
print("time", tt)

np.save("hist", history)

print("cya")
#
