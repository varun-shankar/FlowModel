import SU2
import copy, glob, os, sys, time
import pandas as pd
import numpy as np
import geometry3
import torch
import meshio
import shutil

config_filename = 'adjoint.cfg'

SU2_config = SU2.io.Config(config_filename)
SU2_config["HISTORY_OUTPUT"].append("AERO_COEFF")
state = SU2.io.State()

SU2_config.NUMBER_PART = int(sys.argv[1])#32 # nthreads
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
    
    SU2_config.NUMBER_PART = int(sys.argv[1])
    konfig = copy.deepcopy(SU2_config)

    state = SU2.io.State()
    state.find_files(konfig)

    infod = SU2.run.direct(konfig)
    state.update(infod)

    konfig['MESH_FILENAME'] = 'mesh.su2'
    #Get Functions
    drag = SU2.eval.func('DRAG',konfig, state)
    
    return state, drag

def run_adjoint(SU2_config, state):

    SU2_config.NUMBER_PART = 1
    # SU2_config['MATH_PROBLEM'] = 'DISCRETE_ADJOINT'
    SU2_config['GRADIENT_METHOD'] = 'DISCRETE_ADJOINT'
    konfig = copy.deepcopy(SU2_config)
    SU2.io.restart2solution(konfig,state)
    konfig['MATH_PROBLEM'] = 'DISCRETE_ADJOINT'
    konfig['GRADIENT_METHOD'] = 'DISCRETE_ADJOINT'
    infoa = SU2.run.adjoint(konfig)
    
    konfig = copy.deepcopy(SU2_config)
    konfig['MATH_PROBLEM'] = 'DISCRETE_ADJOINT'
    # konfig = copy.deepcopy(SU2_config)
    SU2.io.restart2solution(konfig,state)
    infop = SU2.run.projection(konfig)

    filename = 'surface_sens.vtk'
    mesh = meshio.read(filename)
    adjoint_data = mesh.point_data['Sensitivity'][:,0:3]

    return adjoint_data

learning_rate = 2 # play with this
num_iters = 10

# logging
bcs   = [] # bump_center
grads = []
drags = []
times = [] # wall time

## experiment with these initial conditions
# bump center range = 3.2 -- 4.8
xcenter = 4.0
lr = learning_rate

for i in range(num_iters):
    it = i + 1
    tic = time.perf_counter()
    print("#======================================================#")
    print("Iteration ", it)
    print("#======================================================#")

    print("#================================#")
    print("Building Geometry ")
    print("#================================#")
    bump_center = np.array([xcenter, 0])
    geometry3.build_shape(bump_center)

    print("#================================#")
    print("SU2 Forward Solve ")
    print("#================================#")
    state, drag = run_direct(SU2_config)

    print("#================================#")
    print("SU2 Adjoint Solve ")
    print("#================================#")
    adjoint_data = torch.tensor(run_adjoint(SU2_config, state))
    grad = np.array(sum(adjoint_data))

    tt = time.perf_counter() - tic

    print("#======================================================#")
    print("Completed iteration:", it)
    print('Bump Center:', bump_center)
    print('Gradient: ', grad)
    print('Drag:', drag)
    print('Wall Time:', tt)
    print("#======================================================#")

    xcenter -= lr * grad[0]

    # append logs
    bcs.append(bump_center)
    drags.append(drag)
    times.append(tt)

    ##shutil.move('mesh.su2',f'test_data/mesh_{i}.su2')
    #shutil.move('surface_sens.vtk',f'test_data/sens_{i}.vtk')

# save logs
name = "gd_log_lr_" + str(lr)
np.save(name, [bcs, drags, times])
#
