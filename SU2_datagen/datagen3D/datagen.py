import SU2
import copy, glob, os, sys, time
import geometry3
import torch
import meshio
import shutil
import pandas as pd
import numpy as np

ncases = 200

config_filename = 'adjoint.cfg'

SU2_config = SU2.io.Config(config_filename)
SU2_config["HISTORY_OUTPUT"].append("AERO_COEFF")
state = SU2.io.State()

#SU2_config.NUMBER_PART = 32 # nthreads
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

xcenters = np.linspace(3.2, 4.8,ncases)

drags = []
times = []
grads = []

for i in range(ncases):
    tic = time.perf_counter()
    print("#======================================================#")
    print("Case ", i+1)
    print("#======================================================#")

    print("#================================#")
    print("Building Geometry ")
    print("#================================#")
    bump_center = [xcenters[i],0]
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

    drags.append(drag)
    times.append(tt)
    grads.append(grad)

    print("#======================================================#")
    print("Case:", i+1)
    print('Bump Center:', bump_center)
    print('Drag:', drag)
    print('grads:', grads)
    print('Time taken:', tt)
    print("#======================================================#")

    shutil.move('mesh.su2',f'data/mesh_{i}.su2')
    shutil.move('surface_sens.vtk',f'data/sens_{i}.vtk')

name = "datagen_log"
np.save(name, [xcenters, drags, grads, times])
#
