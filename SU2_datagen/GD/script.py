import SU2
import copy, glob, os, sys, time
import pandas as pd
import geometry
import torch
import meshio
import shutil

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
    adjoint_data = mesh.point_data['Sensitivity'][:,0:2]

    return adjoint_data

rc = 0.5
num_nodes = 64
learning_rate = 0.3 # play with this
num_iters = 5

# logging
dvs   = [] # design variable
drags = []
grads = []
times = [] # rolling wall time

## experiment with these initial conditions
# bump_center = torch.Tensor([5.5])
bump_center = torch.FloatTensor(1).uniform_(3, 5)
bc1 = torch.Tensor([3.0])
bc2 = torch.Tensor([4.0])
bc3 = torch.Tensor([5.0])

bump_rad = 0.2

dv = bc1.clone().detach().requires_grad_(True)

# iterate over learning rate

tic = time.perf_counter()
for i in range(num_iters):
    it = i + 1
    print("#======================================================#")
    print("Iteration ", it)
    print("#======================================================#")

    print("#================================#")
    print("Building Geometry ")
    print("#================================#")
    geometry.build_shape(dv,bump_rad,num_nodes,.3)

    print("#================================#")
    print("SU2 Forward Solve ")
    print("#================================#")
    state, drag = run_direct(SU2_config)

    print("#================================#")
    print("SU2 Adjoint Solve ")
    print("#================================#")
    adjoint_data = torch.tensor(run_adjoint(SU2_config, state))

    print("#================================#")
    print("Updating Bump Center ")
    print("#================================#")
    pos = geometry.get_bump_pos(dv,bump_rad,num_nodes)
    pos.backward(adjoint_data.clone().detach())

    with torch.no_grad():
        dv -= dv.grad * learning_rate

    # append logs
    drags.append(drag)
    dvs.append(dv.item())
    tt = time.perf_counter() - tic
    times.append(tt)
    print("#======================================================#")
    print("Completed iteration:", it)
    print('Bump Center:', dv.item())
    print('Gradient: ', dv.grad.item())
    print('Drag:', drag)
    print('Wall Time:', tt)
    print("#======================================================#")

    with torch.no_grad():
        dv.grad.zero_()

    ##shutil.move('mesh.su2',f'test_data/mesh_{i}.su2')
    #shutil.move('surface_sens.vtk',f'test_data/sens_{i}.vtk')

print(torch.tensor([drags,dvs,times]).T)
