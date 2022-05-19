import SU2
import copy
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
    
    return state

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


lr = 1
iter = 0
for j in range(10):
    bump_center = torch.FloatTensor(1).uniform_(3, 5)
    bump_rad = 0.2
    dv = bump_center.clone().detach().requires_grad_(True)

    for i in range(1):
        geometry.build_shape(dv,bump_rad,64,.3)
        state = run_direct(SU2_config)
        adjoint_data = run_adjoint(SU2_config, state)
        pos = geometry.get_bump_pos(dv,bump_rad,64)
        pos.backward(torch.tensor(adjoint_data))
        print(dv.grad)
        with torch.no_grad():
            dv -= dv.grad * lr
            dv.grad.zero_()
        shutil.move('flow.vtk',f'test_data/flow_{iter}.vtk')
        shutil.move('surface_sens.vtk',f'test_data/sens_{iter}.vtk')
        shutil.move('mesh.su2',f'test_data/mesh_{iter}.su2')
        iter += 1
