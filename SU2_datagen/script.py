import SU2
import copy, glob, os, sys, time
import pandas as pd
import geometry
import torch
import meshio
import shutil
sys.path.append('/home/opc/data/ml-cfd/FlowModel')
# from model_def import LitModel
# import flowmodel.data.load_SU2 as load
# from flowmodel.data.modules import Data
import torch.nn.functional as F
from torch_cluster import knn_graph, radius_graph
import numpy as np
from skopt import gp_minimize

# load_id = None#'3vfhiraw'
# if load_id is None:
#     ckpt = max(glob.glob('../checkpoints/*'), key=os.path.getctime)
# else:
#     ckpt = max(glob.glob('../checkpoints/run-'+load_id+'*'), key=os.path.getctime)
# print('Loading '+ckpt+' ...')
# model = LitModel.load_from_checkpoint(ckpt).to('cuda:0')
# def gen_data(rc, num_nodes, meshfile):
#     p, b = load.load_SU2_test(meshfile, ['bump','platform','inlet','outlet','top','ground'])
#     # Sample internal nodes
#     if num_nodes != -1:
#         n_bounds = sum(b!=0)
#         torch.manual_seed(42)
#         idx = torch.cat([(b!=0).nonzero().flatten(),
#             (b==0).nonzero().flatten()[
#                 torch.randperm(sum(b==0))[:num_nodes-n_bounds]]],dim=0)
#         p = p[idx,:]; b = b[idx]

#     pos = p
#     b1hot = F.one_hot(b.long()).float()
#     edge_index = radius_graph(pos, r=rc, max_num_neighbors=32)

#     data = Data(x=p-p.mean(dim=0), y=torch.zeros(1,1,1), irreps_io=[['1o','0e+2x1o']],
#                 dts=torch.ones(1), emb_node=b1hot,
#                 pos=pos, edge_index=edge_index, rc=rc)
#     return data.to('cuda:0')


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

rc = 0.7
num_nodes = 2000
learning_rate = 0.2
num_iters = 15
inner_iters = 9
drags = []
dvs = []
times = []
bump_center = torch.Tensor([5])
# bump_center = torch.FloatTensor(1).uniform_(3, 5)
bump_rad = 0.2
dv = bump_center.clone().detach().requires_grad_(True)


def loss(dv):
    geometry.build_shape(dv[0],bump_rad,64,.3)
    state, drag = run_direct(SU2_config)
    return drag

def cb(res):
    drags.append(res.func_vals[-1])
    dvs.append(res.x_iters[-1][0])
    times.append(time.perf_counter()-tic)

tic = time.perf_counter()
res = gp_minimize(loss,[(3.0,5.0)],n_calls=num_iters,n_initial_points=1,
    x0=[dv.detach().numpy()],callback=cb)
print(res)

# tic = time.perf_counter()
# for i in range(num_iters):
#     geometry.build_shape(dv,bump_rad,64,.3)

#     lr = learning_rate
#     drag = 0
#     if i<inner_iters:
#         data = gen_data(rc, num_nodes, 'mesh.su2')
#         with torch.no_grad():
#             adjoint_data = model(data)[0,data.emb_node[:,1].bool(),-3:-1].cpu()
#         torch.cuda.empty_cache()
#     else:
#         state, drag = run_direct(SU2_config)
#         shutil.move('flow.vtk',f'test_data/flow_{i}.vtk')
#         adjoint_data = torch.tensor(run_adjoint(SU2_config, state))
#         lr = 1

#     pos = geometry.get_bump_pos(dv,bump_rad,64)
#     pos.backward(adjoint_data.clone().detach())
#     print('grad: ', dv.grad.item())
#     drags.append(drag)
#     dvs.append(dv.item())
#     times.append(time.perf_counter()-tic)
    
#     with torch.no_grad():
#         dv -= dv.grad * lr
#         dv.grad.zero_()
#     print('dv  : ', dv.item())

#     shutil.move('mesh.su2',f'test_data/mesh_{i}.su2')
#     # shutil.move('surface_sens.vtk',f'test_data/sens_{i}.vtk')

# tic = time.perf_counter()
# for i in range(num_iters):
#     times.append(time.perf_counter()-tic)
#     geometry.build_shape(dv,bump_rad,64,.3)
#     data = gen_data(rc, num_nodes, 'mesh.su2')
#     with torch.no_grad():
#         adjoint_data = model(data)[0,data.emb_node[:,1].bool(),-3:-1].cpu()
#     torch.cuda.empty_cache()

#     lr = learning_rate
#     drag = 0
#     if i%inner_iters==0 or i==num_iters-1:
#         state, drag = run_direct(SU2_config)
#         shutil.move('flow.vtk',f'test_data/flow_{i}.vtk')
#         if (i>0 and i!=num_iters-1 and drag>0.9*drags[-inner_iters]):
#             adjoint_data = torch.tensor(run_adjoint(SU2_config, state))
#             lr = 0.5; learning_rate *= 0.5
#     if i==num_iters-2:
#         adjoint_data = torch.tensor(run_adjoint(SU2_config, state))
#         lr = 0.2

#     pos = geometry.get_bump_pos(dv,bump_rad,64)
#     pos.backward(adjoint_data.clone().detach())
#     print('grad: ', dv.grad.item())
#     drags.append(drag)
#     dvs.append(dv.item())
    
#     with torch.no_grad():
#         dv -= dv.grad * lr
#         dv.grad.zero_()
#     print('dv  : ', dv.item())

#     shutil.move('mesh.su2',f'test_data/mesh_{i}.su2')
#     # shutil.move('surface_sens.vtk',f'test_data/sens_{i}.vtk')

out = torch.tensor([drags,dvs,times]).T
print(out)
np.savetxt("out.csv", out, delimiter=",")