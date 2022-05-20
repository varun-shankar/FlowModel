import torch
import numpy as np
import meshio

def read_mesh_file(file: str):

    if file.endswith('su2'):
        with open(file,'r') as fl:
            lines = fl.readlines()
        ndime = int(lines[0].split('=')[1])
        nelements = int(lines[1].split('=')[1])
        elements = np.loadtxt(file,skiprows=2,max_rows=nelements)
        
        npoints = int(lines[2+nelements].split('=')[1])
        points = np.loadtxt(file,skiprows=3+nelements,max_rows=npoints)
        
        data = {
            'dimension': ndime,
            'nelements': nelements,
            'elements': elements,
            'npoints': npoints,
            'points': points,
        }
        nmarkers = int(lines[3+nelements+npoints].split('=')[1])
        markers = {}
        start_ind = 4+nelements+npoints
        for i in range(nmarkers):
            name = str(lines[start_ind].split('=')[1]).strip()
            nelements = int(lines[start_ind+1].split('=')[1])
            elements = np.loadtxt(file,skiprows=start_ind+2,max_rows=nelements)
            markers[name] = {
                'nelements': nelements,
                'elements': elements
            }
            start_ind += nelements + 2

        data['markers'] = markers

    return data

def get_marker_info_from_file(file, marker):
    data = read_mesh_file(file)

    marker_data = data['markers'][marker]

    points = np.unique(marker_data['elements'][:,1:].flatten()).astype(int)
    connectivity = marker_data['elements'][:,1:].astype(int)

    output = {
        'id': points,
        'x': data['points'][points,0],
        'y': data['points'][points,1],
        'connectivity': connectivity
    }
    if data['dimension'] == 3:
        output['z'] = data['points'][points,2]

    return output

def load_SU2(meshfile, sensfile, flowfile, markers, fields):
    mesh = meshio.read(flowfile)
    p = torch.tensor(mesh.points).float()
    b = torch.zeros(p.shape[0])

    i = len(markers)
    for m in reversed(markers):
        b[get_marker_info_from_file(meshfile,m)['id']] = i
        i -= 1
    
    v = []
    for f in fields:
        vf = torch.tensor(mesh.point_data[f]).float()
        v.append(vf)
    v = torch.cat(v, dim=-1)

    sens = meshio.read(sensfile)
    s = torch.tensor(sens.point_data['Sensitivity']).float()

    return p, b, v, s

def load_SU2_test(meshfile, markers):
    p = torch.tensor(read_mesh_file(meshfile)['points']).float()
    p[:,-1] = 0
    b = torch.zeros(p.shape[0])

    i = len(markers)
    for m in reversed(markers):
        b[get_marker_info_from_file(meshfile,m)['id']] = i
        i -= 1

    return p, b