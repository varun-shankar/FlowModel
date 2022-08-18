import gmsh
import math
from random import random
import numpy as np
import os
import torch

def build_shape(bump_center):
    # range 3.2 - 4.8, -.8 - .8
    cmd = f'sed -i \'1c\cx = {bump_center[0]}; cz = {bump_center[1]};\' car.geo'
    os.system(cmd)

    gmsh.initialize()
    gmsh.open('car.geo')

    # gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.model.mesh.generate(3)
    gmsh.write('mesh.su2')
    
    gmsh.finalize()

    return

build_shape([4,0])