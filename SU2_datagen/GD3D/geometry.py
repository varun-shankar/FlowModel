import gmsh
import math
from random import random
import numpy as np
import torch

def build_shape(bump_center, bump_rad, npoints, lc):

    gmsh.initialize()
    gmsh.model.add('shape')
    # gmsh.open('shape.geo')

    gmsh.model.occ.addPoint(0, 0, 0, 1.0, 101)
    gmsh.model.occ.addPoint(0, 5, 0, 1.0, 102)
    gmsh.model.occ.addPoint(20, 5, 0, 1.0, 103)
    gmsh.model.occ.addPoint(20, 0, 0, 1.0, 104)

    gmsh.model.occ.addPoint(7, 0, 0, 1.0, 105)
    gmsh.model.occ.addPoint(6.75, 0, 0, 1.0, 106)
    gmsh.model.occ.addPoint(6.5, 0.25, 0, 1.0, 107)
    gmsh.model.occ.addPoint(6, 0.75, 0, 1.0, 108)
    gmsh.model.occ.addPoint(5.75, 1, 0, 1.0, 109)
    gmsh.model.occ.addPoint(5.5, 1, 0, 1.0, 110)

    gmsh.model.occ.addPoint(2.5, 1, 0, 1.0, 111)
    gmsh.model.occ.addPoint(2.25, 1, 0, 1.0, 112)
    gmsh.model.occ.addPoint(2, 0.75, 0, 1.0, 113)
    gmsh.model.occ.addPoint(1.5, 0.25, 0, 1.0, 114)
    gmsh.model.occ.addPoint(1.25, 0, 0, 1.0, 115)
    gmsh.model.occ.addPoint(1, 0, 0, 1.0, 116)
    
    gmsh.model.occ.addLine(101, 102, 1)
    gmsh.model.occ.addLine(102, 103, 2)
    gmsh.model.occ.addLine(103, 104, 3)
    gmsh.model.occ.addLine(104, 105, 4)
    gmsh.model.occ.addBezier([105, 106, 107], 5)
    gmsh.model.occ.addLine(107, 108, 6)
    gmsh.model.occ.addBezier([108, 109, 110], 7)
    gmsh.model.occ.addBezier([111, 112, 113], 8)
    gmsh.model.occ.addLine(113, 114, 9)
    gmsh.model.occ.addBezier([114, 115, 116], 10)
    gmsh.model.occ.addLine(116, 101, 11)
    
    points = []
    lines_bump = []
    for n in range(npoints):
        angle = math.pi*n/(npoints-1)
        x = bump_center+bump_rad*math.cos(angle)
        y = 1+bump_rad*math.sin(angle)
        points.append(gmsh.model.occ.addPoint(x,y,0))
    for i in range(1,npoints):
        lines_bump.append(gmsh.model.occ.addLine(points[i-1],points[i]))
    
    l1 = gmsh.model.occ.addLine(110, points[0])
    l2 = gmsh.model.occ.addLine(points[-1], 111)
    lines_platform = [5, 6, 7, l1, l2, 8, 9, 10]
    
    gmsh.model.occ.addCurveLoop([1, 2, 3, 4, 5, 6, 7, l1, *lines_bump, l2, 8, 9, 10, 11], 1)
    gmsh.model.occ.addPlaneSurface([1], 1)
    
    # loop = gmsh.model.occ.addCurveLoop(lines_bump)
    # gmsh.model.occ.addPlaneSurface([loop],2)
    # gmsh.model.occ.cut([(2,1)],[(2,2)])

    gmsh.model.occ.synchronize()
    domain = gmsh.model.addPhysicalGroup(2,[1])
    tag_left = gmsh.model.addPhysicalGroup(1,[1])
    gmsh.model.setPhysicalName(1, tag_left, "inlet")
    tag_top = gmsh.model.addPhysicalGroup(1,[2])
    gmsh.model.setPhysicalName(1, tag_top, "top")
    tag_right = gmsh.model.addPhysicalGroup(1,[3])
    gmsh.model.setPhysicalName(1, tag_right, "outlet")
    tag_ground = gmsh.model.addPhysicalGroup(1,[4, 11])
    gmsh.model.setPhysicalName(1, tag_ground, "ground")
    tag_platform = gmsh.model.addPhysicalGroup(1,lines_platform)
    gmsh.model.setPhysicalName(1, tag_platform, "platform")
    tag_bump = gmsh.model.addPhysicalGroup(1,lines_bump)
    gmsh.model.setPhysicalName(1, tag_bump, "bump")
    
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "EdgesList", [11,*lines_platform,*lines_bump])
    gmsh.model.mesh.field.setNumber(1, "NumPointsPerCurve", 100)
    gmsh.model.mesh.field.add("Distance", 2)
    gmsh.model.mesh.field.setNumbers(2, "EdgesList", [4])
    gmsh.model.mesh.field.setNumber(2, "NumPointsPerCurve", 100)

    gmsh.model.mesh.field.add("Threshold", 3)
    gmsh.model.mesh.field.setNumber(3, "IField", 1)
    gmsh.model.mesh.field.setNumber(3, "LcMin", lc / 10)
    gmsh.model.mesh.field.setNumber(3, "LcMax", lc)
    gmsh.model.mesh.field.setNumber(3, "DistMin", 0.05)
    gmsh.model.mesh.field.setNumber(3, "DistMax", 0.15)
    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "IField", 2)
    gmsh.model.mesh.field.setNumber(4, "LcMin", lc / 5)
    gmsh.model.mesh.field.setNumber(4, "LcMax", lc)
    gmsh.model.mesh.field.setNumber(4, "DistMin", 0.2)
    gmsh.model.mesh.field.setNumber(4, "DistMax", 0.5)

    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [3,4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    # gmsh.model.add('shape')

    # points = []
    # lines = []
    # points.append(gmsh.model.occ.addPoint(0,0,0))

    # points.append(gmsh.model.occ.addPoint(0,height,0))
    # line_left = gmsh.model.occ.addLine(points[1],points[0])

    # points.append(gmsh.model.occ.addPoint(length,height,0))
    # line_top = gmsh.model.occ.addLine(points[2],points[1])

    # points.append(gmsh.model.occ.addPoint(length,0,0))
    # line_right = gmsh.model.occ.addLine(points[3],points[2])

    # lines_back = []
    # points.append(gmsh.model.occ.addPoint(step_corner+step_length,0,0))
    # points.append(gmsh.model.occ.addPoint(step_corner+step_length,step_height,0))
    # points.append(gmsh.model.occ.addPoint(bump_center+bump_rad,step_height,0))
    # for i in range(4,len(points)):
    #     lines_back.append(gmsh.model.occ.addLine(points[i-1],points[i]))

    # lines_bump = []
    # for n in range(1,npoints):
    #     angle = math.pi*n/(npoints-1)
    #     x = bump_center+bump_rad*math.cos(angle)
    #     y = step_height+bump_rad*math.sin(angle)
    #     points.append(gmsh.model.occ.addPoint(x,y,0))
    # for i in range(len(points)-npoints+1,len(points)):
    #     lines_bump.append(gmsh.model.occ.addLine(points[i-1],points[i]))

    # lines_front = []
    # points.append(gmsh.model.occ.addPoint(step_corner,step_height,0))
    # points.append(gmsh.model.occ.addPoint(step_corner,0,0))
    # for i in range(len(points)-2,len(points)):
    #     lines_front.append(gmsh.model.occ.addLine(points[i-1],points[i]))
    # lines_front.append(gmsh.model.occ.addLine(points[-1],points[0]))

    # lines = [line_left]+[line_top]+[line_right]+lines_back+lines_bump+lines_front
    # loop = gmsh.model.occ.addCurveLoop(lines)
    # gmsh.model.occ.addPlaneSurface([loop],1)
    
    # gmsh.model.occ.synchronize()
    # tag_left = gmsh.model.addPhysicalGroup(1,[line_left])
    # gmsh.model.setPhysicalName(1, tag_left, "inlet")
    # tag_top = gmsh.model.addPhysicalGroup(1,[line_top])
    # gmsh.model.setPhysicalName(1, tag_top, "top")
    # tag_right = gmsh.model.addPhysicalGroup(1,[line_right])
    # gmsh.model.setPhysicalName(1, tag_right, "outlet")
    # tag_back = gmsh.model.addPhysicalGroup(1,lines_back)
    # gmsh.model.setPhysicalName(1, tag_back, "back")
    # tag_bump = gmsh.model.addPhysicalGroup(1,lines_bump)
    # gmsh.model.setPhysicalName(1, tag_bump, "bump")
    # tag_front = gmsh.model.addPhysicalGroup(1,lines_front)
    # gmsh.model.setPhysicalName(1, tag_front, "front")

    # gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.model.mesh.generate(2)
    gmsh.write('mesh.su2')
    
    gmsh.finalize()

    return

def get_bump_pos(bump_center, bump_rad, npoints):
    pos = torch.zeros(npoints,2)
    for n in range(npoints):
        angle = math.pi*n/(npoints-1)
        pos[n,0] = bump_center+bump_rad*math.cos(angle)
        pos[n,1] = 1+bump_rad*math.sin(angle)

    return pos
