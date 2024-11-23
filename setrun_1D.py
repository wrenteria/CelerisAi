import taichi as ti
from celeris.domain import Topodata, BoundaryConditions,Domain
from celeris.solver import Solver
from celeris.runner import Evolve
import argparse
import time
import numpy as np

ti.init(arch = ti.cpu)
precision =ti.f32

baty = Topodata(filename='Topo1D.txt',path='./examples/1D',datatype='xz')
x1 = 0
x2 = 480
print(baty.z())
bc = BoundaryConditions(West=2,celeris=False,path='./examples/1D',filename='irrWaves1D.txt')
d = Domain(topodata=baty,x1=x1,x2=x2,Nx=480)
