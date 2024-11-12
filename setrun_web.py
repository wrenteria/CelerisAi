import taichi as ti
from celeris.domain import Topodata, BoundaryConditions,Domain
from celeris.solver import Solver
from celeris.runner import Evolve
import time

ti.init(arch = ti.gpu, advanced_optimization = True, kernel_profiler = False)

# 1) Set the topography data
baty = Topodata(datatype='celeris',path='./examples/Balboa') 

# 2) Set Boundary conditions
bc = BoundaryConditions(celeris=True,path='./examples/Balboa')

# 3) Build Numerical Domain 
d = Domain(topodata=baty)

# 4) Solve using SWE BOUSS
solver = Solver(domain=d, boundary_conditions=bc)

# 5) Execution
# print("Cold start")
# run = Evolve(solver = solver, maxsteps= 100)
# run = None
# time.sleep(1)
# print("Warm start")
run = Evolve(solver = solver, maxsteps= 10000, saveimg=True)

# run.Evolve_Headless() # Faster , no visualization

# Showing -> 'h'    # cmap any on matplotlib Library e.g. 'BuGn', 'Blues'
run.Evolve_Display(cmapWater='Blues')

