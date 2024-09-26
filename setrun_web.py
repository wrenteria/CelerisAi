import taichi as ti
from celeris.domain import Topodata, BoundaryConditions,Domain
from celeris.solver import Solver
from celeris.runner import Evolve
import time

ti.init(arch = ti.cuda)

# 1) Set the topography data
baty = Topodata(datatype='celeris',path='./examples/Balboa') 

# 2) Set Boundary conditions
bc = BoundaryConditions(celeris=True,path='./examples/Balboa')

# 3) Build Numerical Domain 
d = Domain(topodata=baty)

# 4) Solve using SWE BOUSS
solver = Solver(domain=d, boundary_conditions=bc)

# 5) Execution
run = Evolve(solver = solver, maxsteps= 10000)

# run.Evolve_Headless() # Faster , no visualization

# Showing -> 'h'    # cmap any on matplotlib Library e.g. 'BuGn', 'Blues'
run.Evolve_Display(cmapWater='Blues')

