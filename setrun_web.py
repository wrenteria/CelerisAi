"""
setrun_web.py — Run a CelerisAI example using the SAME configuration layout as CelerisWEBGPU

This example demonstrates how to use an example folder obtained from CelerisWEBGPU
(https://plynett.github.io/) and run the same case with CelerisAI.

Why this matters:
- CelerisAI is designed to read the same example-folder structure used by CelerisWEBGPU.
- A user can first explore/tune a case interactively in the web app, then reuse that exact
  example folder locally with CelerisAI (no re-formatting of inputs).

How to get the example folders (CelerisWEBGPU repository):
- Download/copy an example folder from:
  https://github.com/plynett/plynett.github.io/tree/main/examples
- Keep the folder structure unchanged.

Expected local layout for this script:
- In the current directory, you should have an `examples/` folder.
- Inside `examples/`, you should have one or more case folders (e.g., `Balboa/`).

Example:
./examples/Balboa/   <-- contains topo + boundary + config files used by CelerisWEBGPU

In THIS repository / directory:
- The folder is assumed to be already downloaded and located at `./examples/`.
- If you placed it elsewhere, just change EXAMPLES_DIR and/or CASE below.

Tip:
- Read the comments on each step (topography, boundary conditions, domain, solver, run)
  to understand how the CelerisWEBGPU configuration is mapped into CelerisAI objects.
  
"""
import taichi as ti
from celeris.domain import Topodata, BoundaryConditions,Domain
from celeris.solver import Solver
from celeris.runner import Evolve
import time

# -----------------------------------------------------------------------------
# User settings (match your downloaded CelerisWEBGPU example folder)
# -----------------------------------------------------------------------------
EXAMPLES_DIR = "./examples"   # folder that contains downloaded CelerisWEBGPU examples
CASE = "Balboa"               # example case folder name inside EXAMPLES_DIR
CASE_PATH = f"{EXAMPLES_DIR}/{CASE}"

# Taichi execution backend:
# - ti.gpu is typically best on machines with CUDA/Metal/Vulkan working properly
# - fall back to ti.cpu if you are debugging or GPU is not available
ti.init(arch = ti.gpu)

# Numeric precision used internally by the solver:
# - ti.f32 is recommended (stable + fast)
precision =ti.f32 # ti.f16 for half-precision

# 1) Set the topography data
# The 'datatype="celeris"' tells CelerisAI to interpret files in the same way the
# CelerisWEBGPU examples are organized.
baty = Topodata(datatype='celeris',path=CASE_PATH)

# 2) Set Boundary conditions
# celeris=True: read the config.json file from CelerisWEBGPU
bc = BoundaryConditions(celeris=True,path='./examples/Balboa',precision=precision)

# 3) Build Numerical Domain
d = Domain(topodata=baty,precision=precision)

# 4) Solve using SWE BOUSS
# - 'SWE'   : depth-averaged shallow-water equations
# - 'Bouss' : enhanced Boussinesq-type model (weakly dispersive, nonlinear)
solver = Solver(domain=d, boundary_conditions=bc)
solver.model ='Bouss'

# 5) Execution
# saveimg=True , will save plots at ./plots directory
run = Evolve(solver = solver, maxsteps= 10000, saveimg=True)

# Visualization:
# variable options depend on what the solver exposes; common ones are:
# - 'h'   : total water depth
# - 'eta' : free-surface elevation
# - 'vor' : vorticity-like diagnostic (if enabled in your build)
#
# cmapWater can be any valid Matplotlib colormap name.
run.Evolve_Display(variable='h',cmapWater='Blues')
#run.Evolve_Display(variable='eta',vmin=-5,vmax=5,cmapWater='seismic')
#run.Evolve_Display(variable='vor',vmin=-0.5,vmax=0.5,cmapWater='jet')

# Headless execution (No Visualization) is typically faster:
# run.Evolve_Headless()

