"""
setrun_1D.py — Minimal 1D setup script for running CelerisAI

This example demonstrates how to configure and execute CelerisAI in 1D using:
- a 1D bathymetry/topography profile file (x–z format)
- a boundary forcing time series (incident waves at the west boundary)
- the CelerisAI Domain/Solver/Runner interface

Expected folder structure:
- This script assumes the example files are located in:
    ./examples/1D/

Required input files in ./examples/1D/:
1) Topo1D.txt
   - 1D bathymetry/topography profile
   - datatype='xz' means the file is interpreted as (x, z) pairs

2) irrWaves1D.txt
   - boundary forcing time series used at the WEST boundary

How to run:
    python setrun_1D.py

Notes for users:
- If you want to run on CPU instead of GPU, change: ti.init(arch=ti.cpu)
- If you change the physical length (x2-x1) or resolution (Nx), keep them consistent
  with your topo file extent and desired grid spacing.
- The model is set to 'Bouss' (enhanced Boussinesq-type) to include weak dispersion.
  For pure shallow water, use model='SWE' (if supported in your solver build).

Tip:
- For headless runs (no GUI), use run.Evolve_Headless() and disable display.
"""

import taichi as ti

from celeris.domain import Topodata, BoundaryConditions, Domain
from celeris.solver import Solver
from celeris.runner import Evolve


# -----------------------------------------------------------------------------
# 0) Taichi backend (GPU by default)
# -----------------------------------------------------------------------------
# Recommended:
# - ti.gpu: best performance if your GPU backend is correctly installed
# - ti.cpu: safer if you are debugging or running on a machine without GPU support
ti.init(arch=ti.gpu)


# -----------------------------------------------------------------------------
# 1) Input files (1D topo + boundary forcing)
# -----------------------------------------------------------------------------
EXAMPLE_DIR = "./examples/1D"

# 1D bathymetry/topography profile
# datatype='xz' means this is a 1D profile given as (x, z) pairs
baty = Topodata(
    filename="Topo1D.txt",
    path=EXAMPLE_DIR,
    datatype="xz"
)

# Boundary conditions object:
# - West=2 selects a specific type of west boundary condition
# - celeris=False indicates we are NOT using the full CelerisWebGPU folder convention here
# - filename provides the time series used for boundary forcing at the west boundary
bc = BoundaryConditions(
    West=2,
    celeris=False,
    path=EXAMPLE_DIR,
    filename="irrWaves1D.txt"
)


# -----------------------------------------------------------------------------
# 2) Numerical domain (1D)
# -----------------------------------------------------------------------------
# x1, x2 define the physical domain [x1, x2]
# Nx is the number of grid cells in x
#
# In this example:
# - x from 0 to 480 (meters, or your chosen length unit)
# - Nx = 480 => dx ≈ 1.0 unit
#
# Keep x2-x1 and Nx consistent with your topo profile extent and desired resolution.
d = Domain(
    topodata=baty,
    x1=0.0,
    x2=480.0,
    Nx=480
)


# -----------------------------------------------------------------------------
# 3) Solver configuration
# -----------------------------------------------------------------------------
# model='Bouss': enhanced Boussinesq-type formulation (weakly dispersive, nonlinear)
# timeScheme=2: scheme selection 
# pred_or_corrector=True: enables predictor/corrector scheme 
# useBreakingModel=True: enables wave breaking parameterization

solver = Solver(
    model="Bouss",
    domain=d,
    boundary_conditions=bc,
    timeScheme=2,
    pred_or_corrector=True,
    useBreakingModel=True
)


# -----------------------------------------------------------------------------
# 4) Run / Evolve
# -----------------------------------------------------------------------------
# maxsteps: number of time steps to run
run = Evolve(solver=solver, maxsteps=10000)

# Display mode (1D visualization)
run.Evolve_1D_Display()

# For headless (no GUI) runs, comment out the display line above and use:
# run.Evolve_Headless()

