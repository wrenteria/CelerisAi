# Add Celeris to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath('../..'))

import taichi as ti
from celeris.domain import Topodata, BoundaryConditions,Domain
from celeris.solver import Solver
from celeris.runner import Evolve
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
"""
SETUP file to perform Continuous Data Assimilation (CDA) in an area, then predict the flow field evolution in space and time:

Given observed data of surface elevation (η), in an assimilation area, estimate the full flow field
(η,hu). Data to be assimilated corresponds to a time serie of surface profiles
generate by other Boussinesq simulation. 
The Δt in this problem is the same as the observed data. 
"""
### OBSERVATION PARAMETERS
#'eta_obs' array from data observation -TO BE ASSIMILATED
eta_observed = np.load('./scratch/synth_eta1d.npy')
# 'hu_obs' to verify learning  -TO BE LEARNED
hu_observed = np.load('./scratch/synth_u1d.npy')
# shape of the observed dataset
Ntot,Nx = eta_observed.shape

### CELERISAI Set up
model = 'SWE' # model to be used in the CelerisAi solver.
precision =ti.f32
ti.init(arch = ti.gpu)
#ti.init(arch = ti.cpu,debug=True)
baty = Topodata(filename='Topo1D.txt',path='../../examples/1D',datatype='xz')
bc = BoundaryConditions(celeris=False,precision=precision)
d = Domain(topodata=baty,x1=0.0,x2=480.0,Nx=480,differentiability=True,precision=precision)
solver = Solver(model=model,domain=d,boundary_conditions=bc,pred_or_corrector=2,useBreakingModel=True)
run = Evolve(solver = solver, maxsteps= Ntot)
# Plotting parameters
x_coord,zz =solver.domain.topofield()
LimAssim = Nx//3

### OPTIMIZATION PARAMETERS
# Define a scalar loss
loss = ti.field(dtype=precision, shape=(),needs_grad=True) 
# Define a learning rate
l_r = 0.01
# Number of iterations
NumIters = 1
##  ADAM OPTIMIZER
# outside your loop
m = ti.Vector.field(4, ti.f32, shape=(Nx,1))   # 1st moment
v = ti.Vector.field(4, ti.f32, shape=(Nx,1))   # 2nd moment
beta1, beta2, eps = 0.9, 0.999, 1e-8
timestep = ti.field( ti.i32,   shape=() )

# Load the observed data into Taichi arrays
eta_obs =ti.field( dtype=precision,shape=eta_observed.shape)
eta_obs.from_numpy(eta_observed)
hu_obs =ti.field( dtype=precision,shape=hu_observed.shape)
hu_obs.from_numpy(hu_observed)

@ti.kernel
def GradientDescent(lr: ti.f32,W:ti.template()):
    # Update the data containers
    for I in ti.grouped(W):
        W[I].x -= W.grad[I].x * lr
        W[I].y -= W.grad[I].y * lr 
        W[I].z -= W.grad[I].z * lr 
@ti.kernel
def Adam(lr: precision,W: ti.template()):
    timestep[None] += 1
    for I in ti.grouped(W):
        grad = W.grad[I]
        # update moments
        m[I] = beta1 * m[I] + (1 - beta1) * grad
        v[I] = beta2 * v[I] + (1 - beta2) * grad * grad
        # bias correction
        m_hat = m[I] / (1 - ti.pow(beta1,timestep[None]))
        v_hat = v[I] / (1 - ti.pow(beta2,timestep[None]))
        # parameter update
        W[I] -= lr * m_hat / (ti.sqrt(v_hat) + eps)                             
@ti.kernel
def compute_loss(k: ti.i32):
    # Compute the loss function on Assimilation Area
    for i in range(LimAssim):
        loss[None] += (eta_obs[k,i] - run.solver.State[i,0].x)**2
        
def clearGradients():
    # Otherwise the gradient accumulates over the simulation
    run.solver.State.grad.fill(0.0)
    run.solver.stateUVstar.grad.fill(0.0)
    run.solver.oldGradients.grad.fill(0.0)
    run.solver.oldOldGradients.grad.fill(0.0)
    run.solver.F_G_star_oldOldGradients.grad.fill(0.0)
    
def plot(k):
    if k==1 or k%10==0:
        q=run.solver.State
        q_np = q.to_numpy()
        B=run.solver.Bottom
        B_np = B.to_numpy()
        fig, ax = plt.subplots(2, figsize=(6, 6))
        ax[0].set_title(f'Surface Elevation $\eta$ at t={k*run.solver.dt:.1f}s')
        ax[0].plot(x_coord,B_np[2,:,0],c='0.8')
        ax[0].plot(x_coord,q_np[:,0,0], 'blue',label='Assimilated')
        ax[0].plot(x_coord,eta_observed[k], 'k',linestyle='--',label=f'Observ.')
        ax[0].axvline(x_coord[LimAssim],color='c',linestyle='--')
        ax[0].set_ylim(-3,3)
        ax[0].set_xlim(0,470)
        ax[0].set_ylabel('$\eta$')
        ax[0].legend(loc='upper left')  
        
        ax1 = ax[1].twinx()
        ax[1].set_title(f'Momentum hu')
        ax1.plot(x_coord,B_np[2,:,0],c='0.8')
        ax1.set_ylabel('B(m)',color='gray',fontsize=9)
        ax1.tick_params(axis="y", labelcolor='gray')
        ax1.set_ylim(2*-4.85,2*9.69)
        ax[1].plot(x_coord,q_np[:,0,1], 'red',label='Learned')
        ax[1].plot(x_coord,hu_observed[k], 'k',linestyle='--',label=f'Observ.')
        ax[1].set_ylim(-15,30)
        ax[1].set_xlim(0,470)
        ax[1].set_ylabel('$hu$')
        ax[1].set_xlabel('x-coord')
        ax[1].legend(loc='upper left')  
        ax[1].axvline(x_coord[LimAssim],color='c',linestyle='--')      
        plt.savefig(f'./plots/output_{k:04d}.png')
        plt.close()

def verbosity(k,data_container,index):
    itera =f'TimeStep: {k} Loss: {loss[None]:.4f}'
    grad_np = data_container.grad.to_numpy()  # shape (Nx,1,4)
    max_ = np.max(np.abs(grad_np[:,:,index]))
    mean_ = np.mean(np.abs(grad_np[:,:,index]))
    print(itera)
    print(f"| max|∂L/∂D₀| = {max_:.3e} - mean |∂L/∂D₀| = {mean_:.3e}")

#############################################################################################    
## Assimilation / Simulation
# Initialize data containers
run.Evolve_0()
for k in range(Ntot):
    time = k*run.solver.dt
    print(f'Working at step:{k} Time: {time:.3f}')
    plot(k)
    for opt in range(NumIters):
        loss[None]=0.0
        clearGradients()
        with ti.ad.Tape(loss=loss ):
            # Assimilation / correction
            run.Evolve_Steps(k)
            compute_loss(k)
        verbosity(k,run.solver.State,0)
        verbosity(k,run.solver.State,1)
        Adam(l_r,run.solver.State)
        Adam(l_r,run.solver.stateUVstar)
        Adam(l_r,run.solver.oldGradients)
        Adam(l_r,run.solver.oldOldGradients) 
        if model=='Bouss':
              Adam(l_r,run.solver.F_G_star_oldOldGradients) 

