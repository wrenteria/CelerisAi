from celeris.solver import *
from celeris.utils import *
from taichi import tools # For saving images
import imageio # For creating GIFs
import os
import time

# Done at top of translation unit for now, clean-up later - JB
base_frame_dir = './plots/' # ! Assumes we are in CelerisAi/ directory
os.makedirs(base_frame_dir, exist_ok=True) # Ensure base_frame_dir exists
frame_paths = [] # List of frame paths for later use, i.e. making gifs/mp4s


@ti.data_oriented
class Evolve:
    """
    Controls and runs the main simulation loop of CelerisAi in various modes (headless,
    1D, 2D with visualization, etc.).

    This class ties together the `Domain`, `BoundaryConditions`, and `Solver` classes, and 
    manages the time-stepping workflow. It includes methods for:

    1. **Initialization** (`Evolve_0`):
      
       - Fills the bottom field with bathymetry/topography data.
       - Initializes solver states (water height, velocity, etc.).
       - Computes tridiagonal coefficients if using a Boussinesq model.

    2. **Main Time-Stepping** (`Evolve_Steps`):
     
       - Runs reconstruction (Pass1) and flux computations (Pass2).
       - Handles wave breaking if enabled.
       - Integrates the solution one or more steps forward in time (Pass3, Pass3Bous, Pass3_SedTrans).
       - Updates boundary conditions.
       - Optionally solves tridiagonal systems for Boussinesq dispersion.
       - Copies or shifts data between fields for multi-stage time integrators.

    3. **Headless Execution** (`Evolve_Headless`):
    
       - Executes the simulation loop without rendering or displaying results, 
         minimizing overhead and focusing on performance.
       - Periodically logs timing information and can save the simulation states (e.g. `State` arrays).

    4. **1D Visualization** (`Evolve_1D_Display`):
    
       - Specialized loop for 1D simulations, displaying free surface (eta) 
         and bathymetry in a window using either taichi-gui or legacy GUI fallback.

    5. **2D Visualization** (`Evolve_Display`):
    
       - Interactive loop for 2D simulations.  
       - Renders wave height (h), free surface elevation (eta), or vorticity (vor) 
         in real-time.  
       - Allows saving images and assembling them into a GIF.

    6. **Rendering and Color Mapping** (Kernels like `paint`, `paint_new`, `painting_h`, `painting_eta`, `painting_vor`, etc.):
    
       - Populates 2D Taichi fields (`self.image`, `self.solver.pixel`, etc.) based on 
         solver results, for real-time visualization.  
       - Supports multiple coloring strategies (e.g. realistic wave colors, topography shading, sediment rendering).

    Args:
        domain (Domain): The domain class containing spatial parameters.
        boundary_conditions (BoundaryConditions): Class managing boundary setup (walls, waves, etc.).
        solver (Solver): The main numerical solver class controlling the fluid model, time scheme, etc.
        maxsteps (int, optional): Maximum number of time steps to simulate. Defaults to 1000.
        outdir (str, optional): Output directory path for saving states, frames, etc. Defaults to None.
        saveimg (bool, optional): If True, saves image frames at intervals (for creating GIFs or offline processing). 
            Defaults to False.
        vmin (float, optional): Minimum value for visualization color scaling (e.g. wave elevation). Defaults to -1.5.
        vmax (float, optional): Maximum value for visualization color scaling. Defaults to 1.5.

    Attributes:
        solver (Solver): The numerical solver controlling fluid/morphodynamics.
        maxsteps (int): Number of time steps for the simulation run.
        dt (float): Time step size imported from the solver.
        timeScheme (int): Time integration scheme (Euler, predictor, predictor-corrector).
        saveimg (bool): Flag indicating whether to save frames.
        vmin (float): Minimum scale for color mapping wave or vorticity values.
        vmax (float): Maximum scale for color mapping wave or vorticity values.
        outdir (str): Directory for saving outputs.
        image (ti.Vector.field): 2D field to hold RGB color information for visualization.
        ocean (ti.Vector.field): 1D array of color samples (RGB) for water visualization or general colormap usage.
        colormap_ocean (str): A string identifier for the colormap used for water.
        bottom1D, indexbottom1D, eta1D: Fields used in 1D visualization of bottom topography and water surface.
        x_scale, y_scale (float): Scaling factors for 1D plots in the GUI.

    Typical Usage:
    
        >>> evolve = Evolve(domain=dom, boundary_conditions=bc, solver=sol, maxsteps=2000, outdir="results")
        >>> evolve.Evolve_Display(vmin=-1.0, vmax=1.0, variable='eta', cmapWater='Blues_r', showSediment=True)

    Note:
        - The class attempts to use Taichi's GGUI if available for improved performance 
          and better UI control. If GGUI is not available, it falls back to legacy Taichi GUI.
        - Various coloring kernels (`painting_h`, `painting_eta`, etc.) can be customized 
          to match user-defined styles or to highlight specific flow features.
    """
    def __init__(self,
                 domain=None,
                 boundary_conditions = None,
                 solver=None,
                 maxsteps= 1000,
                 outdir=None,
                 saveimg=False,
                 vmin=-1.5,
                 vmax=1.5,
                 ):
        self.solver = solver
        self.maxsteps=maxsteps
        self.dt = self.solver.dt
        self.timeScheme = self.solver.timeScheme
        self.saveimg = saveimg
        self.vmin = vmin
        self.vmax = vmax
        self.outdir = outdir
        # To visualization
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(self.solver.nx,self.solver.ny))
        self.ocean = ti.Vector.field(3, dtype=ti.f16, shape=16)
        self.colormap_ocean = 'Blues_r'
        # To visualize 1D
        self.bottom1D = ti.Vector.field(2, dtype=ti.f32, shape = self.solver.nx)
        self.indexbottom1D = ti.field(dtype=ti.i32, shape = 2*self.solver.nx)
        self.eta1D = ti.Vector.field(2, dtype=ti.f32, shape = self.solver.nx)
        self.x_scale = self.solver.nx * self.solver.dx
        self.y_scale = 2 * self.solver.base_depth

    def Evolve_0(self):
        """
        One-time initialization steps:
          - Fills bottom field (bathymetry/topo).
          - Initializes solver states (fluid variables).
          - Computes tridiagonal coefficients if the model is Boussinesq.
          - Prints simulation parameters (model type, time step, etc.).
        """
        self.solver.fill_bottom_field()
        self.solver.fill_bottom_friction()
        self.solver.InitStates()
        self.solver.tridiag_coeffs_X()
        self.solver.tridiag_coeffs_Y()
        print('Model: ',self.solver.model)
        print('Numerical Scheme: ',self.solver.timeScheme,' dx:',self.solver.dx,' dy:',self.solver.dy)
        print('Breaking Model: ', self.solver.useBreakingModel,' Sediment Transport: ', self.solver.useSedTransModel)
        print('Time delta: ',self.dt)

    def Evolve_Steps(self,step=0):
        """
        Advances the solution by one time step (or one sub-step) according to the selected time scheme.

        Internally calls:
          - Pass1 (and Pass1_SedTrans if sediment is enabled)
          - Pass2
          - Optional wave breaking model
          - Pass3 or Pass3Bous for predictor step
          - BoundaryPass to enforce boundary conditions
          - Run_Tridiag_solver for Boussinesq models
          - (If 4th-order  predictor-corrector) a second cycle of Pass1, Pass2, Pass3 or Pass3Bous
          - Copies or shifts old/predicted states for multi-stage time integrators
        """
        i = step

        self.solver.Pass1(step=i)

        if self.solver.useSedTransModel:
            self.solver.Pass1_SedTrans()

        self.solver.Pass2()

        if self.solver.useBreakingModel:
            self.solver.Pass_Breaking(time=self.dt*i-self.dt)

        if self.solver.model=='SWE':
            self.solver.Pass3(pred_or_corrector=1)      # Predictor Step in 'SWE'
        else:
            self.solver.Pass3Bous(pred_or_corrector=1)  # Predictor Step in 'BOUSS'

        self.solver.copy_states(src=self.solver.dU_by_dt,dst=self.solver.predictedGradients)

        if self.solver.useSedTransModel:
            self.solver.Pass3_SedTrans(pred_or_corrector=1)
            self.solver.copy_states(src=self.solver.dU_by_dt_Sed,dst=self.solver.predictedGradients_Sed)

        self.solver.BoundaryPass(time=self.dt*i , txState=self.solver.current_stateUVstar)

        self.solver.Run_Tridiag_solver() # Run TridiagSolver for Bouss and copy current_stateUVstar to NewState

        if self.solver.model!='SWE':
            self.solver.BoundaryPass(time=self.dt*i , txState=self.solver.NewState)

        if self.solver.model=='Bouss':
            self.solver.copy_states(src=self.solver.F_G_star_oldGradients,dst=self.solver.F_G_star_oldOldGradients)
            self.solver.copy_states(src=self.solver.predictedF_G_star,dst=self.solver.F_G_star_oldGradients)

        if self.solver.timeScheme==2:
            self.solver.copy_states(src=self.solver.NewState,dst=self.solver.State)

            if self.solver.useSedTransModel:
                self.solver.copy_states(src=self.solver.NewState_Sed,dst=self.solver.State_Sed)

            self.solver.Pass1(step=i)

            if self.solver.useSedTransModel:
                self.solver.Pass1_SedTrans()

            self.solver.Pass2()

            if self.solver.useBreakingModel:
                self.solver.Pass_Breaking(time=self.dt*i)

            if self.solver.model=='SWE':
                self.solver.Pass3(pred_or_corrector=2)
            else:
                self.solver.Pass3Bous(pred_or_corrector=2)

            if self.solver.useSedTransModel:
                self.solver.Pass3_SedTrans(pred_or_corrector=2)

            self.solver.BoundaryPass(time=self.dt*i , txState=self.solver.current_stateUVstar)

            self.solver.Run_Tridiag_solver()

            if self.solver.model!='SWE':
                self.solver.BoundaryPass(time=self.dt*i , txState=self.solver.NewState)

            if self.solver.useSedTransModel:
                self.solver.Update_Bottom()
                if self.solver.model=='Bouss':
                    self.solver.fill_bottom_field()
                    self.solver.tridiag_coeffs_X()
                    self.solver.tridiag_coeffs_Y()

        # shift gradients
        self.solver.copy_states(src=self.solver.oldGradients,dst=self.solver.oldOldGradients)
        self.solver.copy_states(src=self.solver.predictedGradients,dst=self.solver.oldGradients)

        # Copy future states
        self.solver.copy_states(src=self.solver.NewState,dst=self.solver.State)
        self.solver.copy_states(src=self.solver.current_stateUVstar,dst=self.solver.stateUVstar)


        if self.solver.useSedTransModel:
            self.solver.copy_states(src=self.solver.oldGradients_Sed,dst=self.solver.oldOldGradients_Sed)
            self.solver.copy_states(src=self.solver.predictedGradients_Sed,dst=self.solver.oldGradients_Sed)
            self.solver.copy_states(src=self.solver.NewState_Sed,dst=self.solver.State_Sed)

        # To test pressure
        #self.solver.Ship_pressure(px_init=10,py_init=50,steps=int(i))

    def Evolve_Headless(self):
        """
        Runs CelerisAi without any visualization, printing timing info periodically
        and optionally saving states to disk.

        Steps:
          1. Calls Evolve_0() to initialize fields and solver state.
          2. Loops over `maxsteps`, calling Evolve_Steps() each iteration.
          3. Logs simulation time and performance metrics every 100 steps.
          4. If an output directory is specified, saves solver state arrays to .npy files.
        """
        self.Evolve_0()
        start_time = time.time()

        for i in range(self.maxsteps):
            self.Evolve_Steps(i)
            if i==1:
                start_time = time.time() - 0.00001  # reset the "start" time as there is overhead before loop starts, and add small shift to prevent float divide by zero

            if i==1 or (i%100)==0:
                compTime = time.time() - start_time
                print('Current Simulation time: {:2.2f}s at step: {}-- Ratio:{:2.2f}--CompTime:{:2.2f}'.format(self.dt*i,i,(self.dt*i)/compTime,compTime))
                if self.solver.outdir:
                    state=self.solver.State.to_numpy()
                    np.save('{}/state_{}.npy'.format(self.outdir,int(i)),state)

    @ti.func
    def brk_color (self, x,y0,y1,x0,x1):
        """
        Interpolates between two values (y0, y1) based on x in [x0, x1].

        Used to smoothly vary color or other scalar values between two extremes:
            (x0 -> y0) to (x1 -> y1).
        """
        return  (y0 * (x1 - x) + y1 * (x - x0) ) / (x1 - x0)

    @ti.kernel
    def paint_new(self):
        """
        A simple rendering kernel mixing bottom topography and wave height.
        Uses a linear interpolation (brk_color) to assign colors to each cell
        based on water depth or topography.
        """
        for i,j in ti.ndrange((0,self.solver.nx),(0,self.solver.ny)):
            self.solver.pixel[i,j] = self.brk_color(self.solver.Bottom[2,i,j], 0.75, 1,self.solver.maxtopo, -1*self.solver.maxtopo)
            flow = self.solver.State[i,j][0] -self.solver.Bottom[2,i,j]
            if flow > 0.0001 :
                #self.solver.pixel[i,j] = self.brk_color(self.solver.State[i,j][0], 0, 0.75,self.vmin,self.vmax)
                self.solver.pixel[i,j] = self.brk_color(flow, 0, 0.75,0.0001,self.solver.base_depth+3)


    @ti.kernel
    def InitColors(self,arr:ti.types.ndarray(dtype=ti.f16, ndim=2)):
        """
        Copies an external NumPy array of shape (N, 3) (RGB colors) into the 
        internal taichi field `self.ocean`.

        Args:
            arr (np.ndarray): (N, 3) array of float16 color data (e.g., from a Matplotlib colormap).
        """
        for i in self.ocean:
            self.ocean[i].x = arr[i,0]
            self.ocean[i].y = arr[i,1]
            self.ocean[i].z = arr[i,2]

    @ti.kernel
    def painting_h(self):
        """
        Kernel for visualizing water depth (h) in a "realistic wave" style.

        - Normalizes water depth to [0, base_depth].
        - Chooses colors from the `ocean` array based on the normalized depth (linear interpolation).
        - Make a difference between the water areas (flow > 0.25) from shallow/wet sand and land.
        """
        num_colors = self.ocean.shape[0]
        step = 1.0 / num_colors
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            flow = self.solver.State[i,j][0] -self.solver.Bottom[2,i,j] # only water
            land = max(0.0,self.solver.Bottom[2,i,j]) # only positive topo
            water_col =  flow/self.solver.base_depth  # Water column normalized
            land_elevation = land  / self.solver.maxtopo # Topo normalized
            index = int(water_col / step)           # which color interval we're in
            index = ti.min(index, num_colors - 2)   # clamp index to avoid out-of-bounds
            t = (water_col - index * step) / step   # fractional position between the colors
            if flow > 0.25:  # Water area
                self.image[i, j] =  self.ocean[index] * (1 - t) +  self.ocean[index + 1] * t
            elif 0.25<flow<0:
                self.image[i, j] =  ti.Vector([0.8039,0.7921,0.7372]) # Wet Sand
            else:
                self.image[i, j] = ti.Vector([0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25])


    @ti.kernel
    def painting_eta(self):
        """
        Kernel for visualizing free surface elevation (eta).

        - Normalizes eta to [vmin, vmax] for color lookup.
        - Distinguishes water areas from wet sand and land similar to `painting_h`.
        """
        num_colors = self.ocean.shape[0]
        step = 1.0 / num_colors
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            flow = self.solver.State[i,j][0] -self.solver.Bottom[2,i,j] # only water
            wave = self.solver.State[i,j][0]
            wave = (wave-self.vmin)/(self.vmax-self.vmin)
            land = max(0.0,self.solver.Bottom[2,i,j]) # only positive topo
            land_elevation = land  / self.solver.maxtopo # Topo normalized
            index = int(wave / step)           # which color interval we're in
            index = ti.min(index, num_colors - 2)   # clamp index to avoid out-of-bounds
            t = (wave - index * step) / step   # fractional position between the colors
            if flow > 0.0:  # Water area
                self.image[i, j] =  self.ocean[index] * (1 - t) +  self.ocean[index + 1] * t
            elif 0.25<flow<0:
                self.image[i, j] =  ti.Vector([0.8039,0.7921,0.7372]) # Wet Sand
            else:
                self.image[i, j] = ti.Vector([0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25])

    @ti.kernel
    def painting_vor(self):
        """
        Kernel for visualizing vorticity (vor).

        - Approximates vorticity by differences in velocity between adjacent cells.
        - Normalizes the result into [vmin, vmax] for color lookups in `ocean`.
        - Distinguishes water vs. land similar to `painting_h`/`painting_eta`.
        """
        num_colors = self.ocean.shape[0]
        step = 1.0 / num_colors
        for i, j in ti.ndrange(self.solver.nx, self.solver.ny):
            rightIdx = ti.min(i + 1, self.solver.nx - 1)
            upIdx = ti.min(j + 1, self.solver.ny - 1)

            B = self.solver.Bottom[2,i,j]
            B_right = self.solver.Bottom[2,rightIdx,j]
            B_up = self.solver.Bottom[2,i,upIdx]

            q = self.solver.State[i, j]
            q_up = self.solver.State[i, upIdx]
            q_right = self.solver.State[rightIdx, j]

            h = q.x -B
            h_right = q_right.x - B_right
            h_up = q_up.x - B_up

            v_right = 0.0
            u_up = 0.0
            u = 0.0
            v = 0.0

            if h_right>0.05:
                v_right = q_right.z/h_right
            if h_up>0.05:
                u_up = q_right.y/h_up
            if h>0.05:
                v = q.z/h
                u = q.y/h

            vor = (v_right-v)/self.solver.dx - (u_up-u)/self.solver.dy

            vor = (vor-self.vmin)/(self.vmax-self.vmin)
            land = max(0.0,B) # only positive topo
            land_elevation = land  / self.solver.maxtopo # Topo normalized
            index = int(vor / step)           # which color interval we're in
            index = ti.min(index, num_colors - 2)   # clamp index to avoid out-of-bounds
            t = (vor - index * step) / step   # fractional position between the colors
            if h > 0.0:  # Water area
                self.image[i, j] =  self.ocean[index] * (1 - t) +  self.ocean[index + 1] * t
            elif 0.25<h<0:
                self.image[i, j] =  ti.Vector([0.8039,0.7921,0.7372]) # Wet Sand
            else:
                self.image[i, j] = ti.Vector([0.6 + land_elevation * 0.4, 0.4 + land_elevation * 0.3, 0.25])
    @ti.kernel
    def paint(self):
        """
        General paint kernel that merges bottom topography and free-surface height.

        - Assigns color based on bottom topography if there's no water.
        - Interpolates water color if flow depth > 0.0001.
        - If sediment transport is enabled, merges sediment concentration into final color.
        """
        for i,j in ti.ndrange((0,self.solver.nx),(0,self.solver.ny)):
            self.solver.pixel[i,j] = self.brk_color(self.solver.Bottom[2,i,j], 0.75, 1,self.solver.maxtopo, -1*self.solver.maxtopo)
            flow = self.solver.State[i,j][0] -self.solver.Bottom[2,i,j]
            # Merge water and topo values
            if flow > 0.0001 :
                self.solver.pixel[i,j] = self.brk_color(flow, 0, 0.75, 0.0001,self.solver.base_depth+3)
                if self.solver.useSedTransModel :
                    sed = self.solver.State_Sed[i,j][0]/flow
                    if sed > 0.0001  :
                        self.solver.pixel[i,j] = self.brk_color(sed, 0.65, 0.75, 0.1, 5.0 ) # Sed

    @ti.kernel
    def bottom_paint(self):
        """
        Fills `bottom1D` with scaled bottom data for 1D visualization.
        Also populates `indexbottom1D` so that line plotting can connect them in order.
        """
        for i in self.bottom1D:
            self.bottom1D[i].x = i*self.solver.dx/self.x_scale
            self.bottom1D[i].y = 0.5+self.solver.Bottom[2,i,0]/self.y_scale

        for i in range(2*self.solver.nx-2):
            self.indexbottom1D[i] = (i + 1) // 2
            

    @ti.kernel
    def eta_paint(self):
        """
        Fills `eta1D` with scaled free-surface data for 1D visualization.
        """
        for i in self.eta1D:
            self.eta1D[i].x = i*self.solver.dx/self.x_scale
            self.eta1D[i].y = 0.5+self.solver.State[i,0][0]/20
    
    def Evolve_1D_Display(self):
        """
        Interactive loop for a 1D simulation of CelerisAi.

        - Initializes bottom, runs the main solver steps, and displays the results 
          in a small taichi-gui or GGUI window.
        - Plots the free surface (eta) and bottom profile in each iteration.
        - Optionally saves frames and can compile them into a GIF if desired.
        """
        plotpath = './plots'
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
        i = 0
        use_ggui = None
        window = None
        canvas = None
        try:
            window = ti.ui.Window("CelerisAi(1D)", (1000,200))
            canvas = window.get_canvas()
            canvas.set_background_color(color=(1,1,1))
            use_ggui = True
        except:
            # TODO : Formal error handling and logging
            print("GGUI not available, reverting to legacy Taichi GUI.")
            use_ggui = False
            use_fast_gui = False # Need ti.Vector.field equiv to self.solver.pixel to use fast_gui
            window = ti.GUI(  # noqa: F405
                'CelerisAi(1D)', (1000, 200), fast_gui=use_fast_gui
                ) # fast_gui - display directly on frame buffer if not drawing shapes or text
            canvas = None
            print("Legacy GUI initialized.")
        else:
            print("GGUI initialized without issues.")

        self.Evolve_0()
        self.bottom_paint() # To plot the bottom line
        start_time = time.time() - 0.00001

        while window.running:
            self.eta_paint()
            if use_ggui:
                canvas.circles(self.eta1D,radius=0.005,color = (0., 150/255., 255./255))
                canvas.lines(self.bottom1D,width=0.01,indices=self.indexbottom1D,color = (128/255., 0.0, 0.))
            else:
                canvas.circles(self.eta1D,radius=0.005,color = (0., 150/255., 255./255))
                canvas.circles(self.bottom1D,radius=0.0075,color = (255/255., 87/255., 51./250))
            
            self.Evolve_Steps(i)

            if i==1 or (i%100)==0:
                compTime = time.time() - start_time
                print('Current Simulation time: {:2.2f}s at step: {}-- Ratio:{:2.2f}--CompTime:{:2.2f}'.format(self.dt*i,i,(self.dt*i)/compTime,compTime))
                
                if self.saveimg:
                    frame = int(i)
                    frame_filename = 'frame_{}.png'.format(frame)
                    frame_path = os.path.join(base_frame_dir, frame_filename)
                    frame_paths.append(frame_path)
                    if use_ggui:
                        window.save_image(frame_path)
                    else:
                        tools.imwrite(self.solver.pixel.to_numpy(), frame_path)

                #window.show()
                if self.outdir:
                    state=self.solver.State.to_numpy()
                    np.save('{}/state_{}.npy'.format(self.outdir,int(i)),state)
           
            window.show()
            
            if i > self.maxsteps:
                if frame_paths: # Check if there are frames to create a GIF
                    gif_filename = f"video.gif"
                    gif_path = os.path.join(base_frame_dir, gif_filename)
                    try:
                        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
                            for frame_path in frame_paths:
                                image = imageio.imread(frame_path)
                                writer.append_data(image)
                        print(f"GIF created at {gif_path}")
                    except Exception as e:
                        print(f"Error creating GIF: {e}")
                break
            i = i+1

    
    def Evolve_Display(self,vmin=None,vmax=None,variable='h',cmapWater='Blues_r',showSediment=False):
        """
        Interactive loop for a 2D simulation of CelerisAi with real-time visualization.

        - Calls `Evolve_0()` once to initialize solver fields.
        - Creates a window (either GGUI or legacy GUI).
        - Uses a custom colormap from `celeris_matplotlib()` if `showSediment` is True 
          and the solver has sediment transport enabled.
        - Allows switching between different visualization variables:
            * `h`: Water depth
            * `eta`: Free surface elevation
            * `vor`: Vorticity
        - Saves frames if `saveimg` is True, can compile them into a GIF, and optionally 
          saves solver states to `.npy`.

        Args:
            vmin (float, optional): Minimum colormap value for rendering. Defaults to None (class-level vmin).
            vmax (float, optional): Maximum colormap value for rendering. Defaults to None (class-level vmax).
            variable (str, optional): Which variable to render (`h`, `eta`, or `vor`). Defaults to 'h'.
            cmapWater (str, optional): Matplotlib colormap name for water. Defaults to 'Blues_r'.
            showSediment (bool, optional): If True, merges in a sediment colormap when sediment transport is active. 
                Defaults to False.
        """
        if vmin!=None:
            self.vmin = vmin
        if vmax!=None:
            self.vmax = vmax

        plotpath = './plots'
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
        i = 0
        use_ggui = None
        window = None
        canvas = None
        try:
            window = ti.ui.Window("CelerisAi", (self.solver.nx,self.solver.ny))
            canvas = window.get_canvas()
            use_ggui = True
        except:
            # TODO : Formal error handling and logging
            print("GGUI not available, reverting to legacy Taichi GUI.")
            use_ggui = False
            use_fast_gui = False # Need ti.Vector.field equiv to self.solver.pixel to use fast_gui
            window = ti.GUI(  # noqa: F405
                'CelerisAi', (self.solver.nx, self.solver.ny), fast_gui=use_fast_gui
                ) # fast_gui - display directly on frame buffer if not drawing shapes or text
            canvas = None
            print("Legacy GUI initialized.")
        else:
            print("GGUI initialized without issues.")

        # Customized colormap
        if showSediment:
            cmap = celeris_matplotlib(water=cmapWater,sediment='afmhot_r', SedTrans=self.solver.useSedTransModel )
        else:
            cmap = celeris_matplotlib(water=cmapWater)
        #cmap = celeris_waves()
        #cmap = celeris_matplotlib(water='Blues_r',sediment='afmhot_r', SedTrans=self.useSedTransModel )

        self.Evolve_0()
        # Set colors - using the matplotlib colormapsand convert these into Taichi tensors
        numpy_ocean = ColorsfromMPL(cmapWater)
        self.InitColors(numpy_ocean)

        start_time = time.time()

        while window.running:
            #self.paint()
            #self.paint_new()
            #self.paint()
            if variable=='h':
                self.painting_h()
            if variable=='eta':
                self.painting_eta()
            if variable=='vor':
                self.painting_vor()

            if use_ggui:
                # canvas.contour(self.solver.pixel, cmap_name=cmap ) # Same functionality as set cmap-pixel-to np
                # canvas.contour(self.solver.pixel,cmap_name='plasma') # Same functionality as set cmap-pixel-to np
                canvas.set_image(self.image) # using the Taichi tensors to render the image
            else:
                #window.set_image(self.solver.pixel)
                window.set_image(self.image) # using the Taichi tensors to render the image
            self.Evolve_Steps(i)


            if i==1:
                start_time = time.time() - 0.00001  # reset the "start" time as there is overhead before loop starts, and add small shift to prevent float divide by zero

            if i==1 or (i%100)==0:
                compTime = time.time() - start_time
                print('Current Simulation time: {:2.2f}s at step: {}-- Ratio:{:2.2f}--CompTime:{:2.2f}'.format(self.dt*i,i,(self.dt*i)/compTime,compTime))
                frame = int(i)
                frame_filename = 'frame_{}.png'.format(frame)
                frame_path = os.path.join(base_frame_dir, frame_filename)
                frame_paths.append(frame_path)
                if self.saveimg:
                    if use_ggui:
                        window.save_image(frame_path)
                    else:
                        tools.imwrite(self.solver.pixel.to_numpy(), frame_path)


                # if self.saveimg and not use_ggui:
                #     try:
                #         window.show(frame_path)
                #     except Exception as e:
                #         print(f"Error showing frame: {e},  fallback to tools.imwrite...")
                #         try:
                #             tools.imwrite(self.solver.pixel.to_numpy(), frame_path)
                #             frame_paths.append(frame_path)
                #         except Exception as e:
                #             print(f"Error writing frame with tools.imwrite: {e}")
                #     else:
                #         frame_paths.append(frame_path)
                # elif self.saveimg and use_ggui:
                #     try:
                #         tools.imwrite(self.solver.pixel.to_numpy(), frame_path)
                #         frame_paths.append(frame_path)
                #     except Exception as e:
                #         print(f"Error writing frame with tools.imwrite: {e}")
                # elif not use_ggui and not self.saveimg:
                #     window.show()
                # else:
                #     print("WARNING - No output method selected, frame not saved or displayed...")
                # if not use_ggui:
                #     continue

                #window.show()
                if self.solver.outdir:
                    state=self.solver.State.to_numpy()
                    np.save('{}/state_{}.npy'.format(self.outdir,int(i)),state)
            # Show window in the right position (after save image) for GGUI systems
            if i%5==0:
                # Improve the performance.The visualization is done only every 5 timesteps
                window.show()
            
            if i > self.maxsteps:
                if frame_paths: # Check if there are frames to create a GIF
                    gif_filename = f"video.gif"
                    gif_path = os.path.join(base_frame_dir, gif_filename)
                    try:
                        with imageio.get_writer(gif_path, mode='I', duration=0.1) as writer:
                            for frame_path in frame_paths:
                                image = imageio.imread(frame_path)
                                writer.append_data(image)
                        print(f"GIF created at {gif_path}")
                    except Exception as e:
                        print(f"Error creating GIF: {e}")
                break
            i = i+1
if __name__ == "__main__":
    print('Runner Module used in Celeris')
