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
    def __init__(self,
                 domain=None,
                 boundary_conditions = None,
                 solver=None,
                 maxsteps= 1000,
                 outdir=None,
                 saveimg=False,
                 vmin=-1.5,
                 vmax=1.5
                 ):
        self.solver = solver
        self.maxsteps=maxsteps
        self.dt = self.solver.dt
        self.timeScheme = self.solver.timeScheme
        self.saveimg = saveimg
        self.vmin = vmin
        self.vmax = vmax

    def Evolve_0(self):
        self.solver.fill_bottom_field()
        self.solver.InitStates()
        self.solver.tridiag_coeffs_X()
        self.solver.tridiag_coeffs_Y()
        print('Model: ',self.solver.model)
        print('Numerical Scheme: ',self.solver.timeScheme,' dx:',self.solver.dx,' dy:',self.solver.dy)
        print('Breaking Model: ', self.solver.useBreakingModel,' Sediment Transport: ', self.solver.useSedTransModel)
        print('Time delta: ',self.dt)

    def Evolve_Steps(self,step=0):
        i = step

        self.solver.Pass1()

        if self.solver.useSedTransModel:
            self.solver.Pass1_SedTrans()

        self.solver.Pass2()

        if self.solver.useBreakingModel:
            self.solver.Pass_Breaking(time=self.dt*i-self.dt)

        if self.solver.model=='SWE':
            self.solver.Pass3(pred_or_corrector=1)      # Predictor Step in 'SWE'
        else:
            self.solver.Pass3Bous(pred_or_corrector=1)  # Predicto Step in 'BOUSS'

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

            self.solver.Pass1()

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
        # Interp. to get changes in color
        return  (y0 * (x1 - x) + y1 * (x - x0) ) / (x1 - x0)

    @ti.kernel
    def paint_new(self):
        for i,j in ti.ndrange((0,self.solver.nx),(0,self.solver.ny)):
            self.solver.pixel[i,j] = self.brk_color(self.solver.Bottom[2,i,j], 0.75, 1,self.solver.maxtopo, -1*self.solver.maxtopo)
            flow = self.solver.State[i,j][0] -self.solver.Bottom[2,i,j]
            if flow > 0.0001 :
                #self.solver.pixel[i,j] = self.brk_color(self.solver.State[i,j][0], 0, 0.75,self.vmin,self.vmax)
                self.solver.pixel[i,j] = self.brk_color(flow, 0, 0.75,0.0001,self.solver.base_depth+3)


    @ti.kernel
    def paint(self):
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


    def Evolve_Display(self,vmin=None,vmax=None,cmapWater='Blues_r',showSediment=False):
        if vmin!=None:
            self.vmin = vmin
        if vmax!=None:
            self.vmax = vmax

        plotpath = './plots'
        if not os.path.exists(plotpath):
            os.makedirs(plotpath)
        i = 0.0
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

        start_time = time.time()

        while window.running:
            self.paint()
            #self.paint_new()
            if use_ggui:
                canvas.contour(
                    self.solver.pixel, cmap_name=cmap
                ) # Same functionality as set cmap-pixel-to np
                # canvas.contour(self.solver.pixel,cmap_name='plasma') # Same functionality as set cmap-pixel-to np
            else:
                window.set_image(
                    self.solver.pixel
                )
            self.Evolve_Steps(i)
            #window.show()

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
