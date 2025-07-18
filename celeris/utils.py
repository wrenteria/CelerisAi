import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as clr
from scipy.interpolate import griddata
import taichi as ti
import taichi.math as tm


Vpi = np.pi#3.141592653589793

def ColorsfromMPL(cmap='Blues'):
    """
    Extracts a small set of color values from a Matplotlib colormap.

    This function accesses any Matplotlib colormap by name, samples 16 color 
    entries from it, and returns them as a NumPy array of type float16. 
    Users can choose any of the built-in Matplotlib colormaps (e.g., 
    "viridis", "jet", "inferno", "Blues", etc.).

    Args:
        cmap (str, optional): Name of the Matplotlib colormap to sample.
            Defaults to "Blues".

    Returns:
        numpy.ndarray: A (16, 4) array of RGBA color values (float16).
            The array index corresponds to a discrete point along the 
            specified colormap.
    """
    cm_cmap =  cm.get_cmap(cmap,16)
    cm_cmap._init()
    cm_cmap = cm_cmap._lut
    cm_cmap = cm_cmap.astype(np.float16)
    return cm_cmap

def celeris_matplotlib(water='seismic',land='terrain',sediment='default',SedTrans=False):
    """
    Creates a customized matplotlib color map for visualizing water, land/topography, 
    and optionally sediment transport.

    This function merges three main color segments:
    
        1. **Water**: Ranges from 0 to 0.75 or 0 to 0.5 (depending on sediment usage).
        2. **Sediment** (optional): Placed between the water and land segments if `SedTrans` is True.
        3. **Land/Topography**: Assigned to the higher range of the color bar (e.g., 0.75 - 1 or 0.75 - 1 
            when `SedTrans` is False, and 0.75 - 1 when `SedTrans` is True).

    Args:
        water (str, optional): Name of the colormap to use for water (default "seismic").
        land (str, optional): Name of the colormap to use for land/topo (default "terrain").
        sediment (str, optional): Name of the colormap to use for sediment. If "default",
            a hard-coded set of color stops (skyblue, tan, peru, saddlebrown) is used. 
            Otherwise, a user-specified colormap is merged (default "default").
        SedTrans (bool, optional): Indicates whether sediment transport is active. 
            If True, the colormap includes an additional segment for sediment; 
            if False, only water and land segments are used (default False).

    Returns:
        matplotlib.colors.LinearSegmentedColormap: A single merged colormap with the 
        specified segments for water, (optionally) sediment, and land.

    Example:
        >>> cmap_no_sed = celeris_matplotlib(SedTrans=False)
        >>> cmap_sed = celeris_matplotlib(water="Blues", sediment="Reds", SedTrans=True)
    """
    if SedTrans==False:
        ## Water Color
        water_cmap = cm.get_cmap(water,256)
        water_cmap._init()
        water_cmap = water_cmap._lut
        N = water_cmap.shape[0]
        ramp_water = np.linspace(0,0.75,N)
        clist = []
        for i in range(N):
            clist.append((ramp_water[i],water_cmap[i]))
        ## Topo Color
        land_cmap = cm.get_cmap(land,256)
        land_cmap._init()
        land_cmap = land_cmap._lut
        N = land_cmap.shape[0]
        ramp_land = np.linspace(0.75,1,N)
        for i in range(N):
            clist.append((ramp_land[i],land_cmap[i]))
        cmap =clr.LinearSegmentedColormap.from_list("", clist,N=256)
    if SedTrans==True:
        ## Water Color
        water_cmap = cm.get_cmap(water,256)
        water_cmap._init()
        water_cmap = water_cmap._lut
        N = water_cmap.shape[0]
        ramp_water = np.linspace(0,0.5,N)
        clist = []
        for i in range(N):
            clist.append((ramp_water[i],water_cmap[i]))

        if sediment=='default':
            clist.append((0.5, 'skyblue'))
            clist.append((0.51, 'tan'))
            clist.append((0.6, 'peru'))
            clist.append((0.75, 'saddlebrown'))
        else:
            ## Sediment Color
            sediment_cmap = cm.get_cmap(sediment,256)
            sediment_cmap._init()
            sediment_cmap = sediment_cmap._lut
            N = sediment_cmap.shape[0]
            ramp_sediment = np.linspace(0.5,0.75,N)
            for i in range(N):
                clist.append((ramp_sediment[i],sediment_cmap[i]))
        ## Topo Color
        land_cmap = cm.get_cmap(land,256)
        land_cmap._init()
        land_cmap = land_cmap._lut
        N = land_cmap.shape[0]
        ramp_land = np.linspace(0.75,1,N)
        for i in range(N):
            clist.append((ramp_land[i],land_cmap[i]))
        cmap =clr.LinearSegmentedColormap.from_list("", clist,N=256)

    return cmap


def celeris_waves():
    """
    Creates a custom color map (colormap) designed to represent realistic sea water gradients.

    This function defines a series of color stops spanning blues, greens, and yellows,
    which can be used to visualize water-related data (e.g., wave heights or velocities).
    The color map transitions from light blues (representing shallower or clearer water)
    through darker blues/greens, and finally into yellowish tones that can highlight
    areas of foam or breaking waves.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: A color map object suitable for
        use with matplotlib plotting functions or other visualization frameworks.
    """
    clist = [(0, 'dodgerblue'),
            (0.125, 'lightsteelblue'),
            (0.25, 'lightskyblue'),
            (0.375, 'aliceblue'),
            (0.5, 'skyblue'),
            (0.625, 'cornflowerblue'),
            (0.75, 'royalblue'),
            (0.8, 'mediumseagreen'),
            (0.87, 'yellowgreen'),
            (0.95, 'greenyellow'),
            (1, 'yellow')]
    cmap =clr.LinearSegmentedColormap.from_list("", clist,N=256)
    return cmap

@ti.func
def MinMod(a,b,c):
    """
    Computes a simple minmod function of three values.

    The minmod function returns:
    
    - The minimum among (a, b, c) if all three are positive.
    - The maximum among (a, b, c) if all three are negative.
    - Zero otherwise.

    Args:
        a (float): First value.
        b (float): Second value.
        c (float): Third value.

    Returns:
        float: The minmod result based on the sign of the inputs.
    """
    a,b,c =float(a),float(b),float(c)
    min_value = 0.0
    if (a>0.0) and (b>0.0) and (c>0.0):
        min_value = ti.min(a,b,c)
    elif (a<0.0) and (b<0.0) and (c<0.0):
        min_value = ti.max(a,b,c)
    return min_value

@ti.func
def cosh(x):
        """
    Returns the hyperbolic cosine of `x`.

    The hyperbolic cosine is defined as:
        cosh(x) = (e^x + e^(-x)) / 2

    Args:
        x (float): The input value.

    Returns:
        float: The value of the hyperbolic cosine of `x`.
    """
        return 0.5*(ti.exp(x)+ti.exp(-1*x))

@ti.func
def sineWave(x,y,t,d,amplitude,period,theta,phase,g,wave_type):
    """
    Computes a sine wave (and related momentum terms) at a given point (x, y) and time t.

    This function uses a dispersion relation and hyperbolic tangent of wave depth
    to approximate wave number (k) and phase speed (c). It then calculates free-surface
    elevation (eta) and horizontal momentum components(hu, hv) based on wave parameters. An optional
    decay term is applied for certain wave types.

    Args:
        x (float): x-coordinate where the wave is evaluated.
        y (float): y-coordinate where the wave is evaluated.
        t (float): Current time in the simulation.
        d (float): Local water depth.
        amplitude (float): Wave amplitude.
        period (float): Wave period.
        theta (float): Wave propagation angle in radians.
        phase (float): Additional phase offset.
        g (float): Gravitational acceleration.
        wave_type (int): Wave type indicator. If `wave_type == 2`, a decay multiplier 
            is applied to the wave for demonstration/limiting purposes.

    Returns:
        ti.Vector([float, float, float]): A 3-component vector:
            - **eta**: Free-surface elevation (wave height) above still water level.
            - **hu**: Momentum in the x-direction (wave speed * wave height).
            - **hv**: Momentum in the y-direction (wave speed * wave height).

    Note:
        - The calculation for wave number `k` uses a simplified relationship assuming
          linear wave theory with a hyperbolic tangent term for finite depth.
        - The term `ti.min(1.0, t / period)` is used to gradually ramp up the wave
          from zero at t=0 (avoid sudden wave onset).
        - If `wave_type == 2`, an additional decay factor is applied as `t` approaches 
          `num_waves * period` (here `num_waves` is hard-coded to 4 in the example). For a transient pulse
        - The returned `hu` and `hv` are computed as a fraction of `g * eta / (c * k) * tanh(k * d)`,
          scaled by the direction cosines `(cos(theta), sin(theta))`.
    """
    omega = 2.0 * Vpi / period
    k = omega * omega / (g * ti.sqrt( ti.tanh(omega * omega * d / g)))
    c = omega / k
    kx = ti.cos(theta) * x * k
    ky = ti.sin(theta) * y * k

    # Gradual wave ramp-up factor (t < period => wave amplitude ramps up linearly in time)
    eta = amplitude * ti.sin(omega * t - kx - ky + phase)*ti.min(1.0, t / period)
    
    ### Check this is only valid for sinewves/irregualr
    num_waves=0
    if wave_type == 2:
        # transient pulse
        num_waves = 4
    if num_waves > 0:
        eta = eta * ti.max(0.0, ti.min(1.0, ((float(num_waves) * period - t)/period)))

    # Compute momentum components
    speed = g * eta / (c * k) * ti.tanh(k * d)
    hu = speed * tm.cos(theta)
    hv = speed * tm.sin(theta)
    return ti.Vector([eta, hu , hv])

@ti.func
def Reconstruct(west, here, east, TWO_THETAc):
    """
    Performs a piecewise linear reconstruction of a variable using a generalized minmod limiter.

    This function takes three consecutive cell-centered values (`west`, `here`, and `east`) 
    along with a limiter parameter (`TWO_THETAc`) and returns two reconstructed interface 
    values at the current cell interfaces (left/right or west/east edges).

    The reconstruction logic:
    
    - Computes slopes (z1, z2, z3) that scale differences between neighboring cells.
    - Finds the minimum among those slopes (when all have the same sign) or zero otherwise.
    - Applies a factor of 0.25 to that minimum slope to limit the reconstruction (i.e., 
      controlling oscillations).
    - Returns the reconstructed values at the left (west) and right (east) edges of 
      the current cell.

    Args:
        west (float): Value of the variable at the cell immediately to the left (j-1).
        here (float): Value of the variable at the current cell (j).
        east (float): Value of the variable at the cell immediately to the right (j+1).
        TWO_THETAc (float): Limiter parameter, typically 2 * theta, where theta is in range [1, 2] 
            for generalized minmod-type limiters.

    Returns:
        ti.types.vector(2, float): 
            A 2-component vector representing the reconstructed value at:
            - [0]: The left (west) interface of the current cell.
            - [1]: The right (east) interface of the current cell.
    """
    z1 = TWO_THETAc * (here - west)
    z2 = (east - west)
    z3 = TWO_THETAc * (east - here)
    min_value = 0.0
    if (z1>0.0) and (z2>0.0) and (z3>0.0):
        min_value = ti.min(ti.min(z1,z2),z3)
    elif (z1<0.0) and (z2<0.0) and (z3<0.0):
        min_value = ti.max(ti.max(z1,z2),z3)

    dx_grad_over_two = 0.25 * min_value
    return ti.Vector([here - dx_grad_over_two, here + dx_grad_over_two])


@ti.func
def CalcUV(h, hu, hv,hc,epsilon,dB_max):
   """
    Computes velocity and scalar concentration at cell edges given water height and momentum.

    This function takes the water depth (h), x-momentum (hu), y-momentum (hv), and 
    a scalar quantity (hc) at the cell edges (usually indexed [N, E, S, W]) and 
    returns the velocity components (u, v) and scalar concentration (c) at those edges. 
    It applies a limiting factor to avoid division by a near-zero depth.

    The key step involves computing:
        divide_by_h = 2.0 * h / (h * h + ti.max(h * h, epsilon_c))

    where:
    
    - `epsilon_c = max(epsilon, dB_max)`
    - `epsilon` is a small threshold to prevent division by zero,
    - `dB_max` represents the maximum bed-elevation difference across edges, 
      ensuring the local depth used for velocity calculation is not less than 
      the difference in water depth across an edge.

    Args:
        h (ti.types.vector): Water depth at edges, shaped [N, E, S, W].
        hu (ti.types.vector): Momentum in the x-direction at edges.
        hv (ti.types.vector): Momentum in the y-direction at edges.
        hc (ti.types.vector): Scalar quantity (e.g. concentration) at edges.
        epsilon (float): Small threshold to avoid division by zero in near-dry cells.
        dB_max (ti.types.vector or float): Maximum bed-elevation difference used to limit depth.

    Returns:
        tuple of ti.types.vector: (u, v, c) where each is a vector shaped [N, E, S, W].
            - **u**: x-velocity at edges.
            - **v**: y-velocity at edges.
            - **c**: Scalar concentration at edges.
    """
   epsilon_c = ti.max(epsilon, dB_max)
   divide_by_h = 2.0 * h / (h*h + ti.max(h*h, epsilon_c))
   #denom = h*h + max(h*h,epsilon_c)
   #divide_by_h = 2*h/ti.max(denom,epsilon_c)
   #this is important - the local depth used for the edges should not be less than the difference in water depth across the edge
   # divide_by_h = h / np.maximum(h2, epsilon)  #u = divide_by_h * hu #v = divide_by_h * hv #c = divide_by_h * hc
   return divide_by_h * hu , divide_by_h * hv , divide_by_h * hc

@ti.func
def CalcUV_Sed(h, hc1, hc2, hc3, hc4, epsilon,dB_max):
    """
    Computes sediment scalar concentrations at cell edges given the water height.

    This function calculates four sediment-related quantities (e.g., scalar concentrations 
    or sediment fractions) at the edges of a cell. It applies a limiting factor based on 
    water depth (`h`) to avoid division by a near-zero depth and to account for 
    significant bed-elevation differences.

    Args:
        h (float): Water depth at the cell edge.
        hc1 (float): Sediment/scalar quantity #1 at the edge.
        hc2 (float): Sediment/scalar quantity #2 at the edge.
        hc3 (float): Sediment/scalar quantity #3 at the edge.
        hc4 (float): Sediment/scalar quantity #4 at the edge.
        epsilon (float): Small threshold to avoid division by zero.
        dB_max (float): Maximum bed-elevation difference, used in limiting depth.

    Returns:
        ti.Vector([float, float, float, float]): 
            A 4-component vector containing the scaled sediment/scalar values
            `[c1, c2, c3, c4]` after applying the depth-limiting factor.
    """
    epsilon_c = ti.max(epsilon, dB_max)
    #h4=h*h*h*h
    divide_by_h = ti.sqrt(2.0) * h / (h*h + ti.max(h*h, epsilon_c))
    c1 = divide_by_h * hc1
    c2 = divide_by_h * hc2
    c3 = divide_by_h * hc3
    c4 = divide_by_h * hc4
    return ti.Vector([c1,c2,c3,c4])


@ti.func
def NumericalFlux(aplus, aminus, Fplus, Fminus, Udifference):
    """
    Computes a wave-speed-based numerical flux between two adjacent cells.

    This function calculates the flux across a cell interface using the wave speeds 
    `aplus` (maximum positive speed) and `aminus` (maximum negative speed) along with 
    flux values from the "plus" and "minus" sides (`Fplus`, `Fminus`) and the state 
    difference (`Udifference`). If the wave speeds cancel each other out 
    (`aplus - aminus == 0.0`), the flux is set to zero.

    The formula implemented is:
    
        flux = (aplus * Fminus - aminus * Fplus + (aplus * aminus) * Udifference) / (aplus - aminus)

    Args:
        aplus (float): Maximum positive wave speed at the cell interface.
        aminus (float): Maximum negative wave speed at the cell interface.
        Fplus (float): Flux contribution from the "plus" (right) side.
        Fminus (float): Flux contribution from the "minus" (left) side.
        Udifference (float): Difference in the conserved variable across the interface 
            (e.g., U_right - U_left).

    Returns:
        float: The computed numerical flux. Returns 0.0 if `(aplus - aminus) == 0.0`.
    """
    numerical_flux = 0.0
    if (aplus - aminus) != 0.0:
        numerical_flux = (aplus * Fminus - aminus * Fplus + aplus * aminus * Udifference) / (aplus - aminus)
    return numerical_flux


@ti.func
def ScalarAntiDissipation(uplus, uminus, aplus, aminus, epsilon):
    """
    Computes an anti-dissipation factor based on local wave speeds and state magnitudes.

    This function calculates a dimensionless ratio `R` that adjusts numerical dissipation
    in a flux-based scheme. The ratio depends on the maximum wave speed (`aplus` or `aminus`)
    and the magnitudes of the state variables `uplus` and `uminus`. If both wave speeds are
    non-zero, a local "Froude-like" number is formed by dividing the larger magnitude of
    `uplus` or `uminus` by the respective wave speed. This number is then augmented by a
    small threshold `epsilon` to yield a final ratio between 0 and 1. If either wave speed
    is zero, the ratio is set to `epsilon`.

    Args:
        uplus (float): The "plus" state or velocity component.
        uminus (float): The "minus" state or velocity component.
        aplus (float): Positive wave speed at the cell interface.
        aminus (float): Negative wave speed at the cell interface.
        epsilon (float): Small threshold to prevent division by zero or extreme values.

    Returns:
        float: Anti-dissipation ratio `R`. A value near 1 indicates lower numerical dissipation,
        while a value near 0 increases dissipation. Defaults to 0 if neither condition applies.
    """
    R = 0.0   # Defaul return if none of the conditions are met
    if (aplus!=0.0 and aminus!=0.0) :
        if ti.abs(uplus)>=ti.abs(uminus):
            Fr = ti.abs(uplus) / aplus
        else:
            Fr = ti.abs(uminus)/aminus
        R = (Fr + epsilon ) / (Fr + 1.0)
    elif (aplus==0.0 or aminus==0.0):
        R = epsilon
    return R


@ti.func
def FrictionCalc(hu, hv, h, base_depth,delta,isManning, g, friction,differentiability):
    """
    Computes a bottom friction term for shallow-water or Boussinesq-type flows.

    This function calculates a friction coefficient based on either a constant friction 
    parameter (`friction`) or Manning's formula (if `isManning == 1`), and then applies it 
    to the momentum components. The water depth `h` is scaled by the `base_depth` for 
    numerical stability and to avoid singularities near dry cells.

    The steps are:
        
        1. Scale the water depth (`h_scaled = h / base_depth`) and compute powers 
           (`h2 = h_scaled^2`, `h4 = h2^2`).
        2. Compute a term `divide_by_h2` which further scales friction based on squared depth.
        3. Ensure the local depth `h` is not below a small threshold `delta` to prevent 
           division by zero.
        4. If `isManning == 1`, convert the `friction` input into Mannings n and compute:
        
               f = g * (friction^2) * (1 / h^(1/3))
               
           otherwise, keep a constant friction value.
        5. Clamp the friction factor to a maximum of 0.5 as a safety measure.
        6. Multiply by the flow speed (computed from `hu`, `hv`) and the scaling factor 
           `divide_by_h2`.

           
    Args:
        hu (float): Momentum in the x-direction.
        hv (float): Momentum in the y-direction.
        h (float): Local water depth.
        base_depth (float): Reference (base) depth for scaling.
        delta (float): Minimum threshold for water depth to avoid division by zero.
        isManning (int): If 1, uses Mannings n formulation; otherwise uses a constant friction.
        g (float): Gravitational acceleration.
        friction (float): Either a constant friction coefficient or Mannings n value 
            depending on `isManning`.

    Returns:
        float: Computed friction term, capped at 0.5, that will be applied to momentum.
    """
    flag=0.0
    h_scaled = h / base_depth
    h2 = h_scaled*h_scaled
    h4 = h2*h2
    divide_by_h2 = 2.0 * h2 / (h4 + ti.max(h4, 1.e-10)) / base_depth / base_depth
    divide_by_h = 1.0 / ti.max(h, delta)
    if differentiability:
        flag=1.0
        divide_by_h = 2.0 * h / (h*h + ti.max(h*h, 1e-10))
    f = friction
    if isManning == 1:
        f = g * ti.pow(friction,2) * ti.pow(ti.abs(divide_by_h),1./3.)
    f = ti.min(f,0.5)    
    return f * ti.sqrt(hu*hu + hv*hv + flag*1e-10) * divide_by_h2


if __name__ == "__main__":
    print('Module of functions used in Celeris')
