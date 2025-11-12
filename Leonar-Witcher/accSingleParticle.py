import pyvista as pv 
import numpy as np
import sympy as sp 
from scipy.optimize import root_scalar
from numba import njit 
from multiprocessing import Pool, cpu_count
from functools import partial


do_poynting = False


doScale_el = "magnitude_el" 
doScale_mag = "magnitude_mag"
doScale_poynting = "magnitude_poynting"

scaleFactor = 1.0


mag_el = 1.0
mag_mag = 1.0 
mag_poynting = 1.0

dt = 0.1

t0 = 5.0

a = 0.03 
b = 3.0 
B = 0.1

q = 1.0
epsilon0 = 1.0
c = 1.0
factor = (q / (4 * np.pi * epsilon0))
mu0 = 1.0

densityx = 51 
limitsx = 2.0

densityy = 51 
limitsy = 5.0

densityz = 3 
limitsz = 1.0



pt = sp.Symbol("pt")



# Circular motion
# px_simp= B * sp.sin(b * pt) 
# py_simp= B * sp.cos(b * pt)
# pz_simp= a * pt 


px_simp = 0.0 * pt  
py_simp = a * pt * pt - 8.1251241 
pz_simp = 0.0 * pt


vx_simp= sp.diff(px_simp, pt)
vy_simp= sp.diff(py_simp, pt)
vz_simp= sp.diff(pz_simp, pt)

Ax_simp= sp.diff(vx_simp, pt)
Ay_simp= sp.diff(vy_simp, pt)
Az_simp= sp.diff(vz_simp, pt)



px = sp.lambdify((pt),px_simp)
py = sp.lambdify((pt),py_simp)
pz = sp.lambdify((pt),pz_simp)


vx = sp.lambdify((pt),vx_simp)
vy = sp.lambdify((pt),vy_simp)
vz = sp.lambdify((pt),vz_simp)


Ax = sp.lambdify((pt),Ax_simp)
Ay = sp.lambdify((pt),Ay_simp)
Az = sp.lambdify((pt),Az_simp)


ps = [px, py, pz]
vs = [vx, vy, vz]
As = [Ax, Ay, Az]


def eval_position(ps, retT):
    """Evaluate position functions, handling constants properly."""
    result = []
    for p in ps:
        val = p(retT)
        if np.isscalar(val):
            val = np.full_like(retT, val, dtype=float)
        result.append(val)
    return np.column_stack(result)


def findRetardedTime(points, t, ps, vs, max_iter=100, tol=1e-8):
    """
    Vectorized Newton-Raphson for all points simultaneously.
    """
    N = len(points)
    
    # Initial guess for all points
    tau = np.random.normal(0.0, 1.0, N)
    tau_new = np.zeros_like(tau)

    for iteration in range(max_iter):
        # Evaluate position and velocity at current tau for all points

        rp = eval_position(ps, tau)
        drp = eval_position(vs, tau)
        
        # Vectorized calculations
        diff = points - rp  # (N, 3)
        norm_diff = np.linalg.norm(diff, axis=1)  # (N,)
        
        # Equation and derivative
        eq_val = tau + norm_diff / c - t
        deq_val = 1 - np.sum(diff * drp, axis=1) / (c * norm_diff)
        
        # Check for problematic derivatives
        bad_deriv = np.abs(deq_val) < 1e-12
        if np.any(bad_deriv):
            # Add small perturbation to avoid division issues
            deq_val[bad_deriv] = 1e-12
        
        # Newton-Raphson update
        tau_new = tau - eq_val / deq_val
        
        # Check convergence
        converged = np.abs(tau_new - tau) < tol
        if np.all(converged):
            return tau_new
        
        tau = tau_new
     
    # Check which points didn't converge
    final_error = np.abs(tau_new - tau)
    if np.any(final_error > tol):
        print(f"Warning: {np.sum(final_error > tol)} points did not converge")
    
    return tau


def compute_retarded(points, t, ps, vs, chunk_size=5000):
    """
    Process points in chunks to balance memory and speed.
    """
    N = len(points)
    

    retT = np.empty(N)
    for i in range(0, N, chunk_size):
        end_idx = min(i + chunk_size, N)
        chunk = points[i:end_idx]
        retT[i:end_idx] = findRetardedTime(chunk, t, ps, vs)
    
    return retT


def vectorField(points, ps, vs, As, t):
    

    """ Contruction of the electric field """ 
   
    # Calculate retarded times for all points
    retT = compute_retarded(points, t, ps, vs) 

    # Retarded positions and velocities
    ptr = eval_position(ps, retT)
    vtr = eval_position(vs, retT)
    atr = eval_position(As, retT)


    # Retarded displacement vector and related quantities
    R_vec = points - ptr  # R = x - r(τ)
    R = np.linalg.norm(R_vec, axis=1)  # |R|
    R_uni = R_vec / R[:, np.newaxis]  # n = R/|R|
    

    # Retarded velocity squared
    v2 = np.sum(vtr**2, axis = 1)
    
    # Vector u 
    u = c * R_uni - vtr

    # Variable outside factor 
    var_out_fac = R / np.power( np.sum(R_vec * u, axis = 1) , 3)

    # Velocity term 
    vel_term = (c**2 - v2)[:, np.newaxis] * u 
    
    # Aceleration term 
    acc_term = np.cross( R_vec, np.cross( u, atr ) ) 

    # Total term 
    total = var_out_fac[:, np.newaxis] * (vel_term + acc_term)

    # Electric field components
    rx = factor * total[:, 0]
    ry = factor * total[:, 1]  
    rz = factor * total[:, 2]

    """ RETURN OF VALUES: DO NOT TOUCH """
    
    # Reconstructing the mesh 
    vectors_el = np.column_stack((rx, ry, rz))
    vectors_mag = 1.0 / c * np.cross(R_uni, vectors_el, axis = 1)

    
    magnitudes_el = np.linalg.norm(vectors_el, axis = 1)
    magnitudes_mag = np.linalg.norm(vectors_mag, axis = 1)

    if do_poynting:
        vectors_poynting = (1.0 / mu0) * np.cross(vectors_el, vectors_mag, axis = 1)
        magnitudes_poynting = np.linalg.norm(vectors_poynting, axis = 1)
    else:
        vectors_poynting = np.zeros_like(vectors_el)
        magnitudes_poynting = np.zeros_like(magnitudes_el)

    
    return vectors_el, vectors_mag, vectors_poynting, magnitudes_el, magnitudes_mag, magnitudes_poynting








""" CREATING THE MESH """


x = np.linspace(-limitsx, limitsx, densityx)
y = np.linspace(-limitsy, limitsy, densityy)
z = np.linspace(-limitsz, limitsz, densityz)

X, Y, Z = np.meshgrid(x, y, z)




points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


# Create the first frame

vectors_el, vectors_mag, vectors_poynting, magnitudes_el, magnitudes_mag, magnitudes_poynting = vectorField(points, ps, vs, As, t=0)


plotter = pv.Plotter()
plotter.set_background('black')

pdata = pv.PolyData(points)


if do_poynting:

    pdata['magnitude_poynting'] = np.clip(magnitudes_poynting, None, 1.0) * mag_poynting # attach scalars for color
    pdata['vectors_poynting'] = vectors_poynting       # attach vectors to points
    pdata['col_poynting'] = magnitudes_poynting  # attach scalars for color

    arrows_poynting = pdata.glyph(
        orient='vectors_poynting',      # use the 'vectors' array for direction
        scale=False,           # don't scale arrows by magnitude (optional)
        factor=1.0             # global scaling factor for arrow size
    )    

    actor_poynting = plotter.add_mesh(
        arrows_poynting,
        scalars='col_poynting',   # color by this scalar
        cmap='rainbow',        # choose any Matplotlib colormap
        lighting=True,
    )

else:

    pdata['vectors_el'] = vectors_el       # attach vectors to points
    pdata['vectors_mag'] = vectors_mag       # attach vectors to points


    pdata['magnitude_el'] = np.power(np.clip(magnitudes_el, None, 1.0), 4) * mag_el # attach scalars for color
    pdata['magnitude_mag'] = np.power(np.clip(magnitudes_mag, None, 1.0), 4) * mag_mag # attach scalars for color


    pdata['col_el'] = magnitudes_el  # attach scalars for color
    pdata['col_mag'] = magnitudes_mag  # attach scalars for color


    arrows_el = pdata.glyph(
        orient='vectors_el',      # use the 'vectors' array for direction
        scale=False,           # don't scale arrows by magnitude (optional)
        factor=1.0             # global scaling factor for arrow size
    )


    arrows_mag = pdata.glyph(
        orient='vectors_mag',      # use the 'vectors' array for direction
        scale=False,           # don't scale arrows by magnitude (optional)
        factor=1.0             # global scaling factor for arrow size
    )

    actor_el = plotter.add_mesh(
        arrows_el,
        scalars='col_el',   # color by this scalar
        cmap='cool',        # choose any Matplotlib colormap
        lighting=True,
    )

    actor_mag = plotter.add_mesh(
        arrows_mag,
        scalars='col_mag',   # color by this scalar
        cmap='autumn',        # choose any Matplotlib colormap
        lighting=True,
    )



# Create a sphere (ball)
sphere = pv.Sphere(radius=0.05, center=(0, 0, 0))

# Add it to the scene
act_sphere = plotter.add_mesh(sphere, color="red", specular=0.4, smooth_shading=True)

plotter.show_grid(color = "gray")



# --- Animation callback using timer events ---
def update_field(t):

    vectors_el, vectors_mag, vectors_poynting, magnitudes_el, magnitudes_mag, magnitudes_poynting = vectorField(points, ps, vs, As, t)


    if do_poynting:  
        cons = 0.3
        pdata['magnitude_poynting'] = np.power(np.clip(magnitudes_poynting, None, 1.0), cons)
        pdata['col_poynting'] = magnitudes_poynting / np.partition(magnitudes_poynting, -5)[-5] * 255.0
        pdata['vectors_poynting'] = vectors_poynting
        new_arrows_poynting = pdata.glyph(orient='vectors_poynting', scale=doScale_poynting, factor=scaleFactor)
        actor_poynting.mapper.SetInputData(new_arrows_poynting)
        actor_poynting.mapper.SetScalarModeToUsePointFieldData()
        actor_poynting.mapper.SelectColorArray('col_poynting')
    else:
        pdata['vectors_el'] = vectors_el 
        pdata['vectors_mag'] = vectors_mag 

        cons = 0.6
        pdata['magnitude_el'] = np.power(np.clip(magnitudes_el, None, 1.0), cons) * mag_el # attach scalars for color
        pdata['magnitude_mag'] = np.power(np.clip(magnitudes_mag, None, 1.0), cons) * mag_mag # attach scalars for color

        pdata['col_el'] = magnitudes_el / np.partition(magnitudes_el, -5)[-5]
        pdata['col_mag'] = magnitudes_mag / np.partition(magnitudes_mag, -5)[-5]

        
        new_arrows_el = pdata.glyph(orient='vectors_el', scale=doScale_el, factor=scaleFactor)
        actor_el.mapper.SetInputData(new_arrows_el)
        actor_el.mapper.SetScalarModeToUsePointFieldData()
        actor_el.mapper.SelectColorArray('col_el')

        new_arrows_mag = pdata.glyph(orient='vectors_mag', scale=doScale_mag, factor=scaleFactor)
        actor_mag.mapper.SetInputData(new_arrows_mag)
        actor_mag.mapper.SetScalarModeToUsePointFieldData()
        actor_mag.mapper.SelectColorArray('col_mag')



    # --- Move the sphere properly ---
    new_center = np.array([px(t), py(t), pz(t)])
    displacement = new_center - sphere.center

    sphere.translate(displacement, inplace=True)  # moves geometry


# --- Play animation with interactive control ---
plotter.show(interactive_update=True)  # Keeps window responsive


running = True
def toggle_pause():
    """Toggle animation on/off when spacebar is pressed."""
    global running
    running = not running
    print("Animation running:" if running else "Animation paused.")


t = 0.0

def past():
    global t
    t -= 1.0
def future():
    global t
    t += 1.0


# Keyboard event for spacebar
plotter.add_key_event("space", toggle_pause)
plotter.add_key_event("n", past)
plotter.add_key_event("m", future)



while True:
    update_field(t)
    plotter.update()
    plotter.render()

    if running:
        t += dt


