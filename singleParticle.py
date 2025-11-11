import pyvista as pv 
import numpy as np
import sympy as sp 
from scipy.optimize import root_scalar
from numba import njit 

doScale_el = "magnitude_el" 
doScale_mag = "magnitude_mag" 
# doScale = "magnitude"
scaleFactor = 5.0


mag_el = 0.0 
mag_mag = 1.0 


dt = 0.1

a = 0.8 
b = 3.0 
B = 0.1

q = 1.0
epsilon0 = 1.0
c = 1.0
factor = (q / (4 * np.pi * epsilon0))


densityx = 15 
limitsx = -3

densityy = 30 
limitsy = -8

densityz = 10 
limitsz = 2



pt = sp.Symbol("pt")



# Circular motion
# px_simp= B * sp.sin(b * pt) 
# py_simp= B * sp.cos(b * pt)
# pz_simp= a * pt 


px_simp = 0.0 * pt  
py_simp = a * pt - 10.0 
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


def findRetardedTime(r, t, ps, vs, max_iter = 100, tol = 1e-8):
    

    def eq_and_deq(tau):
        # Evaluate all functions once per iteration
        rp = np.array([ps[0](tau), ps[1](tau), ps[2](tau)])
        drp = np.array([vs[0](tau), vs[1](tau), vs[2](tau)])
        
        diff = r - rp
        norm_diff = np.linalg.norm(diff)
        
        # eq value
        eq_val = tau + norm_diff / c - t
        
        # deq value
        deq_val = 1 - np.dot(diff, drp) / (c * norm_diff)
        
        return eq_val, deq_val
    
    tau0 = 0.5
    
    for i in range(max_iter):
        fx, dfx = eq_and_deq(tau0)        

        # Check if derivative is zero
        if abs(dfx) < 1e-12 or dfx == np.nan:
            raise ValueError(f"Derivative is zero at x = {x}. Method fails.")
        
        # Newton-Raphson formula: x_new = x - f(x)/f'(x)
        tau1 = tau0 - fx / dfx
        
        # Check for convergence
        if abs(tau1 - tau0) < tol:
            if abs(tau1 - t) < tol:
                print("PELIGRO")
            return tau1
        
        tau0 = tau1

    raise RuntimeError(f"Method did not converge within {max_iter} iterations: {abs(tau0)} > {tol}")


def compute_retarded_times(points, t, ps, vs):
    retT = np.empty(len(points))
    for i, p in enumerate(points):
        retT[i] = findRetardedTime(p, t, ps, vs)
    return retT


def vectorField(points, ps, vs, As, t):
    
    # Taking the points
    x, y, z = points[:, 0], points[:, 1], points[:, 2],

    

    """ Contruction of the electric field """ 
   
    # Calculate retarded times for all points
    retT = compute_retarded_times(points, t, ps, vs) 

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
    vectors_mag = c * np.cross(R_uni, vectors_el, axis = 1)


    vectors_all = np.concatenate([vectors_mag, vectors_el])
    points_all = np.concatenate([points, points])
    
    colors = np.linalg.norm(vectors_all, axis = 1)
    magnitudes = np.clip(colors, None, 1.0)


    magnitudes_el = np.linalg.norm(vectors_el, axis = 1)
    magnitudes_mag = np.clip(np.linalg.norm(vectors_mag, axis = 1), None, 1.0)


    return vectors_el, vectors_mag, magnitudes_el, magnitudes_mag








""" CREATING THE MESH """


x = np.linspace(-limitsx, limitsx, densityx)
y = np.linspace(-limitsy, limitsy, densityy)
z = np.linspace(-limitsz, limitsz, densityz)

X, Y, Z = np.meshgrid(x, y, z)




points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))






# Create the first frame
vectors_el, vectors_mag, magnitudes_el, magnitudes_mag = vectorField(points, ps, vs, As, t=0)
pdata = pv.PolyData(points)
pdata['vectors_el'] = vectors_el       # attach vectors to points
pdata['vectors_mag'] = vectors_mag       # attach vectors to points

pdata['magnitude_el'] = np.clip(magnitudes_el, 0.0, 1.0) * mag_el # attach scalars for color
pdata['magnitude_mag'] = np.clip(magnitudes_mag, 0.0, 1.0) * mag_mag # attach scalars for color


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


plotter = pv.Plotter()
plotter.set_background('black')





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
sphere = pv.Sphere(radius=0.1, center=(0, 0, 0))

# Add it to the scene
act_sphere = plotter.add_mesh(sphere, color="white", specular=0.4, smooth_shading=True)



plotter.show_grid(color = "gray")



# --- Animation callback using timer events ---
def update_field(t):

    
    vectors_el, vectors_mag, magnitudes_el, magnitudes_mag = vectorField(points, ps, vs, As, t)

    pdata['vectors_el'] = vectors_el 
    pdata['vectors_mag'] = vectors_mag 
    pdata['magnitude_el'] = np.clip(magnitudes_el, None, 0.1) * mag_el
    pdata['magnitude_mag'] = np.clip(magnitudes_mag, None, 0.1) * mag_mag
    pdata['col_el'] = 10 - 1 / magnitudes_el 
    pdata['col_mag'] = magnitudes_mag

    new_arrows_el = pdata.glyph(orient='vectors_el', scale=doScale_el, factor=scaleFactor)
    actor_el.mapper.SetInputData(new_arrows_el)
    actor_el.mapper.SetScalarModeToUsePointFieldData()
    actor_el.mapper.SelectColorArray('col_el')


    new_arrows_mag = pdata.glyph(orient='vectors_mag', scale=doScale_mag, factor=scaleFactor)
    actor_mag.mapper.SetInputData(new_arrows_mag)
    
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


