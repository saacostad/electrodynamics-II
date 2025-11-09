import pyvista as pv 
import numpy as np


doScale_el = "magnitude_el" 
doScale_mag = "magnitude_mag" 
scaleFactor = 0.5



def vectorField(points, t):
    
    # Taking the points
    x, y, z = points[:, 0], points[:, 1], points[:, 2],

    """ Definition of the magnetic field field """
    
    s = np.sqrt( np.power(x, 2) + np.power(y, 2) )
    phi = np.arctan2(y, x)

    epsilon0 = 1.0
    mu0 = 1.0 
    q0 = 1.0
    c = 1.0 

    # mag = mu0 * i0 / (2 * np.pi * s) * (c*t) / (np.sqrt( (c*t)**2 - s**2 )) 


    inside = (c*t)**2 - s**2
    valid = inside >= 0  # boolean mask

    mag_mag = np.where(
        valid,
        -mu0 * q0 * s * c / (2 * np.pi) / np.power(inside, 1.5),
        0.0
    )

    u_mag = -mag_mag * np.sin(phi) 
    v_mag = mag_mag * np.cos(phi)
    w_mag = np.zeros_like(u_mag)
    
    

    """ Contruction of the electric field """ 

    mag_el = np.where(
        valid, 
        -(q0 * c) / (2 * np.pi * epsilon0 * np.power( inside, 1.5 ) ),
        0.0
    )

    u_el = np.zeros_like(x)
    v_el = np.zeros_like(y)
    w_el = mag_el 

    # Reconstructing the mesh 
    vectors_mag = np.column_stack((u_mag, v_mag, w_mag))
    vectors_el = np.column_stack((u_el, v_el, w_el))

    vectors_all = np.concatenate([vectors_mag, vectors_el])
    points_all = np.concatenate([points, points])
    
    colors = np.linalg.norm(vectors_all, axis = 1)
    magnitudes = np.clip(colors, None, 1.0)




    magnitudes_el = np.clip(np.linalg.norm(vectors_el, axis = 1), None, 0.3)
    magnitudes_mag = np.clip(np.linalg.norm(vectors_mag, axis = 1), None, 0.8)


    return vectors_el, vectors_mag, magnitudes_el, magnitudes_mag
    # return points_all, vectors_all, magnitudes, colors


""" CREATING THE MESH """


# Radial, angular, and vertical sampling
r = np.linspace(0, 3, 100)       # uniform in radius
theta = np.linspace(0, 2*np.pi, 30)
z = np.linspace(-5, 5, 10)

# Create mesh in cylindrical coordinates
R, THETA, Z = np.meshgrid(r, theta, z, indexing='ij')

# Convert to Cartesian
X = R * np.cos(THETA)
Y = R * np.sin(THETA)




points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))



# Create the first frame
vectors_el, vectors_mag, magnitudes_el, magnitudes_mag = vectorField(points, t=0)
pdata = pv.PolyData(points)
pdata['vectors_el'] = vectors_el       # attach vectors to points
pdata['vectors_mag'] = vectors_mag       # attach vectors to points

pdata['magnitude_el'] = np.sqrt(magnitudes_el)  # attach scalars for color
pdata['magnitude_mag'] = np.sqrt(magnitudes_mag)  # attach scalars for color


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
    scalars='magnitude_el',   # color by this scalar
    cmap='cool',        # choose any Matplotlib colormap
    lighting=True,
)


actor_mag = plotter.add_mesh(
    arrows_mag,
    scalars='magnitude_mag',   # color by this scalar
    cmap='autumn',        # choose any Matplotlib colormap
    lighting=True,
)


plotter.show_grid(color = "gray")

cylinder = pv.Cylinder(
        center = (0, 0, 0),
        direction = (0, 0, 1),
        radius = 0.1,
        height = 15,
        resolution = 15,
        )


cyl_actor = plotter.add_mesh(cylinder, color = "white")

# --- Animation callback using timer events ---
def update_field(t):
    vectors_el, vectors_mag, magnitudes_el, magnitudes_mag = vectorField(points, t)

    pdata['vectors_el'] = vectors_el 
    pdata['vectors_mag'] = vectors_mag 
    pdata['magnitude_el'] = magnitudes_el
    pdata['magnitude_mag'] = magnitudes_mag

    new_arrows_el = pdata.glyph(orient='vectors_el', scale=doScale_el, factor=scaleFactor)
    actor_el.mapper.SetInputData(new_arrows_el)

    new_arrows_mag = pdata.glyph(orient='vectors_mag', scale=doScale_mag, factor=scaleFactor)
    actor_mag.mapper.SetInputData(new_arrows_mag)




# --- Play animation with interactive control ---
plotter.show(interactive_update=True)  # Keeps window responsive


running = True
def toggle_pause():
    """Toggle animation on/off when spacebar is pressed."""
    global running
    running = not running
    print("Animation running:" if running else "Animation paused.")

# Keyboard event for spacebar
plotter.add_key_event("space", toggle_pause)



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
        t += 0.1

