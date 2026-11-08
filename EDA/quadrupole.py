"""
Retarded electric field of oscillating point charges (c = 1).
Dipoles, quadrupoles, or any user-defined set of charges.

Angular-pattern view: the radiation field falls as 1/R, so the field is
multiplied by R and arrow lengths are made strictly proportional to it
(no exponent, no clipping). On a spherical shell or meridian ring, arrow
length then traces |sin(theta)| for a dipole and |sin(2 theta)| for a
linear quadrupole.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import pyvista as pv
import sympy as sp


# =============================================================================
# 1. CONFIGURATION
# =============================================================================
@dataclass
class Config:
    # Field
    softening: float = 1e-3          # avoids 1/R^2 blow-up on a charge
    radiation_only: bool = True     # True: keep only the 1/R acceleration term
    # Arrow scaling
    distance_weighting: float = 0.1  # E -> E * R^p ; p=1 removes the 1/R fall-off
    arrow_length: float = 0.1        # length of the largest arrow (data units)
    normalize_per_frame: bool = False  # False: arrows "breathe" in time, shape is exact
                                       # True : largest arrow always has arrow_length
    envelope_lengths: bool = True    # arrow length = per-point peak over a period

    # Time
    dt: float = 1e-2
    t_start: float = 0.0
    # Oscillation (v_max = amplitude * omega must be < 1)
    amplitude: float = 1e-6
    omega: float = 5.0 * np.pi
    # Mesh: "meridian" | "shell" | "parallel" | "transversal" | "spherical"
    mesh: str = "shell"
    shell_radius: float = 15.0        # outermost shell
    shell_radius_min: float = 10.    # innermost shell
    n_shells: int = 30                # 1 = single shell at shell_radius
    arrow_length: float = 0.5       # reduced so neighbouring shells don't overlap


CFG = Config()

PATTERNS = {
    "sin": lambda th: np.abs(np.sin(th)),
    "sin2": lambda th: np.abs(np.sin(2.0 * th)),
}


# =============================================================================
# 2. PARTICLES / TRAJECTORIES
# =============================================================================
t_sym = sp.Symbol("pt")


def _vectorize(exprs: Sequence[sp.Expr]) -> Callable[[np.ndarray], np.ndarray]:
    """Lambdify 3 sympy expressions into f(t:(N,)) -> (N,3), handling constants."""
    fn = sp.lambdify(t_sym, list(exprs), "numpy")

    def f(t):
        t = np.asarray(t, dtype=float)
        return np.stack(
            [np.broadcast_to(np.asarray(c, dtype=float), t.shape) for c in fn(t)],
            axis=-1,
        )
    return f


@dataclass
class Charge:
    """Point charge with symbolic trajectory r(t)=(x,y,z)."""
    x: sp.Expr
    y: sp.Expr
    z: sp.Expr
    q: float
    pos: Callable = field(init=False, repr=False)
    vel: Callable = field(init=False, repr=False)
    acc: Callable = field(init=False, repr=False)

    def __post_init__(self):
        r = [sp.sympify(c) for c in (self.x, self.y, self.z)]
        v = [sp.diff(c, t_sym) for c in r]
        a = [sp.diff(c, t_sym) for c in v]
        self.pos, self.vel, self.acc = _vectorize(r), _vectorize(v), _vectorize(a)

    def max_speed(self, t0=0.0, t1=50.0, n=4000) -> float:
        return float(np.linalg.norm(self.vel(np.linspace(t0, t1, n)), axis=1).max())


def oscillating_charge(q, center, axis, amp, omega, phase=0.0) -> Charge:
    """Charge oscillating along `axis` around `center`."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    s = amp * sp.cos(omega * t_sym + phase)
    comps = [float(center[i]) + float(axis[i]) * s for i in range(3)]
    return Charge(*comps, q=q)


# ---- Presets (polar axis = z, so theta is the usual polar angle) -------------
def make_dipole(cfg: Config, axis=(0, 0, 1)) -> List[Charge]:
    A, w = cfg.amplitude, cfg.omega
    return [
        oscillating_charge(+1, (0, 0, 0), axis, A, w),
        oscillating_charge(-1, (0, 0, 0), axis, -A, w),
    ]


def make_oscillating_quadrupole(cfg: Config, d=0.05,
                                separation_axis=(0, 0, 1),
                                oscillation_axis=(0, 0, 1)) -> List[Charge]:
    """
    Two antiparallel oscillating dipoles separated by 2d.
      oscillation_axis || separation_axis -> linear quadrupole (|sin 2θ| pattern)
      oscillation_axis  ⟂ separation_axis -> transverse quadrupole (not axisymmetric)
    Keep k*d = omega*d << 1 for a clean quadrupole pattern.
    """
    A, w = cfg.amplitude, cfg.omega
    s = np.asarray(separation_axis, float)
    s /= np.linalg.norm(s)
    charges = []
    for sign in (+1, -1):
        c = sign * d * s
        charges.append(oscillating_charge(+sign, c, oscillation_axis, A, w))
        charges.append(oscillating_charge(-sign, c, oscillation_axis, -A, w))
    return charges


def make_breathing_quadrupole(cfg: Config, d=0.3, axis=(0, 0, 1)) -> List[Charge]:
    """
    Linear (+1, -2, +1) quadrupole whose outer charges breathe in and out:
    +1 at ±(d + A cos wt), -2 fixed at the origin. Zero dipole moment,
    time-varying quadrupole moment Q ~ 4 d A cos(wt)  ->  |sin 2θ| pattern.
    """
    A, w = cfg.amplitude, cfg.omega
    s = np.asarray(axis, float)
    s /= np.linalg.norm(s)
    return [
        oscillating_charge(+1, +d * s, s, A, w),
        oscillating_charge(+1, -d * s, s, -A, w),
        oscillating_charge(-2, (0, 0, 0), s, 0.0, w),
    ]


# =============================================================================
# 3. RETARDED TIME SOLVER
# =============================================================================
def retarded_time(points, t, ch: Charge, tol=1e-11, max_iter=60):
    """
    Solve  f(tau) = tau + |x - r(tau)| - t = 0  for every point.
    For |v|<1, f' = 1 - n.v > 0, so the root is unique: bracket it, run Newton
    from tau = t - |x - r(t)|, and fall back to bisection if a step leaves the
    bracket. Converged points are dropped from later iterations.
    """
    points = np.asarray(points, float)
    N = len(points)
    t_arr = np.full(N, float(t))

    d0 = np.linalg.norm(points - ch.pos(t_arr), axis=1)
    hi = t_arr.copy()
    width = d0 + 1e-3
    lo = t_arr - width
    for _ in range(60):
        f_lo = lo + np.linalg.norm(points - ch.pos(lo), axis=1) - t_arr
        bad = f_lo > 0
        if not bad.any():
            break
        width[bad] *= 2.0
        lo[bad] = t_arr[bad] - width[bad]

    tau = np.clip(t_arr - d0, lo, hi)
    active = np.arange(N)

    for _ in range(max_iter):
        if active.size == 0:
            break
        p, ta = points[active], tau[active]
        diff = p - ch.pos(ta)
        dist = np.maximum(np.linalg.norm(diff, axis=1), 1e-12)
        f = ta + dist - t
        fp = 1.0 - np.sum(diff * ch.vel(ta), axis=1) / dist

        neg = f < 0
        lo[active[neg]] = ta[neg]
        hi[active[~neg]] = ta[~neg]

        with np.errstate(divide="ignore", invalid="ignore"):
            new = ta - f / fp
        l, h = lo[active], hi[active]
        outside = ~np.isfinite(new) | (fp < 1e-12) | (new <= l) | (new >= h)
        new = np.where(outside, 0.5 * (l + h), new)

        tau[active] = new
        done = (np.abs(f) < tol) | (np.abs(new - ta) < tol) | ((h - l) < tol)
        active = active[~done]

    return tau


def retarded_time_chunked(points, t, ch, chunk=20000):
    out = np.empty(len(points))
    for i in range(0, len(points), chunk):
        out[i:i + chunk] = retarded_time(points[i:i + chunk], t, ch)
    return out


# =============================================================================
# 4. FIELD
# =============================================================================
def electric_field(points, charges: Sequence[Charge], t, cfg: Config = CFG):
    """
    Liénard–Wiechert field (c=1):
      E = q [ (n-v)(1-v²) / (κ³R²)  +  n × ((n-v) × a) / (κ³R) ],  κ = 1 - n·v
    The first (velocity) term is dropped if cfg.radiation_only.
    """
    E = np.zeros((len(points), 3))
    for ch in charges:
        tau = retarded_time_chunked(points, t, ch)
        r, v, a = ch.pos(tau), ch.vel(tau), ch.acc(tau)

        Rvec = points - r
        R = np.maximum(np.linalg.norm(Rvec, axis=1), cfg.softening)
        n = Rvec / R[:, None]
        u = n - v
        kappa = 1.0 - np.sum(n * v, axis=1)
        v2 = np.sum(v * v, axis=1)

        acc_term = np.cross(n, np.cross(u, a)) / (kappa**3 * R)[:, None]
        if cfg.radiation_only:
            E += ch.q * acc_term
        else:
            vel_term = (1.0 - v2)[:, None] * u / (kappa**3 * R**2)[:, None]
            E += ch.q * (vel_term + acc_term)
    return E


def scaled_field(points, charges, t, cfg: Config = CFG):
    """
    E weighted by (distance from origin)^p so the 1/R radiation fall-off is
    compensated. Returns (vectors (N,3), magnitudes (N,)).
    """
    E = electric_field(points, charges, t, cfg)
    R0 = np.maximum(np.linalg.norm(points, axis=1), cfg.softening)
    E = E * (R0 ** cfg.distance_weighting)[:, None]
    return E, np.linalg.norm(E, axis=1)


def calibrate(points, charges, cfg: Config = CFG, n_samples=24):
    """
    Sample one oscillation period and return
      ref   : peak of the scaled |E| over all points and times
      shape : scaled |E| / ref at the instant of that peak
      env   : per-point peak of scaled |E| over the period (N,)
    """
    T = 2.0 * np.pi / cfg.omega
    ref, shape = 0.0, None
    env = np.zeros(len(points))
    for k in range(n_samples):
        _, mag = scaled_field(points, charges, cfg.t_start + k * T / n_samples, cfg)
        env = np.maximum(env, mag)
        m = mag.max()
        if m > ref:
            ref, shape = m, mag / m
    return ref, shape, env


def polar_angle(points):
    r = np.maximum(np.linalg.norm(points, axis=1), 1e-12)
    return np.arccos(np.clip(points[:, 2] / r, -1.0, 1.0))


def pattern_rms_error(points, shape, name):
    r = np.linalg.norm(points, axis=1)
    outer = r >= 0.999 * r.max()                  # outermost shell only
    ideal = PATTERNS[name](polar_angle(points[outer]))
    ideal = ideal / ideal.max()
    s = shape[outer] / shape[outer].max()
    return float(np.sqrt(np.mean((s - ideal) ** 2)))

# =============================================================================
# 5. MESH
# =============================================================================
def cartesian_points(nx, ny, nz, lx, ly, lz):
    x, y, z = np.linspace(-lx, lx, nx), np.linspace(-ly, ly, ny), np.linspace(-lz, lz, nz)
    X, Y, Z = np.meshgrid(x, y, z)
    return np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))


def spherical_points(r_max=5, nr=8, ntheta=30, nphi=30):
    r = np.linspace(0, r_max, nr)
    th = np.linspace(0, np.pi, ntheta)
    ph = np.linspace(0, 2 * np.pi, nphi)
    R, TH, PH = np.meshgrid(r, th, ph, indexing="ij")
    return np.column_stack(((R * np.sin(TH) * np.cos(PH)).ravel(),
                            (R * np.sin(TH) * np.sin(PH)).ravel(),
                            (R * np.cos(TH)).ravel()))


def shell_radii(cfg: Config) -> np.ndarray:
    if cfg.n_shells <= 1:
        return np.array([cfg.shell_radius])
    return np.linspace(cfg.shell_radius_min, cfg.shell_radius, cfg.n_shells)


def shell_points(radii, ntheta=14, nphi=24):
    """Concentric spheres (poles excluded). Alternate shells are rotated by half
    a step in phi so arrows on neighbouring shells don't line up radially."""
    th = np.linspace(0, np.pi, ntheta + 2)[1:-1]
    dph = 2 * np.pi / nphi
    out = []
    for i, r0 in enumerate(radii):
        ph = np.arange(nphi) * dph + 0.5 * dph * (i % 2)
        TH, PH = np.meshgrid(th, ph, indexing="ij")
        out.append(np.column_stack(((r0 * np.sin(TH) * np.cos(PH)).ravel(),
                                    (r0 * np.sin(TH) * np.sin(PH)).ravel(),
                                    (r0 * np.cos(TH)).ravel())))
    return np.vstack(out)


def meridian_points(radii, n=48):
    """Concentric rings in the x-z plane (angle from +z is theta). The point
    count scales with radius to keep roughly constant arrow spacing."""
    out = []
    for i, r0 in enumerate(radii):
        ni = max(12, int(round(n * r0 / radii.max())))
        a = np.linspace(0, 2 * np.pi, ni, endpoint=False) + (np.pi / ni) * (i % 2)
        out.append(np.column_stack((r0 * np.sin(a), np.zeros(ni), r0 * np.cos(a))))
    return np.vstack(out)


def build_points(cfg: Config) -> np.ndarray:
    m = cfg.mesh
    if m == "meridian":
        return meridian_points(shell_radii(cfg))
    if m == "shell":
        return shell_points(shell_radii(cfg))
    if m == "parallel":
        return cartesian_points(51, 51, 1, 5.0, 5.0, 0.0)
    if m == "transversal":
        return cartesian_points(31, 51, 51, 1.0, 5.0, 5.0)
    if m == "spherical":
        return spherical_points()
    raise ValueError(f"Unknown mesh mode: {m}")
# =============================================================================
# 6. VISUALIZATION
# =============================================================================
def pattern_curve(name, size, n=200):
    """Closed polar curve  rho = size * f(theta)  in the x-z plane."""
    th = np.linspace(0, 2 * np.pi, n)
    rho = size * PATTERNS[name](th)
    pts = np.column_stack((rho * np.sin(th), np.zeros(n), rho * np.cos(th)))
    return pv.lines_from_points(pts)


class FieldViewer:
    def __init__(self, points, charges, cfg: Config = CFG):
        self.points, self.charges, self.cfg = points, charges, cfg
        self.t = cfg.t_start
        self.running = True

        # Fixed normalization + pattern check
        self.ref, shape, self.env = calibrate(points, charges, cfg) 
        if cfg.pattern and cfg.mesh in ("meridian", "shell"):
            err = pattern_rms_error(points, shape, cfg.pattern)
            print(f"RMS deviation from ideal {cfg.pattern} pattern: {err:.3f} "
                  "(0 = perfect; large -> increase shell_radius or reduce omega*d)")

        self.plotter = pv.Plotter()
        self.plotter.set_background("black")
        self.plotter.show_grid(color="gray")

        self.pdata = pv.PolyData(points)
        self._update_arrays(self.t)
        self.actor = self.plotter.add_mesh(self._glyphs(), scalars="col_el",
                                           cmap="cool", lighting=True, clim=(0.0, 1.0))

        # Ideal pattern overlay (polar plot centred on the charges)
        if cfg.pattern:
            curve = pattern_curve(cfg.pattern, 0.4 * cfg.shell_radius)
            self.plotter.add_mesh(curve, color="yellow", line_width=2)
            self.plotter.add_text(f"yellow: ideal |{cfg.pattern}(theta)| pattern",
                                  font_size=10, color="yellow")

        self.spheres = []
        for ch in charges:
            s = pv.Sphere(radius=0.05, center=ch.pos(np.array([self.t]))[0])
            self.plotter.add_mesh(s, color="red" if ch.q > 0 else "blue",
                                  specular=0.4, smooth_shading=True)
            self.spheres.append(s)

        if cfg.mesh == "meridian":
            self.plotter.view_xz()

        self.plotter.add_key_event("space", self._toggle)
        self.plotter.add_key_event("n", lambda: self._shift(-1.0))
        self.plotter.add_key_event("m", lambda: self._shift(+1.0))

    def _toggle(self):
        self.running = not self.running
        print("Animation running" if self.running else "Animation paused")

    def _shift(self, dt):
        self.t += dt

    def _update_arrays(self, t):
        cfg = self.cfg
        E, mag = scaled_field(self.points, self.charges, t, cfg)
        env_max = self.env.max() if self.env.max() > 0 else 1.0
        if cfg.envelope_lengths:
            length = self.env / env_max                    # static angular pattern
            phase = mag / np.maximum(self.env, 1e-12)      # 0..1, travelling wave
        else:
            ref = mag.max() if cfg.normalize_per_frame else self.ref
            length = mag / (ref if ref > 0 else 1.0)
            phase = length
        self.pdata["vectors_el"] = E
        self.pdata["magnitude_el"] = np.maximum(cfg.arrow_length * length, 1e-6)
        self.pdata["col_el"] = np.clip(phase, 0.0, 1.0)

    def _glyphs(self):
        return self.pdata.glyph(orient="vectors_el", scale="magnitude_el", factor=1.0)

    def _update_frame(self):
        self._update_arrays(self.t)
        self.actor.mapper.SetInputData(self._glyphs())
        self.actor.mapper.SetScalarModeToUsePointFieldData()
        self.actor.mapper.SelectColorArray("col_el")
        for ch, s in zip(self.charges, self.spheres):
            new = ch.pos(np.array([self.t]))[0]
            s.translate(new - np.array(s.center), inplace=True)

    def run(self):
        self.plotter.show(interactive_update=True)
        while not getattr(self.plotter, "_closed", False):
            self._update_frame()
            self.plotter.update()
            self.plotter.render()
            if self.running:
                self.t += self.cfg.dt


# =============================================================================
# 7. MAIN
# =============================================================================
if __name__ == "__main__":
    # ---- Dipole: |sin(theta)| -----------------------------------------------
    # CFG.pattern = "sin"
    # charges = make_dipole(CFG)

    # ---- Linear quadrupole: |sin(2 theta)| (uncomment) ----------------------
    CFG.pattern = "sin2"
    # charges = make_oscillating_quadrupole(CFG, d=0.05)   # keep omega*d << 1
    charges = make_breathing_quadrupole(CFG, d=0.1)     # alternative

    CFG.mesh = "spherical"   # full 3D sphere instead of the x-z ring

    for i, ch in enumerate(charges):
        vmax = ch.max_speed()
        if vmax >= 1.0:
            print(f"WARNING: charge {i} reaches |v|={vmax:.2f} >= c. "
                  "Retarded time is no longer unique; lower amplitude*omega.")

    FieldViewer(build_points(CFG), charges, CFG).run()
