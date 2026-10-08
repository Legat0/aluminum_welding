# 2D longitudinal weld-pool FEM model (scikit-fem).
# Heat, melt convection, solidification. No EM.
# Moving Gaussian surface flux; Darcy mush (He 2003).
# Default properties: Al 5182 (Zhao & DebRoy 2001).
# Mesh coordinate y is plate thickness (z in the notebooks).

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.sparse import bmat
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    FacetBasis,
    Functional,
    LinearForm,
    MeshTri,
    asm,
    condense,
    solve,
)
from skfem.helpers import ddot, dot, grad, mul, sym_grad
from skfem.models.general import divergence


SIGMA_SB = 5.670374419e-8


@dataclass
class Material:
    T_solidus: float = 850.0
    T_liquidus: float = 911.0
    rho: float = 2300.0
    cp_s: float = 900.0
    cp_l: float = 1200.0
    k_s: float = 168.0
    k_l: float = 108.0
    mu: float = 1.1e-3
    dsigma_dT: float = -3.5e-4
    latent: float = 3.95e5
    beta: float = 1.0e-4
    T_ref: float = 911.0
    mush_C: float = 1.0e8
    mush_B: float = 1.0e-3


@dataclass
class Process:
    Lx: float = 0.040
    Lz: float = 0.006
    nx: int = 40
    nz: int = 12
    q_max: float = 6.0e7
    r_b: float = 0.003
    v_weld: float = 0.005
    x0: float = 0.008
    T0: float = 300.0
    Ta: float = 300.0
    h_conv: float = 20.0
    emissivity: float = 0.4
    g: float = 9.81
    enable_flow: bool = True
    dt: float = 5.0e-3
    dt_max: float = 5.0e-3
    cfl: float = 0.4
    t_end: float = 2.0
    picard_iter: int = 2
    intorder: int = 3
    enable_inertia: bool = False  # Stokes OK on coarse grids; True adds (u·∇)u + SUPG


@dataclass
class Diagnostics:
    time: list = field(default_factory=list)
    pool_depth: list = field(default_factory=list)
    pool_length: list = field(default_factory=list)
    T_probe: list = field(default_factory=list)
    heat_in: list = field(default_factory=list)
    heat_loss: list = field(default_factory=list)
    energy_store: list = field(default_factory=list)
    max_speed: list = field(default_factory=list)
    pe_number: list = field(default_factory=list)
    marangoni_corr: list = field(default_factory=list)


def _liquid_fraction_arr(T, Ts, Tl):
    return np.clip((T - Ts) / (Tl - Ts), 0.0, 1.0)


def _darcy_arr(fL, C, B):
    return C * (1.0 - fL) ** 2 / (fL**3 + B)


def _supg_tau(speed, h, dt, diffus):
    """Tezduyar-style SUPG intrinsic time; overflow-safe."""
    s = np.minimum(np.asarray(speed, dtype=np.float64), 50.0)
    hh = np.maximum(np.asarray(h, dtype=np.float64), 1e-12)
    a = (2.0 / max(dt, 1e-12)) ** 2
    b = (2.0 * s / hh) ** 2
    c = (4.0 * diffus / hh**2) ** 2
    return 1.0 / np.sqrt(np.maximum(a + b + c, 1e-30))


class WeldPoolFEM:
    """Taylor–Hood (P2–P1) velocity–pressure + P1 temperature weld-pool solver."""

    def __init__(self, material: Optional[Material] = None, process: Optional[Process] = None):
        self.mat = material or Material()
        self.proc = process or Process()
        self._build_mesh()
        self.reset()

    def _build_mesh(self) -> None:
        p = self.proc
        x = np.linspace(0.0, p.Lx, p.nx + 1)
        y = np.linspace(0.0, p.Lz, p.nz + 1)
        mesh = MeshTri.init_tensor(x, y)
        self.mesh = mesh.with_boundaries(
            {
                "left": lambda xi: np.isclose(xi[0], 0.0),
                "right": lambda xi: np.isclose(xi[0], p.Lx),
                "bottom": lambda xi: np.isclose(xi[1], 0.0),
                "top": lambda xi: np.isclose(xi[1], p.Lz),
            }
        )
        io = p.intorder
        self.basis_u = Basis(self.mesh, ElementVector(ElementTriP2()), intorder=io)
        self.basis_p = Basis(self.mesh, ElementTriP1(), intorder=io)
        self.basis_T = Basis(self.mesh, ElementTriP1(), intorder=io)
        # Same quadrature as basis_T / basis_u for field coupling
        self.basis_u_q = Basis(self.mesh, ElementVector(ElementTriP2()), intorder=io)
        self.fb_u_top = FacetBasis(
            self.mesh, ElementVector(ElementTriP2()), facets=self.mesh.boundaries["top"], intorder=io
        )
        self.fb_T_top = FacetBasis(
            self.mesh, ElementTriP1(), facets=self.mesh.boundaries["top"], intorder=io
        )

        D_wall = self.basis_u.get_dofs(["left", "right", "bottom"]).all()
        D_top = self.basis_u.get_dofs("top")
        D_uy_top = np.concatenate([D_top.nodal["u^2"], D_top.facet["u^2"]])
        self.D_u = np.unique(np.concatenate([D_wall, D_uy_top]))
        self.D_T = self.basis_T.get_dofs(["left", "right", "bottom"]).all()
        # Pin one pressure DOF (index in full u–p vector)
        self.p_pin = int(self.basis_u.N)
        self.D_up = np.unique(np.concatenate([self.D_u, [self.p_pin]]))

        # Probe near arc start, mid-thickness
        locs = self.basis_T.doflocs
        target = np.array([p.x0, 0.5 * p.Lz])
        self.probe_dof = int(np.argmin(np.sum((locs - target[:, None]) ** 2, axis=0)))

        # Precompute divergence block (constant)
        self._B = asm(divergence, self.basis_u, self.basis_p)

        # Sampling grid (cell centres) for plots / pool metrics
        dx = p.Lx / p.nx
        dz = p.Lz / p.nz
        self.xc = (np.arange(p.nx) + 0.5) * dx
        self.yc = (np.arange(p.nz) + 0.5) * dz
        self.X, self.Y = np.meshgrid(self.xc, self.yc, indexing="ij")
        self.Z = self.Y  # alias for notebook compatibility (thickness)
        # Top-surface sample stations (for Marangoni checks)
        self.x_top = self.xc.copy()
        self.y_top = np.full_like(self.x_top, p.Lz)

    def reset(self) -> None:
        p = self.proc
        self.t = 0.0
        self.T = np.full(self.basis_T.N, p.T0, dtype=np.float64)
        self.u = self.basis_u.zeros()
        self.p = self.basis_p.zeros()
        self.fL = np.zeros(self.basis_T.N, dtype=np.float64)
        self.T_max = self.T.copy()
        self.diag = Diagnostics()
        self._H0 = self._total_enthalpy()
        self._Q_cum = 0.0
        self._Q_loss_cum = 0.0
        self._Q_dirichlet_cum = 0.0

    # --- material helpers -------------------------------------------------

    def liquid_fraction(self, T: np.ndarray) -> np.ndarray:
        return _liquid_fraction_arr(T, self.mat.T_solidus, self.mat.T_liquidus)

    def arc_centre(self, t: Optional[float] = None) -> float:
        if t is None:
            t = self.t
        return self.proc.x0 + self.proc.v_weld * t

    def _cp_ref(self) -> float:
        return 0.5 * (self.mat.cp_s + self.mat.cp_l)

    # --- diagnostics ------------------------------------------------------

    def _eval_T_at(self, pts: np.ndarray) -> np.ndarray:
        """Evaluate temperature at points shape (2, N)."""
        from scipy.interpolate import LinearNDInterpolator

        locs = self.basis_T.doflocs
        interp = LinearNDInterpolator(locs.T, self.T)
        vals = interp(pts.T)
        return np.nan_to_num(vals, nan=self.proc.T0)

    def _eval_u_at(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.interpolate import LinearNDInterpolator

        # P2 nodal DOFs at mesh vertices
        ux = self.u[self.basis_u.nodal_dofs[0]]
        uy = self.u[self.basis_u.nodal_dofs[1]]
        locs = self.mesh.p
        ix = LinearNDInterpolator(locs.T, ux)
        iy = LinearNDInterpolator(locs.T, uy)
        pts_t = pts.T
        return (
            np.nan_to_num(ix(pts_t), nan=0.0),
            np.nan_to_num(iy(pts_t), nan=0.0),
        )

    def sample_fields(self):
        """Return T, fL, u, w on the tensor sampling grid (nx, nz)."""
        pts = np.vstack([self.X.ravel(), self.Y.ravel()])
        T = self._eval_T_at(pts).reshape(self.X.shape)
        fL = self.liquid_fraction(T)
        u, w = self._eval_u_at(pts)
        return T, fL, u.reshape(self.X.shape), w.reshape(self.X.shape)

    def _total_enthalpy(self) -> float:
        m = self.mat
        p = self.proc
        cp = self._cp_ref()

        @Functional
        def enthalpy(w):
            fL = _liquid_fraction_arr(w["T"].value, m.T_solidus, m.T_liquidus)
            return m.rho * (cp * (w["T"].value - p.T0) + fL * m.latent)

        return float(asm(enthalpy, self.basis_T, T=self.basis_T.interpolate(self.T)))

    def pool_metrics(self) -> Tuple[float, float]:
        T, _, _, _ = self.sample_fields()
        melt = T >= self.mat.T_liquidus
        if not np.any(melt):
            return 0.0, 0.0
        melted_cols = np.any(melt, axis=1)
        depth = 0.0
        for i in np.where(melted_cols)[0]:
            ks = np.where(melt[i, :])[0]
            if ks.size:
                depth = max(depth, self.proc.Lz - float(self.yc[ks.min()]))
        dx = self.proc.Lx / self.proc.nx
        length = float(np.sum(melted_cols) * dx)
        return depth, length

    def peclet(self) -> float:
        T, fL, u, w = self.sample_fields()
        speed = np.sqrt(u**2 + w**2)
        melt = fL > 0.5
        if not np.any(melt):
            return 0.0
        U = float(np.max(speed[melt]))
        depth, length = self.pool_metrics()
        Lchar = max(depth, 0.5 * length, self.proc.Lx / self.proc.nx)
        alpha = self.mat.k_l / (self.mat.rho * self.mat.cp_l)
        return U * Lchar / alpha

    def max_speed(self) -> float:
        return self._nodal_speed_max()

    def solid_max_speed(self) -> float:
        _, fL, u, w = self.sample_fields()
        speed = np.sqrt(u**2 + w**2)
        solid = fL < 1e-6
        if not np.any(solid):
            return 0.0
        return float(np.max(speed[solid]))

    def marangoni_surface_means(self) -> Tuple[float, float]:
        """Mean top-surface ux left/right of arc on molten vertices."""
        D = self.basis_u.get_dofs("top")
        ux_dofs = D.nodal["u^1"]
        xd = self.basis_u.doflocs[0, ux_dofs]
        yd = self.basis_u.doflocs[1, ux_dofs]
        uvals = self.u[ux_dofs]
        # temperature at the same vertices (P1 shares vertex DOF order)
        # find nearest T dof for each top vertex
        Tlocs = self.basis_T.doflocs
        molten = []
        for i, (x, y) in enumerate(zip(xd, yd)):
            j = int(np.argmin((Tlocs[0] - x) ** 2 + (Tlocs[1] - y) ** 2))
            molten.append(self.T[j] > self.mat.T_solidus)
        molten = np.asarray(molten, dtype=bool)
        xc = self.arc_centre()
        left = (xd < xc) & molten
        right = (xd > xc) & molten
        u_left = float(uvals[left].mean()) if np.any(left) else float("nan")
        u_right = float(uvals[right].mean()) if np.any(right) else float("nan")
        return u_left, u_right

    def marangoni_outflow_corr(self) -> float:
        """Mean ux * sign(x - x_arc) on molten top vertices; >0 => outflow if dσ/dT<0."""
        D = self.basis_u.get_dofs("top")
        ux_dofs = D.nodal["u^1"]
        xd = self.basis_u.doflocs[0, ux_dofs]
        yd = self.basis_u.doflocs[1, ux_dofs]
        uvals = self.u[ux_dofs]
        Tlocs = self.basis_T.doflocs
        molten = np.zeros(len(ux_dofs), dtype=bool)
        for i, (x, y) in enumerate(zip(xd, yd)):
            j = int(np.argmin((Tlocs[0] - x) ** 2 + (Tlocs[1] - y) ** 2))
            molten[i] = self.T[j] > self.mat.T_solidus
        if not np.any(molten):
            return float("nan")
        xc = self.arc_centre()
        return float(np.mean(uvals[molten] * np.sign(xd[molten] - xc)))

    def heat_balance_ratio(self) -> float:
        dH = self._total_enthalpy() - self._H0
        num = abs(self._Q_cum - self._Q_loss_cum - self._Q_dirichlet_cum - dH)
        den = max(abs(self._Q_cum), 1.0)
        return float(num / den)

    def suggest_dt(self) -> float:
        # Backward-Euler FEM: no diffusion CFL. Cap by advection CFL when molten.
        p = self.proc
        umax = max(self._nodal_speed_max(), p.v_weld, 1e-6)
        h = min(p.Lx / p.nx, p.Lz / p.nz)
        dt_cfl = p.cfl * h / umax
        return float(min(p.dt_max, p.dt, dt_cfl))

    def _nodal_speed_max(self) -> float:
        ux = self.u[self.basis_u.nodal_dofs[0]]
        uy = self.u[self.basis_u.nodal_dofs[1]]
        return float(np.max(np.sqrt(ux**2 + uy**2)))

    # --- time stepping ----------------------------------------------------

    def step(self, dt: Optional[float] = None) -> float:
        if dt is None:
            dt = self.suggest_dt()
        if self.proc.enable_flow:
            self._step_momentum(dt)
        else:
            self.u[:] = 0.0
            self.p[:] = 0.0
        self._step_energy(dt)
        self.fL = self.liquid_fraction(self.T)
        self._freeze_solid_velocity()
        self.T_max = np.maximum(self.T_max, self.T)
        self.t += dt
        return dt

    def _freeze_solid_velocity(self) -> None:
        """Zero velocity where both adjacent nodal temperatures are solid."""
        fL_nodes = self.liquid_fraction(self.T)
        # Map P1 nodal solid flag onto P2 velocity nodes via mesh vertices
        solid_vertex = fL_nodes < 1e-6
        # nodal_dofs share vertex ordering with P1 for vertices
        n_verts = self.mesh.p.shape[1]
        if solid_vertex.shape[0] >= n_verts:
            solid_v = solid_vertex[:n_verts]
            self.u[self.basis_u.nodal_dofs[0, solid_v]] = 0.0
            self.u[self.basis_u.nodal_dofs[1, solid_v]] = 0.0

    def _step_momentum(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        rho, mu = m.rho, m.mu
        Ts, Tl = m.T_solidus, m.T_liquidus
        C, B = m.mush_C, m.mush_B
        nu = mu / rho
        u_cap = 2.0  # m/s — physical Marangoni order; prevents blow-up

        # Skip expensive flow solve while fully solid
        if float(np.max(self.T)) < self.mat.T_solidus + 1.0:
            self.u[:] = 0.0
            self.p[:] = 0.0
            return

        u_prev = self.u.copy()
        u_star = self.u.copy()
        T_at_u = self.basis_T.interpolate(self.T)

        @BilinearForm
        def momentum(u, v, w):
            us = w["u_star"]
            Tf = w["T"]
            fL = _liquid_fraction_arr(Tf.value, Ts, Tl)
            A_d = _darcy_arr(fL, C, B)
            out = (rho / dt) * dot(u, v)
            out = out + 2.0 * mu * ddot(sym_grad(u), sym_grad(v))
            out = out + A_d * dot(u, v)
            if p.enable_inertia:
                adv_u = mul(grad(u), us)
                out = out + rho * dot(adv_u, v)
                speed = np.sqrt(us[0] ** 2 + us[1] ** 2) + 1e-12
                tau = _supg_tau(speed, w.h, dt, nu)
                adv_v = mul(grad(v), us)
                out = out + tau * rho * dot(adv_u, adv_v)
            return out

        @LinearForm
        def momentum_load(v, w):
            Tf = w["T"]
            up = w["u_prev"]
            out = (rho / dt) * dot(up, v)
            out = out + rho * p.g * m.beta * (Tf.value - m.T_ref) * v[1]
            return out

        @LinearForm
        def marangoni(v, w):
            Tf = w["T"]
            fL = _liquid_fraction_arr(Tf.value, Ts, Tl)
            return fL * m.dsigma_dT * Tf.grad[0] * v[0]

        for _ in range(p.picard_iter):
            u_star_f = self.basis_u.interpolate(u_star)
            A = asm(momentum, self.basis_u, u_star=u_star_f, T=T_at_u)
            K = bmat([[A, -self._B.T], [-self._B, None]], format="csr")
            f_u = asm(
                momentum_load,
                self.basis_u,
                T=T_at_u,
                u_prev=self.basis_u.interpolate(u_prev),
            )
            T_top = self.fb_T_top.interpolate(self.T)
            f_u = f_u + asm(marangoni, self.fb_u_top, T=T_top)
            f = np.concatenate([f_u, self.basis_p.zeros()])
            uvp = solve(*condense(K, f, D=self.D_up))
            u_new = uvp[: self.basis_u.N].copy()
            if p.enable_inertia:
                u_star = 0.5 * u_star + 0.5 * u_new
            else:
                u_star = u_new
            speed = np.sqrt(
                u_star[self.basis_u.nodal_dofs[0]] ** 2
                + u_star[self.basis_u.nodal_dofs[1]] ** 2
            )
            peak = float(np.max(speed)) if speed.size else 0.0
            if peak > u_cap:
                u_star *= u_cap / peak
            self.p = uvp[self.basis_u.N :].copy()

        self.u = np.clip(u_star, -u_cap, u_cap)

    def _step_energy(self, dt: float) -> None:
        m = self.mat
        pr = self.proc
        rho = m.rho
        cp_ref = self._cp_ref()
        Ts, Tl = m.T_solidus, m.T_liquidus
        dTm = Tl - Ts
        L = m.latent
        ks, kl = m.k_s, m.k_l
        alpha_l = kl / (rho * m.cp_l)

        T_prev = self.T.copy()
        # Cap advection speed for energy SUPG stability on coarse grids
        u_adv = self.u.copy()
        u_cap_e = 0.5
        spd = self._nodal_speed_max()
        if spd > u_cap_e:
            u_adv *= u_cap_e / spd
        u_field = self.basis_u_q.interpolate(u_adv)
        xc = self.arc_centre(self.t)

        @BilinearForm
        def energy(T, v, w):
            Tf = w["T_prev"]  # for property lag
            vel = w["vel"]
            fL = _liquid_fraction_arr(Tf.value, Ts, Tl)
            in_mush = (Tf.value > Ts) & (Tf.value < Tl)
            cp_eff = cp_ref + np.where(in_mush, L / dTm, 0.0)
            k = ks * (1.0 - fL) + kl * fL
            out = (rho * cp_eff / dt) * T * v
            out = out + k * dot(grad(T), grad(v))
            adv = vel[0] * grad(T)[0] + vel[1] * grad(T)[1]
            out = out + rho * cp_ref * adv * v
            # SUPG
            speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2) + 1e-12
            tau = _supg_tau(speed, w.h, dt, alpha_l)
            adv_v = vel[0] * grad(v)[0] + vel[1] * grad(v)[1]
            out = out + tau * rho * cp_ref * adv * adv_v
            return out

        @LinearForm
        def energy_load(v, w):
            Tf = w["T_prev"]
            fL = _liquid_fraction_arr(Tf.value, Ts, Tl)
            in_mush = (Tf.value > Ts) & (Tf.value < Tl)
            cp_eff = cp_ref + np.where(in_mush, L / dTm, 0.0)
            return (rho * cp_eff / dt) * Tf.value * v

        @LinearForm
        def surface_flux(v, w):
            Tf = w["T"]
            x = w.x[0]
            q_in = pr.q_max * np.exp(-3.0 * (x - xc) ** 2 / pr.r_b**2)
            q_loss = pr.h_conv * (Tf.value - pr.Ta) + pr.emissivity * SIGMA_SB * (
                Tf.value**4 - pr.Ta**4
            )
            return (q_in - q_loss) * v

        @Functional
        def integ_qin(w):
            x = w.x[0]
            return pr.q_max * np.exp(-3.0 * (x - xc) ** 2 / pr.r_b**2)

        @Functional
        def integ_qloss(w):
            Tf = w["T"]
            return pr.h_conv * (Tf.value - pr.Ta) + pr.emissivity * SIGMA_SB * (
                Tf.value**4 - pr.Ta**4
            )

        T_prev_f = self.basis_T.interpolate(T_prev)
        A = asm(energy, self.basis_T, T_prev=T_prev_f, vel=u_field)
        f = asm(energy_load, self.basis_T, T_prev=T_prev_f)
        T_top = self.fb_T_top.interpolate(T_prev)
        f = f + asm(surface_flux, self.fb_T_top, T=T_top)

        self.T = solve(*condense(A, f, D=self.D_T, x=np.full(self.basis_T.N, pr.T0)))

        q_in = float(asm(integ_qin, self.fb_T_top))
        q_loss = float(asm(integ_qloss, self.fb_T_top, T=T_top))
        self._Q_cum += q_in * dt
        self._Q_loss_cum += q_loss * dt
        self._Q_dirichlet_cum += self._wall_heat_out() * dt

        # clip extreme temperatures
        self.T = np.clip(self.T, 200.0, 3500.0)

    def _wall_heat_out(self) -> float:
        """Conductive heat leaving through left/right/bottom (FacetBasis)."""
        m = self.mat
        Ts, Tl = m.T_solidus, m.T_liquidus
        ks, kl = m.k_s, m.k_l
        fb = FacetBasis(
            self.mesh,
            ElementTriP1(),
            facets=np.concatenate(
                [
                    self.mesh.boundaries["left"],
                    self.mesh.boundaries["right"],
                    self.mesh.boundaries["bottom"],
                ]
            ),
            intorder=self.proc.intorder,
        )

        @Functional
        def qwall(w):
            Tf = w["T"]
            fL = _liquid_fraction_arr(Tf.value, Ts, Tl)
            k = ks * (1.0 - fL) + kl * fL
            # heat flux out of domain: -(-k grad T · n) wait: Fourier q = -k∇T,
            # outward flux = q·n = -k ∇T·n. Heat leaving domain is +outward flux.
            return -k * (Tf.grad[0] * w.n[0] + Tf.grad[1] * w.n[1])

        return float(asm(qwall, fb, T=fb.interpolate(self.T)))

    def record(self) -> None:
        depth, length = self.pool_metrics()
        self.diag.time.append(self.t)
        self.diag.pool_depth.append(depth)
        self.diag.pool_length.append(length)
        self.diag.T_probe.append(float(self.T[self.probe_dof]))
        self.diag.heat_in.append(self._Q_cum)
        self.diag.heat_loss.append(self._Q_loss_cum + self._Q_dirichlet_cum)
        self.diag.energy_store.append(self._total_enthalpy() - self._H0)
        self.diag.max_speed.append(self.max_speed())
        self.diag.pe_number.append(self.peclet())
        self.diag.marangoni_corr.append(self.marangoni_outflow_corr())

    def run(
        self,
        t_end: Optional[float] = None,
        record_every: float = 0.05,
        callback: Optional[Callable] = None,
        progress: bool = True,
    ) -> Diagnostics:
        if t_end is None:
            t_end = self.proc.t_end
        next_rec = 0.0
        while self.t < t_end - 1e-15:
            dt = min(self.suggest_dt(), t_end - self.t)
            t_step = time.perf_counter()
            self.step(dt)
            step_wall = time.perf_counter() - t_step
            if self.t >= next_rec - 1e-15:
                self.record()
                next_rec += record_every
                if progress:
                    d, _ = self.pool_metrics()
                    print(
                        f"t={self.t:.3f}s  Tmax={self.T.max():.0f}K  "
                        f"depth={1e3 * d:.2f}mm  |u|max={self.diag.max_speed[-1]:.4f}m/s  "
                        f"step={step_wall:.3f}s",
                        flush=True,
                    )
                if callback is not None:
                    callback(self)
        if not self.diag.time or self.diag.time[-1] < self.t - 1e-12:
            self.record()
        return self.diag


def quick_process(**overrides) -> Process:
    base = dict(
        Lx=0.030,
        Lz=0.005,
        nx=24,
        nz=6,
        q_max=8.0e7,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.006,
        t_end=0.35,
        dt=5.0e-3,
        dt_max=5.0e-3,
        cfl=0.25,
        picard_iter=1,
        intorder=3,
        enable_inertia=False,
    )
    base.update(overrides)
    return Process(**base)
