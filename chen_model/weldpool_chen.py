# Chen 2020 BOP weld-pool model (torch frame).
# Half-plate 3D MAC: heat, melt convection, mushy Darcy, averaged droplet,
# alloy fraction Yd, Lorentz from axisymmetric metal EM.
# Arc plasma not solved — Gaussian surface heat / current / pressure BCs.
# Reference: Chen et al., Appl. Math. Model. 81 (2020) 386–400 (I=95 A BOP).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized, spsolve


MU0 = 4.0e-7 * np.pi
SIGMA_SB = 5.670374419e-8


@dataclass
class AlloyProps:
    """Thermophysical props for one alloy (plate or wire)."""

    name: str
    T_solidus: float
    T_liquidus: float
    rho: float = 2700.0
    cp_s: float = 900.0
    cp_l: float = 1170.0
    k_s: float = 160.0
    k_l: float = 100.0
    mu: float = 1.2e-3
    latent: float = 3.9e5
    beta: float = 1.0e-4
    sigma_e: float = 3.5e6
    sigma_e_solid: float = 1.5e7


# Approximate Al–Mg / Al–Si props (Davis / ASM; not tabulated in Chen PDF).
AA5754 = AlloyProps("AA5754", T_solidus=880.0, T_liquidus=920.0, k_s=150.0, k_l=95.0)
AA4043 = AlloyProps("AA4043", T_solidus=850.0, T_liquidus=905.0, k_s=170.0, k_l=100.0)


@dataclass
class Material:
    plate: AlloyProps = field(default_factory=lambda: AA5754)
    wire: AlloyProps = field(default_factory=lambda: AA4043)
    dsigma_dT: float = -3.5e-4
    mush_C: float = 1.0e7
    mush_B: float = 1.0e-3
    T_ref: float = 920.0

    def mix(self, Yd: np.ndarray) -> dict:
        """Linear mix of plate (Yd=0) and wire (Yd=1) properties."""
        Y = np.clip(Yd, 0.0, 1.0)
        p, w = self.plate, self.wire
        return {
            "T_solidus": (1.0 - Y) * p.T_solidus + Y * w.T_solidus,
            "T_liquidus": (1.0 - Y) * p.T_liquidus + Y * w.T_liquidus,
            "rho": (1.0 - Y) * p.rho + Y * w.rho,
            "cp_s": (1.0 - Y) * p.cp_s + Y * w.cp_s,
            "cp_l": (1.0 - Y) * p.cp_l + Y * w.cp_l,
            "k_s": (1.0 - Y) * p.k_s + Y * w.k_s,
            "k_l": (1.0 - Y) * p.k_l + Y * w.k_l,
            "mu": (1.0 - Y) * p.mu + Y * w.mu,
            "latent": (1.0 - Y) * p.latent + Y * w.latent,
            "beta": (1.0 - Y) * p.beta + Y * w.beta,
            "sigma_e": (1.0 - Y) * p.sigma_e + Y * w.sigma_e,
            "sigma_e_solid": (1.0 - Y) * p.sigma_e_solid + Y * w.sigma_e_solid,
        }


@dataclass
class Process:
    # Domain (half-plate y>=0; z=0 surface; z=-Lz bottom)
    Lx: float = 0.040
    Ly: float = 0.012
    Lz: float = 0.003
    nx: int = 60
    ny: int = 20
    nz: int = 20
    # Arc fixed in torch frame
    x_arc: float = 0.020
    I_arc: float = 95.0
    U_arc: float = 18.5  # assumed (not in Chen PDF)
    eta: float = 0.75  # assumed arc efficiency
    r_q: float = 0.003  # heat spot radius
    r_j: float = 0.0025  # current / pressure spot
    p_arc_max: float = 200.0  # Pa peak arc pressure (assumed)
    v_weld: float = 0.015  # 0.9 m/min
    v_feed: float = 0.0716667  # 4.3 m/min
    R_wire: float = 0.0006  # 1.2 mm diameter
    f_drop: float = 93.0  # Hz
    v_drop: float = 0.6  # m/s impact (assumed; Murphy-style)
    T_drop: float = 1200.0
    R_dep: float = 0.0012  # deposit cylinder radius ~ droplet
    H_dep: float = 0.0010  # deposit depth from surface
    T0: float = 300.0
    Ta: float = 300.0
    h_conv: float = 20.0
    h_bottom: float = 50.0
    emissivity: float = 0.4
    g: float = 9.81
    enable_flow: bool = True
    enable_em: bool = True
    enable_drop: bool = True
    cfl: float = 0.25
    dt_max: float = 1.5e-3
    t_end: float = 2.0
    em_every: int = 20
    F_em_clip: float = 5.0e6
    # EM axisymmetric grid
    nr_em: int = 40
    nz_em: int = 24

    @property
    def P_arc(self) -> float:
        return self.eta * self.I_arc * self.U_arc

    @property
    def q_max(self) -> float:
        return 3.0 * self.P_arc / (np.pi * self.r_q**2)

    @property
    def j_max(self) -> float:
        return 3.0 * self.I_arc / (np.pi * self.r_j**2)

    @property
    def m_dot(self) -> float:
        """Full-plane wire mass rate [kg/s]."""
        return AA4043.rho * np.pi * self.R_wire**2 * self.v_feed

    @property
    def bead_area(self) -> float:
        """Full reinforcement cross-section from mass balance [m²]."""
        return np.pi * self.R_wire**2 * self.v_feed / self.v_weld


@dataclass
class Diagnostics:
    time: list = field(default_factory=list)
    pool_depth: list = field(default_factory=list)
    pool_length: list = field(default_factory=list)
    pool_width: list = field(default_factory=list)
    RH: list = field(default_factory=list)
    gamma: list = field(default_factory=list)
    T_max: list = field(default_factory=list)
    max_speed: list = field(default_factory=list)
    pe_number: list = field(default_factory=list)
    T_frames: list = field(default_factory=list)
    fL_frames: list = field(default_factory=list)
    Yd_frames: list = field(default_factory=list)
    speed_frames: list = field(default_factory=list)
    u_frames: list = field(default_factory=list)
    w_frames: list = field(default_factory=list)


class WeldPoolChen:
    """Stationary-torch half-plate weld pool (Chen 2020 BOP style)."""

    def __init__(
        self,
        material: Optional[Material] = None,
        process: Optional[Process] = None,
    ):
        self.mat = material or Material()
        self.proc = process or Process()
        self._build_grid()
        self.reset()

    def _build_grid(self) -> None:
        p = self.proc
        self.dx = p.Lx / p.nx
        self.dy = p.Ly / p.ny
        self.dz = p.Lz / p.nz
        self.xc = (np.arange(p.nx) + 0.5) * self.dx
        self.yc = (np.arange(p.ny) + 0.5) * self.dy
        self.zw = -p.Lz + np.arange(p.nz + 1) * self.dz
        self.zc = -p.Lz + (np.arange(p.nz) + 0.5) * self.dz
        self.xu = np.arange(p.nx + 1) * self.dx
        self.yv = np.arange(p.ny + 1) * self.dy
        self.X, self.Y, self.Z = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")

    def reset(self) -> None:
        p = self.proc
        self.t = 0.0
        self._step_count = 0
        self.T = np.full((p.nx, p.ny, p.nz), p.T0, dtype=np.float64)
        self.Yd = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.fL = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        # Torch frame: solid streams toward -x at welding speed
        self.u = np.full((p.nx + 1, p.ny, p.nz), -p.v_weld, dtype=np.float64)
        self.v = np.zeros((p.nx, p.ny + 1, p.nz), dtype=np.float64)
        self.w = np.zeros((p.nx, p.ny, p.nz + 1), dtype=np.float64)
        self.p = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Fx = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Fy = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Fz = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.diag = Diagnostics()
        self._poisson_solve = None
        self._build_poisson()
        self._deposit_mask = self._make_deposit_mask()
        self._V_dep = float(np.sum(self._deposit_mask) * self.dx * self.dy * self.dz)
        if self._V_dep < 1e-16:
            self._V_dep = self.dx * self.dy * self.dz
        if p.enable_em:
            self._update_em()

    def _make_deposit_mask(self) -> np.ndarray:
        p = self.proc
        r2 = (self.X - p.x_arc) ** 2 + self.Y**2
        return (r2 <= p.R_dep**2) & (self.Z >= -p.H_dep)

    def props(self) -> dict:
        return self.mat.mix(self.Yd)

    def liquid_fraction(self, T: np.ndarray, props: Optional[dict] = None) -> np.ndarray:
        if props is None:
            props = self.props()
        Ts, Tl = props["T_solidus"], props["T_liquidus"]
        dT = np.maximum(Tl - Ts, 1.0)
        return np.clip((T - Ts) / dT, 0.0, 1.0)

    def _k(self, fL: np.ndarray, props: dict) -> np.ndarray:
        return props["k_s"] * (1.0 - fL) + props["k_l"] * fL

    def _darcy_coeff(self, fL: np.ndarray) -> np.ndarray:
        m = self.mat
        return m.mush_C * (1.0 - fL) ** 2 / (fL**3 + m.mush_B)

    def surface_heat_flux(self):
        p = self.proc
        XX, YY = np.meshgrid(self.xc, self.yc, indexing="ij")
        q_in = p.q_max * np.exp(-3.0 * ((XX - p.x_arc) ** 2 + YY**2) / p.r_q**2)
        T_top = self.T[:, :, -1]
        q_loss = p.h_conv * (T_top - p.Ta) + p.emissivity * SIGMA_SB * (T_top**4 - p.Ta**4)
        return q_in - q_loss, q_in, q_loss

    def surface_arc_pressure(self) -> np.ndarray:
        p = self.proc
        XX, YY = np.meshgrid(self.xc, self.yc, indexing="ij")
        return p.p_arc_max * np.exp(-3.0 * ((XX - p.x_arc) ** 2 + YY**2) / p.r_j**2)

    def _cell_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_c = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        v_c = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        w_c = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return u_c, v_c, w_c

    def pool_metrics(self) -> Tuple[float, float, float]:
        props = self.props()
        melt = self.T >= props["T_liquidus"]
        if not np.any(melt):
            return 0.0, 0.0, 0.0
        depth = 0.0
        melted_ij = np.any(melt, axis=2)
        for i, j in zip(*np.where(melted_ij)):
            ks = np.where(melt[i, j, :])[0]
            if ks.size:
                depth = max(depth, -float(self.zc[ks.min()]))
        length = float(np.sum(np.any(melt, axis=(1, 2))) * self.dx)
        half_w = float(np.sum(np.any(melt, axis=(0, 2))) * self.dy)
        return depth, length, 2.0 * half_w

    def reinforcement_height(self, bead_width: Optional[float] = None) -> float:
        """Parabolic bead: RH = 3 A / (2 BW) from mass balance."""
        if bead_width is None:
            _, _, bead_width = self.pool_metrics()
        if bead_width < 1e-6:
            return 0.0
        return float(1.5 * self.proc.bead_area / bead_width)

    def peclet(self) -> float:
        u_c, v_c, w_c = self._cell_velocity()
        # Relative to solid stream
        ur = u_c + self.proc.v_weld
        speed = np.sqrt(ur**2 + v_c**2 + w_c**2)
        melt = self.fL > 0.5
        if not np.any(melt):
            return 0.0
        U = float(np.max(speed[melt]))
        depth, length, width = self.pool_metrics()
        Lchar = max(depth, 0.5 * length, 0.5 * width, self.dx)
        props = self.props()
        rho = float(np.mean(props["rho"][melt]))
        cp = float(np.mean(props["cp_l"][melt]))
        k = float(np.mean(props["k_l"][melt]))
        alpha = k / (rho * cp)
        return U * Lchar / max(alpha, 1e-12)

    def suggest_dt(self) -> float:
        p = self.proc
        props = self.props()
        umax = max(
            float(np.max(np.abs(self.u))),
            float(np.max(np.abs(self.v))),
            float(np.max(np.abs(self.w))),
            p.v_weld,
            1e-6,
        )
        hmin = min(self.dx, self.dy, self.dz)
        dt_cfl = p.cfl * hmin / umax
        rho = float(np.mean(props["rho"]))
        cp = float(np.mean(props["cp_s"]))
        k = float(np.mean(props["k_s"]))
        alpha = k / (rho * cp)
        # Slightly relaxed Fourier number: Al conductivity makes z-explicit stiff
        dt_diff = 0.20 * hmin**2 / max(alpha, 1e-12)
        mu = float(np.mean(props["mu"]))
        nu = mu / rho
        dt_visc = 0.20 * hmin**2 / max(nu, 1e-12)
        return float(min(p.dt_max, dt_cfl, dt_diff, dt_visc))

    # ------------------------------------------------------------------ EM
    def _update_em(self) -> None:
        """Axisymmetric potential in metal; map j×B under the arc to 3D."""
        p = self.proc
        props = self.props()
        nr, nz = p.nr_em, p.nz_em
        Rmax = min(p.Ly, 0.5 * p.Lx)
        dr = Rmax / nr
        dz = p.Lz / nz
        rc = (np.arange(nr) + 0.5) * dr
        zc = -p.Lz + (np.arange(nz) + 0.5) * dz

        # Conductivity from mid-plane temperature under arc (scalar mean)
        T_ref = float(np.mean(self.T))
        fL_ref = float(np.clip(
            (T_ref - self.mat.plate.T_solidus)
            / max(self.mat.plate.T_liquidus - self.mat.plate.T_solidus, 1.0),
            0.0,
            1.0,
        ))
        se_l = float(np.mean(props["sigma_e"]))
        se_s = float(np.mean(props["sigma_e_solid"]))
        se = np.full((nr, nz), (1.0 - fL_ref) * se_s + fL_ref * se_l)

        # Surface current density (top row j=nz-1): inject I via Gaussian
        j_surf = p.j_max * np.exp(-3.0 * (rc / p.r_j) ** 2)
        # Normalize so ∫ j 2π r dr = I
        integ = float(np.sum(j_surf * 2.0 * np.pi * rc * dr))
        if integ > 0.0:
            j_surf *= p.I_arc / integ

        # Solve ∇·(σ∇φ)=0 with −σ ∂z φ = j_surf on top, φ=0 on bottom
        N = nr * nz
        rows, cols, data = [], [], []
        rhs = np.zeros(N)

        def idx(i, j):
            return i * nz + j

        for i in range(nr):
            for j in range(nz):
                r = idx(i, j)
                if j == 0:
                    rows.append(r)
                    cols.append(r)
                    data.append(1.0)
                    rhs[r] = 0.0
                    continue
                diag = 0.0
                # r-neighbors (axis Neumann; outer Neumann)
                for ni in (i - 1, i + 1):
                    if ni < 0 or ni >= nr:
                        continue
                    se_f = 0.5 * (se[i, j] + se[ni, j])
                    face_r = 0.5 * (rc[i] + rc[ni])
                    area = 2.0 * np.pi * face_r * dz
                    c = se_f * area / dr
                    rows.append(r)
                    cols.append(idx(ni, j))
                    data.append(c)
                    diag -= c
                # z-neighbors
                for nj in (j - 1, j + 1):
                    if nj < 0 or nj >= nz:
                        continue
                    se_f = 0.5 * (se[i, j] + se[i, nj])
                    area = 2.0 * np.pi * rc[i] * dr
                    c = se_f * area / dz
                    rows.append(r)
                    cols.append(idx(i, nj))
                    data.append(c)
                    diag -= c
                rows.append(r)
                cols.append(r)
                data.append(diag if abs(diag) > 1e-30 else 1.0)
                # Top Neumann: σ ∂φ/∂z · (−n) = j → inject current into top cell
                if j == nz - 1:
                    area = 2.0 * np.pi * rc[i] * dr
                    rhs[r] = -j_surf[i] * area

        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*singular.*")
            try:
                phi = spsolve(A, rhs).reshape((nr, nz))
            except Exception:
                self.Fx.fill(0.0)
                self.Fy.fill(0.0)
                self.Fz.fill(0.0)
                return
        if phi is None or np.any(~np.isfinite(phi)):
            self.Fx.fill(0.0)
            self.Fy.fill(0.0)
            self.Fz.fill(0.0)
            return

        Jr = np.zeros((nr, nz))
        Jz = np.zeros((nr, nz))
        for i in range(nr):
            for j in range(nz):
                if i == 0:
                    dphdr = (phi[1, j] - phi[0, j]) / dr if nr > 1 else 0.0
                elif i == nr - 1:
                    dphdr = (phi[i, j] - phi[i - 1, j]) / dr
                else:
                    dphdr = (phi[i + 1, j] - phi[i - 1, j]) / (2 * dr)
                if j == 0:
                    dphdz = (phi[i, 1] - phi[i, 0]) / dz if nz > 1 else 0.0
                elif j == nz - 1:
                    dphdz = (phi[i, j] - phi[i, j - 1]) / dz
                else:
                    dphdz = (phi[i, j + 1] - phi[i, j - 1]) / (2 * dz)
                Jr[i, j] = -se[i, j] * dphdr
                Jz[i, j] = -se[i, j] * dphdz

        I_enc = np.cumsum(Jz * 2.0 * np.pi * rc[:, None] * dr, axis=0)
        Bth = MU0 * I_enc / (2.0 * np.pi * (rc[:, None] + 1e-16))
        Fr = np.clip(-Jz * Bth, -p.F_em_clip, p.F_em_clip)
        Fz = np.clip(Jr * Bth, -p.F_em_clip, p.F_em_clip)

        # Map cylindrical (Fr, Fz) → Cartesian under arc
        self.Fx.fill(0.0)
        self.Fy.fill(0.0)
        self.Fz.fill(0.0)
        for i in range(p.nx):
            for jy in range(p.ny):
                xr = self.xc[i] - p.x_arc
                yr = self.yc[jy]
                rr = np.sqrt(xr**2 + yr**2)
                if rr > Rmax:
                    continue
                ir = int(np.clip(rr / dr, 0, nr - 1))
                for k in range(p.nz):
                    iz = int(np.clip((self.zc[k] + p.Lz) / dz, 0, nz - 1))
                    fr = Fr[ir, iz]
                    fz = Fz[ir, iz]
                    if rr > 1e-9:
                        self.Fx[i, jy, k] = fr * xr / rr
                        self.Fy[i, jy, k] = fr * yr / rr
                    self.Fz[i, jy, k] = fz

        self._em_rc = rc
        self._em_zc = zc
        self._em_Fr = Fr
        self._em_Fz = Fz
        self._em_Jz = Jz

    # -------------------------------------------------------------- physics
    def step(self, dt: Optional[float] = None) -> float:
        if dt is None:
            dt = self.suggest_dt()
        if (
            self.proc.enable_em
            and self._step_count % max(self.proc.em_every, 1) == 0
        ):
            self._update_em()
        if self.proc.enable_flow:
            self._step_momentum(dt)
        else:
            self.u.fill(-self.proc.v_weld)
            self.v.fill(0.0)
            self.w.fill(0.0)
        self._step_energy(dt)
        self._step_alloy(dt)
        props = self.props()
        self.fL = self.liquid_fraction(self.T, props)
        self._apply_solid_velocity()
        self.t += dt
        self._step_count += 1
        return dt

    def _apply_solid_velocity(self) -> None:
        """Force solid / near-solid to stream at −v_weld; keep symmetry BCs."""
        p = self.proc
        solid = self.fL < 1e-6
        solid_u = np.zeros_like(self.u, dtype=bool)
        solid_u[1:-1, :, :] = solid[:-1, :, :] & solid[1:, :, :]
        solid_u[0, :, :] = solid[0, :, :]
        solid_u[-1, :, :] = solid[-1, :, :]
        self.u[solid_u] = -p.v_weld

        solid_v = np.zeros_like(self.v, dtype=bool)
        solid_v[:, 1:-1, :] = solid[:, :-1, :] & solid[:, 1:, :]
        solid_v[:, 0, :] = True
        solid_v[:, -1, :] = solid[:, -1, :]
        self.v[solid_v] = 0.0

        solid_w = np.zeros_like(self.w, dtype=bool)
        solid_w[:, :, 1:-1] = solid[:, :, :-1] & solid[:, :, 1:]
        solid_w[:, :, 0] = True
        solid_w[:, :, -1] = True
        self.w[solid_w] = 0.0

        # Inlet / outlet faces: solid stream
        self.u[0, :, :] = -p.v_weld
        self.u[-1, :, :] = -p.v_weld
        self.v[:, 0, :] = 0.0
        self.w[:, :, 0] = 0.0
        self.w[:, :, -1] = 0.0

    def _drop_sources(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Fx_drop, Fz_drop, Q_drop) volumetric rates; half-domain mass."""
        p = self.proc
        nx, ny, nz = p.nx, p.ny, p.nz
        Fx = np.zeros((nx, ny, nz))
        Fz = np.zeros((nx, ny, nz))
        Q = np.zeros((nx, ny, nz))
        if not p.enable_drop or self._V_dep < 1e-16:
            return Fx, Fz, Q
        # Half plate receives half the wire mass (symmetry)
        m_half = 0.5 * p.m_dot
        mask = self._deposit_mask
        # Momentum: downward impact (w < 0)
        Fz[mask] = m_half * (-p.v_drop) / self._V_dep
        # Thermal: deposit enthalpy of liquid drop relative to local T
        props = self.props()
        cp = props["cp_l"]
        lat = props["latent"]
        Q[mask] = (
            m_half
            * (cp[mask] * (p.T_drop - self.T[mask]) + lat[mask])
            / self._V_dep
        )
        return Fx, Fz, Q

    def _step_momentum(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        props = self.props()
        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        rho = props["rho"]
        mu = props["mu"]
        fL = self.fL
        A = self._darcy_coeff(fL)
        u, v, w = self.u, self.v, self.w
        u_s = -p.v_weld
        Fx_d, Fz_d, _ = self._drop_sources()
        Fx_em, Fy_em, Fz_em = self.Fx, self.Fy, self.Fz

        # Mean density for projection stability
        rho0 = float(np.mean(rho))

        # ----- u -----
        u_star = u.copy()
        if nx >= 2:
            ui = u[1:-1, :, :]
            ue = u[2:, :, :]
            uw = u[:-2, :, :]
            v_adv = 0.5 * (
                0.5 * (v[:-1, 1:, :] + v[1:, 1:, :])
                + 0.5 * (v[:-1, :-1, :] + v[1:, :-1, :])
            )
            w_adv = 0.5 * (
                0.5 * (w[:-1, :, 1:] + w[1:, :, 1:])
                + 0.5 * (w[:-1, :, :-1] + w[1:, :, :-1])
            )
            dudx = np.where(ui >= 0.0, (ui - uw) / dx, (ue - ui) / dx)
            un = np.empty_like(ui)
            us = np.empty_like(ui)
            un[:, :-1, :] = u[1:-1, 1:, :]
            un[:, -1, :] = u[1:-1, -1, :]
            us[:, 1:, :] = u[1:-1, :-1, :]
            us[:, 0, :] = u[1:-1, 0, :]
            dudy = np.where(v_adv >= 0.0, (ui - us) / dy, (un - ui) / dy)
            ut = np.empty_like(ui)
            ub = np.empty_like(ui)
            ut[:, :, :-1] = u[1:-1, :, 1:]
            ut[:, :, -1] = u[1:-1, :, -1]
            ub[:, :, 1:] = u[1:-1, :, :-1]
            ub[:, :, 0] = u[1:-1, :, 0]
            dudz = np.where(w_adv >= 0.0, (ui - ub) / dz, (ut - ui) / dz)
            adv = ui * dudx + v_adv * dudy + w_adv * dudz

            d2x = (ue - 2.0 * ui + uw) / dx**2
            d2y = np.zeros_like(ui)
            if ny >= 3:
                d2y[:, 1:-1, :] = (
                    u[1:-1, 2:, :] - 2.0 * u[1:-1, 1:-1, :] + u[1:-1, :-2, :]
                ) / dy**2
            if ny >= 2:
                d2y[:, 0, :] = (u[1:-1, 1, :] - u[1:-1, 0, :]) / dy**2
                d2y[:, -1, :] = (u[1:-1, -2, :] - 3.0 * u[1:-1, -1, :]) / dy**2
            d2z = np.zeros_like(ui)
            if nz >= 3:
                d2z[:, :, 1:-1] = (
                    u[1:-1, :, 2:] - 2.0 * u[1:-1, :, 1:-1] + u[1:-1, :, :-2]
                ) / dz**2
            if nz >= 2:
                d2z[:, :, 0] = (u[1:-1, :, 1] - 3.0 * u[1:-1, :, 0]) / dz**2
            fL_top = 0.5 * (fL[:-1, :, -1] + fL[1:, :, -1])
            mu_top = 0.5 * (mu[:-1, :, -1] + mu[1:, :, -1])
            dTdx = (self.T[1:, :, -1] - self.T[:-1, :, -1]) / dx
            tau_x = fL_top * m.dsigma_dT * dTdx
            u_g = u[1:-1, :, -1] + dz * tau_x / np.maximum(mu_top, 1e-8)
            if nz >= 2:
                d2z[:, :, -1] = (u_g - 2.0 * u[1:-1, :, -1] + u[1:-1, :, -2]) / dz**2
            else:
                d2z[:, :, 0] = (u_g - 3.0 * u[1:-1, :, 0]) / dz**2

            rho_f = 0.5 * (rho[:-1, :, :] + rho[1:, :, :])
            mu_f = 0.5 * (mu[:-1, :, :] + mu[1:, :, :])
            visc = (mu_f / rho_f) * (d2x + d2y + d2z)
            A_f = 0.5 * (A[:-1, :, :] + A[1:, :, :])
            Fx_f = 0.5 * (Fx_em[:-1, :, :] + Fx_em[1:, :, :]) + 0.5 * (
                Fx_d[:-1, :, :] + Fx_d[1:, :, :]
            )
            # Darcy toward solid stream u_s
            body = Fx_f / rho_f + (A_f / rho_f) * u_s
            u_star[1:-1, :, :] = (u[1:-1, :, :] + dt * (-adv + visc + body)) / (
                1.0 + dt * A_f / rho_f
            )
        u_star[0, :, :] = u_s
        u_star[-1, :, :] = u_s

        # ----- v -----
        v_star = v.copy()
        if ny >= 2:
            vi = v[:, 1:-1, :]
            vn = v[:, 2:, :]
            vs = v[:, :-2, :]
            u_adv = 0.5 * (
                0.5 * (u[1:, :-1, :] + u[1:, 1:, :])
                + 0.5 * (u[:-1, :-1, :] + u[:-1, 1:, :])
            )
            w_adv = 0.5 * (
                0.5 * (w[:, :-1, 1:] + w[:, 1:, 1:])
                + 0.5 * (w[:, :-1, :-1] + w[:, 1:, :-1])
            )
            ve = np.empty_like(vi)
            vw = np.empty_like(vi)
            ve[:-1, :, :] = v[1:, 1:-1, :]
            ve[-1, :, :] = v[-1, 1:-1, :]
            vw[1:, :, :] = v[:-1, 1:-1, :]
            vw[0, :, :] = v[0, 1:-1, :]
            dvdx = np.where(u_adv >= 0.0, (vi - vw) / dx, (ve - vi) / dx)
            dvdy = np.where(vi >= 0.0, (vi - vs) / dy, (vn - vi) / dy)
            vt = np.empty_like(vi)
            vb = np.empty_like(vi)
            vt[:, :, :-1] = v[:, 1:-1, 1:]
            vt[:, :, -1] = v[:, 1:-1, -1]
            vb[:, :, 1:] = v[:, 1:-1, :-1]
            vb[:, :, 0] = v[:, 1:-1, 0]
            dvdz = np.where(w_adv >= 0.0, (vi - vb) / dz, (vt - vi) / dz)
            adv = u_adv * dvdx + vi * dvdy + w_adv * dvdz
            d2y = (vn - 2.0 * vi + vs) / dy**2
            d2x = (ve - 2.0 * vi + vw) / dx**2
            if nx >= 2:
                d2x[0, :, :] = (v[1, 1:-1, :] - 3.0 * v[0, 1:-1, :]) / dx**2
                d2x[-1, :, :] = (v[-2, 1:-1, :] - 3.0 * v[-1, 1:-1, :]) / dx**2
            d2z = np.zeros_like(vi)
            if nz >= 3:
                d2z[:, :, 1:-1] = (
                    v[:, 1:-1, 2:] - 2.0 * v[:, 1:-1, 1:-1] + v[:, 1:-1, :-2]
                ) / dz**2
            if nz >= 2:
                d2z[:, :, 0] = (v[:, 1:-1, 1] - 3.0 * v[:, 1:-1, 0]) / dz**2
            fL_top = 0.5 * (fL[:, :-1, -1] + fL[:, 1:, -1])
            mu_top = 0.5 * (mu[:, :-1, -1] + mu[:, 1:, -1])
            dTdy = (self.T[:, 1:, -1] - self.T[:, :-1, -1]) / dy
            tau_y = fL_top * m.dsigma_dT * dTdy
            v_g = v[:, 1:-1, -1] + dz * tau_y / np.maximum(mu_top, 1e-8)
            if nz >= 2:
                d2z[:, :, -1] = (v_g - 2.0 * v[:, 1:-1, -1] + v[:, 1:-1, -2]) / dz**2
            else:
                d2z[:, :, 0] = (v_g - 3.0 * v[:, 1:-1, 0]) / dz**2
            rho_f = 0.5 * (rho[:, :-1, :] + rho[:, 1:, :])
            mu_f = 0.5 * (mu[:, :-1, :] + mu[:, 1:, :])
            visc = (mu_f / rho_f) * (d2x + d2y + d2z)
            A_f = 0.5 * (A[:, :-1, :] + A[:, 1:, :])
            Fy_f = 0.5 * (Fy_em[:, :-1, :] + Fy_em[:, 1:, :])
            body = Fy_f / rho_f
            v_star[:, 1:-1, :] = (v[:, 1:-1, :] + dt * (-adv + visc + body)) / (
                1.0 + dt * A_f / rho_f
            )
        v_star[:, 0, :] = 0.0
        v_star[:, -1, :] = 0.0

        # ----- w -----
        w_star = w.copy()
        if nz >= 2:
            wi = w[:, :, 1:-1]
            wt = w[:, :, 2:]
            wb = w[:, :, :-2]
            u_adv = 0.5 * (
                0.5 * (u[1:, :, :-1] + u[1:, :, 1:])
                + 0.5 * (u[:-1, :, :-1] + u[:-1, :, 1:])
            )
            v_adv = 0.5 * (
                0.5 * (v[:, 1:, :-1] + v[:, 1:, 1:])
                + 0.5 * (v[:, :-1, :-1] + v[:, :-1, 1:])
            )
            we = np.empty_like(wi)
            ww = np.empty_like(wi)
            we[:-1, :, :] = w[1:, :, 1:-1]
            we[-1, :, :] = w[-1, :, 1:-1]
            ww[1:, :, :] = w[:-1, :, 1:-1]
            ww[0, :, :] = w[0, :, 1:-1]
            wn = np.empty_like(wi)
            ws = np.empty_like(wi)
            wn[:, :-1, :] = w[:, 1:, 1:-1]
            wn[:, -1, :] = w[:, -1, 1:-1]
            ws[:, 1:, :] = w[:, :-1, 1:-1]
            ws[:, 0, :] = w[:, 0, 1:-1]
            dwdx = np.where(u_adv >= 0.0, (wi - ww) / dx, (we - wi) / dx)
            dwdy = np.where(v_adv >= 0.0, (wi - ws) / dy, (wn - wi) / dy)
            dwdz = np.where(wi >= 0.0, (wi - wb) / dz, (wt - wi) / dz)
            adv = u_adv * dwdx + v_adv * dwdy + wi * dwdz
            d2z = (wt - 2.0 * wi + wb) / dz**2
            d2x = (we - 2.0 * wi + ww) / dx**2
            if nx >= 2:
                d2x[0, :, :] = (w[1, :, 1:-1] - 3.0 * w[0, :, 1:-1]) / dx**2
                d2x[-1, :, :] = (w[-2, :, 1:-1] - 3.0 * w[-1, :, 1:-1]) / dx**2
            d2y = np.zeros_like(wi)
            if ny >= 3:
                d2y[:, 1:-1, :] = (
                    w[:, 2:, 1:-1] - 2.0 * w[:, 1:-1, 1:-1] + w[:, :-2, 1:-1]
                ) / dy**2
            if ny >= 2:
                d2y[:, 0, :] = (w[:, 1, 1:-1] - w[:, 0, 1:-1]) / dy**2
                d2y[:, -1, :] = (w[:, -2, 1:-1] - 3.0 * w[:, -1, 1:-1]) / dy**2
            rho_f = 0.5 * (rho[:, :, :-1] + rho[:, :, 1:])
            mu_f = 0.5 * (mu[:, :, :-1] + mu[:, :, 1:])
            visc = (mu_f / rho_f) * (d2x + d2y + d2z)
            T_f = 0.5 * (self.T[:, :, :-1] + self.T[:, :, 1:])
            beta_f = 0.5 * (props["beta"][:, :, :-1] + props["beta"][:, :, 1:])
            buoy = p.g * beta_f * (T_f - m.T_ref)
            A_f = 0.5 * (A[:, :, :-1] + A[:, :, 1:])
            Fz_f = 0.5 * (Fz_em[:, :, :-1] + Fz_em[:, :, 1:]) + 0.5 * (
                Fz_d[:, :, :-1] + Fz_d[:, :, 1:]
            )
            body = buoy + Fz_f / rho_f
            w_star[:, :, 1:-1] = (w[:, :, 1:-1] + dt * (-adv + visc + body)) / (
                1.0 + dt * A_f / rho_f
            )
        w_star[:, :, 0] = 0.0
        w_star[:, :, -1] = 0.0

        self._project(u_star, v_star, w_star, dt, rho0)
        self.u = u_star
        self.v = v_star
        self.w = w_star

    def _build_poisson(self) -> None:
        nx, ny, nz = self.proc.nx, self.proc.ny, self.proc.nz
        dx2, dy2, dz2 = self.dx**2, self.dy**2, self.dz**2
        N = nx * ny * nz
        rows, cols, data = [], [], []

        def idx(i, j, k):
            return (i * ny + j) * nz + k

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = idx(i, j, k)
                    if i == 0 and j == 0 and k == 0:
                        rows.append(r)
                        cols.append(r)
                        data.append(1.0)
                        continue
                    diag = 0.0
                    if i + 1 < nx:
                        rows.append(r)
                        cols.append(idx(i + 1, j, k))
                        data.append(1.0 / dx2)
                        diag -= 1.0 / dx2
                    if i - 1 >= 0:
                        rows.append(r)
                        cols.append(idx(i - 1, j, k))
                        data.append(1.0 / dx2)
                        diag -= 1.0 / dx2
                    if j + 1 < ny:
                        rows.append(r)
                        cols.append(idx(i, j + 1, k))
                        data.append(1.0 / dy2)
                        diag -= 1.0 / dy2
                    if j - 1 >= 0:
                        rows.append(r)
                        cols.append(idx(i, j - 1, k))
                        data.append(1.0 / dy2)
                        diag -= 1.0 / dy2
                    if k + 1 < nz:
                        rows.append(r)
                        cols.append(idx(i, j, k + 1))
                        data.append(1.0 / dz2)
                        diag -= 1.0 / dz2
                    if k - 1 >= 0:
                        rows.append(r)
                        cols.append(idx(i, j, k - 1))
                        data.append(1.0 / dz2)
                        diag -= 1.0 / dz2
                    rows.append(r)
                    cols.append(r)
                    data.append(diag if diag != 0.0 else 1.0)

        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        self._poisson_solve = factorized(A.tocsc())

    def _project(self, u_star, v_star, w_star, dt, rho0: float) -> None:
        nx, ny, nz = self.proc.nx, self.proc.ny, self.proc.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        # Arc pressure as Neumann on top: incorporated as surface force →
        # modify w* near top before projection via pressure jump on cells
        p_arc = self.surface_arc_pressure()
        # Add −∇p_arc into top-layer cells as effective pressure BC on z=0:
        # after projection, set surface pressure contribution
        div = (
            (u_star[1:, :, :] - u_star[:-1, :, :]) / dx
            + (v_star[:, 1:, :] - v_star[:, :-1, :]) / dy
            + (w_star[:, :, 1:] - w_star[:, :, :-1]) / dz
        )
        rhs = (rho0 / dt) * div.ravel(order="C")
        rhs[0] = 0.0
        phi = self._poisson_solve(rhs).reshape((nx, ny, nz), order="C")
        # Superimpose arc pressure on top cells (drives mild surface pressure)
        phi[:, :, -1] += p_arc
        self.p = phi
        u_star[1:-1, :, :] -= (dt / rho0) * (phi[1:, :, :] - phi[:-1, :, :]) / dx
        v_star[:, 1:-1, :] -= (dt / rho0) * (phi[:, 1:, :] - phi[:, :-1, :]) / dy
        w_star[:, :, 1:-1] -= (dt / rho0) * (phi[:, :, 1:] - phi[:, :, :-1]) / dz
        u_star[0, :, :] = -self.proc.v_weld
        u_star[-1, :, :] = -self.proc.v_weld
        v_star[:, 0, :] = 0.0
        v_star[:, -1, :] = 0.0
        w_star[:, :, 0] = 0.0
        w_star[:, :, -1] = 0.0

    def _step_energy(self, dt: float) -> None:
        p = self.proc
        props = self.props()
        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        T = self.T
        fL = self.liquid_fraction(T, props)
        u_c, v_c, w_c = self._cell_velocity()
        rho = props["rho"]
        cp_ref = 0.5 * (props["cp_s"] + props["cp_l"])
        latent = props["latent"]
        Ts, Tl = props["T_solidus"], props["T_liquidus"]

        q_net, _, _ = self.surface_heat_flux()
        _, _, Q_drop = self._drop_sources()

        k = self._k(fL, props)
        ke = np.zeros((nx + 1, ny, nz))
        ke[1:-1, :, :] = 0.5 * (k[:-1, :, :] + k[1:, :, :])
        ke[0, :, :] = k[0, :, :]
        ke[-1, :, :] = k[-1, :, :]
        kn = np.zeros((nx, ny + 1, nz))
        kn[:, 1:-1, :] = 0.5 * (k[:, :-1, :] + k[:, 1:, :])
        kn[:, 0, :] = k[:, 0, :]
        kn[:, -1, :] = k[:, -1, :]
        kt = np.zeros((nx, ny, nz + 1))
        kt[:, :, 1:-1] = 0.5 * (k[:, :, :-1] + k[:, :, 1:])
        kt[:, :, 0] = k[:, :, 0]
        kt[:, :, -1] = k[:, :, -1]

        Fx = np.zeros((nx + 1, ny, nz))
        Fx[1:-1, :, :] = ke[1:-1, :, :] * (T[1:, :, :] - T[:-1, :, :]) / dx
        Fy = np.zeros((nx, ny + 1, nz))
        Fy[:, 1:-1, :] = kn[:, 1:-1, :] * (T[:, 1:, :] - T[:, :-1, :]) / dy
        Fz = np.zeros((nx, ny, nz + 1))
        Fz[:, :, 1:-1] = kt[:, :, 1:-1] * (T[:, :, 1:] - T[:, :, :-1]) / dz
        q_bot = p.h_bottom * (T[:, :, 0] - p.Ta)
        Fz[:, :, 0] = q_bot
        Fz[:, :, -1] = q_net

        div_cond = (
            (Fx[1:, :, :] - Fx[:-1, :, :]) / dx
            + (Fy[:, 1:, :] - Fy[:, :-1, :]) / dy
            + (Fz[:, :, 1:] - Fz[:, :, :-1]) / dz
        )

        Te = np.empty_like(T)
        Tw = np.empty_like(T)
        Tn = np.empty_like(T)
        Ts_ = np.empty_like(T)
        Tt = np.empty_like(T)
        Tb = np.empty_like(T)
        # Solid streams −x: cold upstream is x=Lx (east face)
        Te[:-1, :, :] = T[1:, :, :]
        Te[-1, :, :] = p.T0
        Tw[1:, :, :] = T[:-1, :, :]
        Tw[0, :, :] = T[0, :, :]
        Tn[:, :-1, :] = T[:, 1:, :]
        Tn[:, -1, :] = T[:, -1, :]
        Ts_[:, 1:, :] = T[:, :-1, :]
        Ts_[:, 0, :] = T[:, 0, :]
        Tt[:, :, :-1] = T[:, :, 1:]
        Tt[:, :, -1] = T[:, :, -1]
        Tb[:, :, 1:] = T[:, :, :-1]
        Tb[:, :, 0] = T[:, :, 0]

        dTdx = np.where(u_c >= 0.0, (T - Tw) / dx, (Te - T) / dx)
        dTdy = np.where(v_c >= 0.0, (T - Ts_) / dy, (Tn - T) / dy)
        dTdz = np.where(w_c >= 0.0, (T - Tb) / dz, (Tt - T) / dz)
        adv = u_c * dTdx + v_c * dTdy + w_c * dTdz

        dT_mush = np.maximum(Tl - Ts, 1.0)
        in_mush = (T > Ts) & (T < Tl)
        cp_eff = cp_ref + np.where(in_mush, latent / dT_mush, 0.0)
        rhs = div_cond - rho * cp_ref * adv + Q_drop
        T_star = T + dt / (rho * cp_eff) * rhs

        # Dirichlet cold metal at upstream face (x = Lx)
        T_star[-1, :, :] = p.T0
        # Soft clamp
        self.T = np.clip(T_star, 200.0, 2500.0)
        self.fL = self.liquid_fraction(self.T, self.props())

    def _step_alloy(self, dt: float) -> None:
        p = self.proc
        if not p.enable_drop:
            return
        Y = self.Yd
        u_c, v_c, w_c = self._cell_velocity()
        dx, dy, dz = self.dx, self.dy, self.dz
        props = self.props()
        rho = props["rho"]
        fL = self.fL

        Ye = np.empty_like(Y)
        Yw = np.empty_like(Y)
        Yn = np.empty_like(Y)
        Ys = np.empty_like(Y)
        Yt = np.empty_like(Y)
        Yb = np.empty_like(Y)
        Ye[:-1, :, :] = Y[1:, :, :]
        Ye[-1, :, :] = 0.0  # cold plate inlet
        Yw[1:, :, :] = Y[:-1, :, :]
        Yw[0, :, :] = Y[0, :, :]
        Yn[:, :-1, :] = Y[:, 1:, :]
        Yn[:, -1, :] = Y[:, -1, :]
        Ys[:, 1:, :] = Y[:, :-1, :]
        Ys[:, 0, :] = Y[:, 0, :]
        Yt[:, :, :-1] = Y[:, :, 1:]
        Yt[:, :, -1] = Y[:, :, -1]
        Yb[:, :, 1:] = Y[:, :, :-1]
        Yb[:, :, 0] = Y[:, :, 0]

        dYdx = np.where(u_c >= 0.0, (Y - Yw) / dx, (Ye - Y) / dx)
        dYdy = np.where(v_c >= 0.0, (Y - Ys) / dy, (Yn - Y) / dy)
        dYdz = np.where(w_c >= 0.0, (Y - Yb) / dz, (Yt - Y) / dz)
        adv = u_c * dYdx + v_c * dYdy + w_c * dYdz

        # Source only in liquid deposit region
        m_half = 0.5 * p.m_dot
        Sd = np.zeros_like(Y)
        liq_dep = self._deposit_mask & (fL > 0.1)
        n_liq = float(np.sum(liq_dep))
        if n_liq > 0:
            V_liq = n_liq * dx * dy * dz
            Sd[liq_dep] = m_half / V_liq  # kg/(m³·s) of wire alloy
        else:
            Sd[self._deposit_mask] = m_half / self._V_dep

        # Liquid: advect + wire source; solid: stream at u_s = -v_weld
        melt = fL > 0.05
        u_s = -p.v_weld
        dY = np.zeros_like(Y)
        dY[melt] = dt * (-adv[melt] + Sd[melt] / np.maximum(rho[melt], 1.0))
        solid = ~melt
        # u_s < 0 → upwind from east: ∂Y/∂x ≈ (Ye - Y)/dx
        dY[solid] = dt * (-u_s * (Ye[solid] - Y[solid]) / dx)

        Y_new = Y + dY
        Y_new[-1, :, :] = 0.0
        self.Yd = np.clip(Y_new, 0.0, 1.0)

    def record(self, record_fields: bool = False) -> None:
        depth, length, width = self.pool_metrics()
        rh = self.reinforcement_height(width)
        u_c, v_c, w_c = self._cell_velocity()
        ur = u_c + self.proc.v_weld
        speed = np.sqrt(ur**2 + v_c**2 + w_c**2)
        self.diag.time.append(self.t)
        self.diag.pool_depth.append(depth)
        self.diag.pool_length.append(length)
        self.diag.pool_width.append(width)
        self.diag.RH.append(rh)
        self.diag.gamma.append(depth / self.proc.Lz if self.proc.Lz > 0 else 0.0)
        self.diag.T_max.append(float(self.T.max()))
        self.diag.max_speed.append(float(np.max(speed)))
        self.diag.pe_number.append(self.peclet())
        if record_fields:
            self.diag.T_frames.append(self.T.copy())
            self.diag.fL_frames.append(self.fL.copy())
            self.diag.Yd_frames.append(self.Yd.copy())
            self.diag.speed_frames.append(speed.copy())
            self.diag.u_frames.append(u_c.copy())
            self.diag.w_frames.append(w_c.copy())

    def summary(self) -> dict:
        depth, length, width = self.pool_metrics()
        rh = self.reinforcement_height(width)
        return {
            "WD_mm": 1e3 * depth,
            "BW_mm": 1e3 * width,
            "RH_mm": 1e3 * rh,
            "gamma": depth / self.proc.Lz,
            "length_mm": 1e3 * length,
            "T_max": float(self.T.max()),
            "Yd_max": float(self.Yd.max()),
            "P_arc_W": self.proc.P_arc,
            "m_dot_kg_s": self.proc.m_dot,
            "bead_area_mm2": 1e6 * self.proc.bead_area,
        }

    def run(
        self,
        t_end: Optional[float] = None,
        record_every: float = 0.05,
        callback: Optional[Callable] = None,
        progress: bool = True,
        record_fields: bool = False,
    ) -> Diagnostics:
        if t_end is None:
            t_end = self.proc.t_end
        next_rec = self.t if not self.diag.time else self.diag.time[-1] + record_every
        wall0 = time.perf_counter()
        while self.t < t_end - 1e-15:
            dt = min(self.suggest_dt(), t_end - self.t)
            t0 = time.perf_counter()
            self.step(dt)
            dt_wall = time.perf_counter() - t0
            if self.t >= next_rec - 1e-15:
                self.record(record_fields=record_fields)
                next_rec += record_every
                if progress:
                    s = self.summary()
                    print(
                        f"t={self.t:.3f}s  Tmax={s['T_max']:.0f}K  "
                        f"WD={s['WD_mm']:.2f}mm  BW={s['BW_mm']:.2f}mm  "
                        f"RH={s['RH_mm']:.2f}mm  gamma={s['gamma']:.2f}  "
                        f"|u|={self.diag.max_speed[-1]:.3f}m/s  "
                        f"step={1e3 * dt_wall:.0f}ms  "
                        f"wall={time.perf_counter() - wall0:.1f}s",
                        flush=True,
                    )
                if callback is not None:
                    callback(self)
        if not self.diag.time or self.diag.time[-1] < self.t - 1e-12:
            self.record(record_fields=record_fields)
        return self.diag


def bop_95A(**overrides) -> Process:
    """Chen 2020 Table 1 BOP case I=95 A (working explicit grid)."""
    base = dict(
        Lx=0.040,
        Ly=0.012,
        Lz=0.003,
        # ~ plan 60x20x20; nz reduced for explicit Al conduction stability
        nx=48,
        ny=16,
        nz=12,
        x_arc=0.020,
        I_arc=95.0,
        U_arc=18.5,
        eta=0.75,
        r_q=0.003,
        r_j=0.0025,
        p_arc_max=200.0,
        v_weld=0.9 / 60.0,
        v_feed=4.3 / 60.0,
        R_wire=0.0006,
        f_drop=93.0,
        v_drop=0.6,
        T_drop=1200.0,
        t_end=1.5,
        dt_max=2.0e-3,
        cfl=0.25,
        enable_flow=True,
        enable_em=True,
        enable_drop=True,
        em_every=25,
    )
    base.update(overrides)
    return Process(**base)


def quick_bop_95A(**overrides) -> Process:
    """Coarse grid for smoke tests."""
    base = dict(
        nx=30,
        ny=12,
        nz=12,
        nr_em=24,
        nz_em=16,
        t_end=0.4,
        dt_max=2.0e-3,
        em_every=30,
    )
    base.update(overrides)
    return bop_95A(**base)
