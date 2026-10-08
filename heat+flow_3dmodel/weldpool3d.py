# 3D half-plate weld-pool model: heat, melt convection, solidification.
# No EM. Moving Gaussian surface flux. Darcy mushy zone (He 2003).
# Default properties: Al 5182 (Zhao & DebRoy 2001).
# Domain: x along weld, y from symmetry (y=0), z from -Lz (bottom) to 0 (heated surface).
# Heat / Marangoni on z=0; plate depth is z < 0.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized


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
    mush_C: float = 1.0e7
    mush_B: float = 1.0e-3


@dataclass
class Process:
    Lx: float = 0.020
    Ly: float = 0.008
    Lz: float = 0.004
    nx: int = 32
    ny: int = 12
    nz: int = 10
    P_arc: float = 2000.0  # W, full-plane arc power; q_max = 3P/(pi r_b^2)
    r_b: float = 0.0025
    v_weld: float = 0.008
    x0: float = 0.005
    T0: float = 300.0
    Ta: float = 300.0
    h_conv: float = 20.0
    h_bottom: float = 50.0
    emissivity: float = 0.4
    g: float = 9.81
    enable_flow: bool = True
    cfl: float = 0.25
    dt_max: float = 2.0e-3
    t_end: float = 0.4

    @property
    def q_max(self) -> float:
        return 3.0 * self.P_arc / (np.pi * self.r_b**2)


@dataclass
class Diagnostics:
    time: list = field(default_factory=list)
    pool_depth: list = field(default_factory=list)
    pool_length: list = field(default_factory=list)
    pool_width: list = field(default_factory=list)
    T_probe: list = field(default_factory=list)
    heat_in: list = field(default_factory=list)
    heat_loss: list = field(default_factory=list)
    energy_store: list = field(default_factory=list)
    max_speed: list = field(default_factory=list)
    pe_number: list = field(default_factory=list)
    T_max_frames: list = field(default_factory=list)
    T_frames: list = field(default_factory=list)
    fL_frames: list = field(default_factory=list)
    speed_frames: list = field(default_factory=list)
    arc_x_frames: list = field(default_factory=list)


class WeldPool3D:
    def __init__(self, material: Optional[Material] = None, process: Optional[Process] = None):
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
        # z=0 heated surface; z=-Lz bottom (index k increases with z)
        self.zw = -p.Lz + np.arange(p.nz + 1) * self.dz
        self.zc = -p.Lz + (np.arange(p.nz) + 0.5) * self.dz
        self.xu = np.arange(p.nx + 1) * self.dx
        self.yv = np.arange(p.ny + 1) * self.dy
        self.X, self.Y, self.Z = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")

    def reset(self) -> None:
        p = self.proc
        self.t = 0.0
        self.T = np.full((p.nx, p.ny, p.nz), p.T0, dtype=np.float64)
        self.fL = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.u = np.zeros((p.nx + 1, p.ny, p.nz), dtype=np.float64)
        self.v = np.zeros((p.nx, p.ny + 1, p.nz), dtype=np.float64)
        self.w = np.zeros((p.nx, p.ny, p.nz + 1), dtype=np.float64)
        self.p = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.T_max = self.T.copy()
        self.diag = Diagnostics()
        self._H0 = self._total_enthalpy()
        self._Q_cum = 0.0
        self._Q_loss_cum = 0.0
        ix = int(np.clip(np.searchsorted(self.xc, p.x0), 0, p.nx - 1))
        iy = 0  # near symmetry plane
        iz = p.nz // 2
        self.probe_ij = (ix, iy, iz)
        self._poisson_solve = None
        self._build_poisson()

    def liquid_fraction(self, T: np.ndarray) -> np.ndarray:
        m = self.mat
        return np.clip((T - m.T_solidus) / (m.T_liquidus - m.T_solidus), 0.0, 1.0)

    def _k(self, fL: np.ndarray) -> np.ndarray:
        return self.mat.k_s * (1.0 - fL) + self.mat.k_l * fL

    def _darcy_coeff(self, fL: np.ndarray) -> np.ndarray:
        m = self.mat
        return m.mush_C * (1.0 - fL) ** 2 / (fL**3 + m.mush_B)

    def arc_centre(self, t: Optional[float] = None) -> float:
        if t is None:
            t = self.t
        return self.proc.x0 + self.proc.v_weld * t

    def surface_heat_flux(self, t: Optional[float] = None):
        """Return q_net, q_in, q_loss on heated face z=0, shape (nx, ny)."""
        p = self.proc
        if t is None:
            t = self.t
        x_c = self.arc_centre(t)
        XX, YY = np.meshgrid(self.xc, self.yc, indexing="ij")
        q_in = p.q_max * np.exp(-3.0 * ((XX - x_c) ** 2 + YY**2) / p.r_b**2)
        T_top = self.T[:, :, -1]  # cells adjacent to z=0
        q_loss = p.h_conv * (T_top - p.Ta) + p.emissivity * SIGMA_SB * (T_top**4 - p.Ta**4)
        return q_in - q_loss, q_in, q_loss

    def _total_enthalpy(self) -> float:
        m = self.mat
        p = self.proc
        fL = self.liquid_fraction(self.T)
        cp_ref = 0.5 * (m.cp_s + m.cp_l)
        H = m.rho * (cp_ref * (self.T - p.T0) + fL * m.latent)
        return float(np.sum(H) * self.dx * self.dy * self.dz)

    def _cell_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_c = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        v_c = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        w_c = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return u_c, v_c, w_c

    def pool_metrics(self) -> Tuple[float, float, float]:
        """Return depth, length, full width (2 * half-width from y=0)."""
        melt = self.T >= self.mat.T_liquidus
        if not np.any(melt):
            return 0.0, 0.0, 0.0
        # depth: max penetration from z=0 into the plate (positive)
        depth = 0.0
        melted_ij = np.any(melt, axis=2)
        for i, j in zip(*np.where(melted_ij)):
            ks = np.where(melt[i, j, :])[0]
            if ks.size:
                depth = max(depth, -float(self.zc[ks.min()]))  # ks.min() → deepest (most negative z)
        length = float(np.sum(np.any(melt, axis=(1, 2))) * self.dx)
        half_w = float(np.sum(np.any(melt, axis=(0, 2))) * self.dy)
        return depth, length, 2.0 * half_w

    def peclet(self) -> float:
        u_c, v_c, w_c = self._cell_velocity()
        speed = np.sqrt(u_c**2 + v_c**2 + w_c**2)
        melt = self.fL > 0.5
        if not np.any(melt):
            return 0.0
        U = float(np.max(speed[melt]))
        depth, length, width = self.pool_metrics()
        Lchar = max(depth, 0.5 * length, 0.5 * width, self.dx)
        alpha = self.mat.k_l / (self.mat.rho * self.mat.cp_l)
        return U * Lchar / alpha

    def suggest_dt(self) -> float:
        p = self.proc
        m = self.mat
        umax = max(
            float(np.max(np.abs(self.u))),
            float(np.max(np.abs(self.v))),
            float(np.max(np.abs(self.w))),
            1e-6,
        )
        hmin = min(self.dx, self.dy, self.dz)
        dt_cfl = p.cfl * hmin / umax
        alpha = m.k_s / (m.rho * m.cp_s)
        dt_diff = 0.12 * hmin**2 / alpha
        nu = m.mu / m.rho
        dt_visc = 0.12 * hmin**2 / max(nu, 1e-12)
        return float(min(p.dt_max, dt_cfl, dt_diff, dt_visc))

    def divergence(self) -> np.ndarray:
        return (
            (self.u[1:, :, :] - self.u[:-1, :, :]) / self.dx
            + (self.v[:, 1:, :] - self.v[:, :-1, :]) / self.dy
            + (self.w[:, :, 1:] - self.w[:, :, :-1]) / self.dz
        )

    def step(self, dt: Optional[float] = None) -> float:
        if dt is None:
            dt = self.suggest_dt()
        if self.proc.enable_flow:
            self._step_momentum(dt)
        else:
            self.u.fill(0.0)
            self.v.fill(0.0)
            self.w.fill(0.0)
        self._step_energy(dt)
        self.fL = self.liquid_fraction(self.T)
        solid = self.fL < 1e-6
        # zero velocity in solid (staggered)
        solid_u = np.zeros_like(self.u, dtype=bool)
        solid_u[1:-1, :, :] = solid[:-1, :, :] & solid[1:, :, :]
        solid_u[0, :, :] = solid[0, :, :]
        solid_u[-1, :, :] = solid[-1, :, :]
        self.u[solid_u] = 0.0

        solid_v = np.zeros_like(self.v, dtype=bool)
        solid_v[:, 1:-1, :] = solid[:, :-1, :] & solid[:, 1:, :]
        solid_v[:, 0, :] = True  # symmetry plane: v=0
        solid_v[:, -1, :] = solid[:, -1, :]
        self.v[solid_v] = 0.0

        solid_w = np.zeros_like(self.w, dtype=bool)
        solid_w[:, :, 1:-1] = solid[:, :, :-1] & solid[:, :, 1:]
        solid_w[:, :, 0] = True
        solid_w[:, :, -1] = True
        self.w[solid_w] = 0.0

        self.T_max = np.maximum(self.T_max, self.T)
        self.t += dt
        return dt

    def _step_momentum(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        rho, mu = m.rho, m.mu
        fL = self.fL
        A = self._darcy_coeff(fL)
        u, v, w = self.u, self.v, self.w

        # ----- u-momentum (faces normal to x) -----
        u_star = u.copy()
        if nx >= 2:
            ui = u[1:-1, :, :]
            ue = u[2:, :, :]
            uw = u[:-2, :, :]

            # advecting v, w at u-faces
            v_adv = 0.5 * (
                0.5 * (v[:-1, 1:, :] + v[1:, 1:, :]) + 0.5 * (v[:-1, :-1, :] + v[1:, :-1, :])
            )
            w_adv = 0.5 * (
                0.5 * (w[:-1, :, 1:] + w[1:, :, 1:]) + 0.5 * (w[:-1, :, :-1] + w[1:, :, :-1])
            )

            dudx = np.where(ui >= 0.0, (ui - uw) / dx, (ue - ui) / dx)

            un = np.empty_like(ui)
            us = np.empty_like(ui)
            # y-neighbors of u (j direction)
            un[:, :-1, :] = u[1:-1, 1:, :]
            un[:, -1, :] = u[1:-1, -1, :]
            us[:, 1:, :] = u[1:-1, :-1, :]
            us[:, 0, :] = u[1:-1, 0, :]  # symmetry: ghost u same → ∂u/∂y=0
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
                d2y[:, 1:-1, :] = (u[1:-1, 2:, :] - 2.0 * u[1:-1, 1:-1, :] + u[1:-1, :-2, :]) / dy**2
            # y=0 symmetry: u_ghost = u[0] → (u[1]-u[0])/dy^2 style
            if ny >= 2:
                d2y[:, 0, :] = (u[1:-1, 1, :] - u[1:-1, 0, :]) / dy**2
                d2y[:, -1, :] = (u[1:-1, -2, :] - 3.0 * u[1:-1, -1, :]) / dy**2  # no-slip outer

            d2z = np.zeros_like(ui)
            if nz >= 3:
                d2z[:, :, 1:-1] = (u[1:-1, :, 2:] - 2.0 * u[1:-1, :, 1:-1] + u[1:-1, :, :-2]) / dz**2
            if nz >= 2:
                d2z[:, :, 0] = (u[1:-1, :, 1] - 3.0 * u[1:-1, :, 0]) / dz**2
            # Marangoni ghost at top for u
            fL_top = 0.5 * (fL[:-1, :, -1] + fL[1:, :, -1])
            dTdx = (self.T[1:, :, -1] - self.T[:-1, :, -1]) / dx
            tau_x = fL_top * m.dsigma_dT * dTdx
            u_g = u[1:-1, :, -1] + dz * tau_x / mu
            if nz >= 2:
                d2z[:, :, -1] = (u_g - 2.0 * u[1:-1, :, -1] + u[1:-1, :, -2]) / dz**2
            else:
                d2z[:, :, 0] = (u_g - 3.0 * u[1:-1, :, 0]) / dz**2

            visc = (mu / rho) * (d2x + d2y + d2z)
            A_f = 0.5 * (A[:-1, :, :] + A[1:, :, :])
            u_star[1:-1, :, :] = (u[1:-1, :, :] + dt * (-adv + visc)) / (1.0 + dt * A_f / rho)

        u_star[0, :, :] = 0.0
        u_star[-1, :, :] = 0.0

        # ----- v-momentum (faces normal to y) -----
        v_star = v.copy()
        if ny >= 2:
            vi = v[:, 1:-1, :]
            vn = v[:, 2:, :]
            vs = v[:, :-2, :]

            u_adv = 0.5 * (
                0.5 * (u[1:, :-1, :] + u[1:, 1:, :]) + 0.5 * (u[:-1, :-1, :] + u[:-1, 1:, :])
            )
            w_adv = 0.5 * (
                0.5 * (w[:, :-1, 1:] + w[:, 1:, 1:]) + 0.5 * (w[:, :-1, :-1] + w[:, 1:, :-1])
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
                d2z[:, :, 1:-1] = (v[:, 1:-1, 2:] - 2.0 * v[:, 1:-1, 1:-1] + v[:, 1:-1, :-2]) / dz**2
            if nz >= 2:
                d2z[:, :, 0] = (v[:, 1:-1, 1] - 3.0 * v[:, 1:-1, 0]) / dz**2
            # Marangoni for v on top
            fL_top = 0.5 * (fL[:, :-1, -1] + fL[:, 1:, -1])
            dTdy = (self.T[:, 1:, -1] - self.T[:, :-1, -1]) / dy
            tau_y = fL_top * m.dsigma_dT * dTdy
            v_g = v[:, 1:-1, -1] + dz * tau_y / mu
            if nz >= 2:
                d2z[:, :, -1] = (v_g - 2.0 * v[:, 1:-1, -1] + v[:, 1:-1, -2]) / dz**2
            else:
                d2z[:, :, 0] = (v_g - 3.0 * v[:, 1:-1, 0]) / dz**2

            visc = (mu / rho) * (d2x + d2y + d2z)
            A_f = 0.5 * (A[:, :-1, :] + A[:, 1:, :])
            v_star[:, 1:-1, :] = (v[:, 1:-1, :] + dt * (-adv + visc)) / (1.0 + dt * A_f / rho)

        v_star[:, 0, :] = 0.0  # symmetry
        v_star[:, -1, :] = 0.0  # outer wall

        # ----- w-momentum (faces normal to z) -----
        w_star = w.copy()
        if nz >= 2:
            wi = w[:, :, 1:-1]
            wt = w[:, :, 2:]
            wb = w[:, :, :-2]

            u_adv = 0.5 * (
                0.5 * (u[1:, :, :-1] + u[1:, :, 1:]) + 0.5 * (u[:-1, :, :-1] + u[:-1, :, 1:])
            )
            v_adv = 0.5 * (
                0.5 * (v[:, 1:, :-1] + v[:, 1:, 1:]) + 0.5 * (v[:, :-1, :-1] + v[:, :-1, 1:])
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
            ws[:, 0, :] = w[:, 0, 1:-1]  # symmetry ghost

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
                d2y[:, 1:-1, :] = (w[:, 2:, 1:-1] - 2.0 * w[:, 1:-1, 1:-1] + w[:, :-2, 1:-1]) / dy**2
            if ny >= 2:
                d2y[:, 0, :] = (w[:, 1, 1:-1] - w[:, 0, 1:-1]) / dy**2
                d2y[:, -1, :] = (w[:, -2, 1:-1] - 3.0 * w[:, -1, 1:-1]) / dy**2

            visc = (mu / rho) * (d2x + d2y + d2z)
            T_f = 0.5 * (self.T[:, :, :-1] + self.T[:, :, 1:])
            buoy = p.g * m.beta * (T_f - m.T_ref)
            A_f = 0.5 * (A[:, :, :-1] + A[:, :, 1:])
            w_star[:, :, 1:-1] = (w[:, :, 1:-1] + dt * (-adv + visc + buoy)) / (
                1.0 + dt * A_f / rho
            )

        w_star[:, :, 0] = 0.0
        w_star[:, :, -1] = 0.0

        self._project(u_star, v_star, w_star, dt)
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

    def _project(self, u_star, v_star, w_star, dt) -> None:
        m = self.mat
        nx, ny, nz = self.proc.nx, self.proc.ny, self.proc.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        div = (
            (u_star[1:, :, :] - u_star[:-1, :, :]) / dx
            + (v_star[:, 1:, :] - v_star[:, :-1, :]) / dy
            + (w_star[:, :, 1:] - w_star[:, :, :-1]) / dz
        )
        rhs = (m.rho / dt) * div.ravel(order="C")
        rhs[0] = 0.0
        phi = self._poisson_solve(rhs).reshape((nx, ny, nz), order="C")
        self.p = phi
        u_star[1:-1, :, :] -= (dt / m.rho) * (phi[1:, :, :] - phi[:-1, :, :]) / dx
        v_star[:, 1:-1, :] -= (dt / m.rho) * (phi[:, 1:, :] - phi[:, :-1, :]) / dy
        w_star[:, :, 1:-1] -= (dt / m.rho) * (phi[:, :, 1:] - phi[:, :, :-1]) / dz
        u_star[0, :, :] = 0.0
        u_star[-1, :, :] = 0.0
        v_star[:, 0, :] = 0.0
        v_star[:, -1, :] = 0.0
        w_star[:, :, 0] = 0.0
        w_star[:, :, -1] = 0.0

    def _step_energy(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        T = self.T
        fL = self.liquid_fraction(T)
        u_c, v_c, w_c = self._cell_velocity()
        cp_ref = 0.5 * (m.cp_s + m.cp_l)

        q_net, q_in, q_loss = self.surface_heat_flux(self.t)
        self._Q_cum += float(np.sum(q_in) * dx * dy) * dt
        self._Q_loss_cum += float(np.sum(q_loss) * dx * dy) * dt

        k = self._k(fL)
        # face conductivities
        ke = np.zeros((nx + 1, ny, nz))
        ke[1:-1, :, :] = 0.5 * (k[:-1, :, :] + k[1:, :, :])
        ke[0, :, :] = k[0, :, :]
        ke[-1, :, :] = k[-1, :, :]

        kn = np.zeros((nx, ny + 1, nz))
        kn[:, 1:-1, :] = 0.5 * (k[:, :-1, :] + k[:, 1:, :])
        kn[:, 0, :] = k[:, 0, :]  # symmetry
        kn[:, -1, :] = k[:, -1, :]

        kt = np.zeros((nx, ny, nz + 1))
        kt[:, :, 1:-1] = 0.5 * (k[:, :, :-1] + k[:, :, 1:])
        kt[:, :, 0] = k[:, :, 0]
        kt[:, :, -1] = k[:, :, -1]

        Fx = np.zeros((nx + 1, ny, nz))
        Fx[1:-1, :, :] = ke[1:-1, :, :] * (T[1:, :, :] - T[:-1, :, :]) / dx
        Fx[0, :, :] = 0.0
        Fx[-1, :, :] = 0.0

        Fy = np.zeros((nx, ny + 1, nz))
        Fy[:, 1:-1, :] = kn[:, 1:-1, :] * (T[:, 1:, :] - T[:, :-1, :]) / dy
        Fy[:, 0, :] = 0.0  # symmetry insulated
        Fy[:, -1, :] = 0.0

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
        Ts = np.empty_like(T)
        Tt = np.empty_like(T)
        Tb = np.empty_like(T)
        Te[:-1, :, :] = T[1:, :, :]
        Te[-1, :, :] = T[-1, :, :]
        Tw[1:, :, :] = T[:-1, :, :]
        Tw[0, :, :] = T[0, :, :]
        Tn[:, :-1, :] = T[:, 1:, :]
        Tn[:, -1, :] = T[:, -1, :]
        Ts[:, 1:, :] = T[:, :-1, :]
        Ts[:, 0, :] = T[:, 0, :]  # symmetry
        Tt[:, :, :-1] = T[:, :, 1:]
        Tt[:, :, -1] = T[:, :, -1]
        Tb[:, :, 1:] = T[:, :, :-1]
        Tb[:, :, 0] = T[:, :, 0]

        dTdx = np.where(u_c >= 0.0, (T - Tw) / dx, (Te - T) / dx)
        dTdy = np.where(v_c >= 0.0, (T - Ts) / dy, (Tn - T) / dy)
        dTdz = np.where(w_c >= 0.0, (T - Tb) / dz, (Tt - T) / dz)
        adv = u_c * dTdx + v_c * dTdy + w_c * dTdz

        dT_mush = m.T_liquidus - m.T_solidus
        in_mush = (T > m.T_solidus) & (T < m.T_liquidus)
        cp_eff = np.full_like(T, cp_ref) + np.where(in_mush, m.latent / dT_mush, 0.0)
        rhs = div_cond - m.rho * cp_ref * adv
        T_star = T + dt / (m.rho * cp_eff) * rhs

        will_mush = ((T <= m.T_solidus) & (T_star > m.T_solidus)) | (
            (T >= m.T_liquidus) & (T_star < m.T_liquidus)
        )
        if np.any(will_mush):
            cp2 = cp_ref + m.latent / dT_mush
            T_star = np.where(will_mush, T + dt / (m.rho * cp2) * rhs, T_star)

        self.T = np.clip(T_star, 200.0, 3500.0)
        self.fL = self.liquid_fraction(self.T)
        self._Q_loss_cum += float(np.sum(np.maximum(q_bot, 0.0)) * dx * dy) * dt

    def record(self, record_fields: bool = False) -> None:
        depth, length, width = self.pool_metrics()
        u_c, v_c, w_c = self._cell_velocity()
        speed = np.sqrt(u_c**2 + v_c**2 + w_c**2)
        ix, iy, iz = self.probe_ij
        self.diag.time.append(self.t)
        self.diag.pool_depth.append(depth)
        self.diag.pool_length.append(length)
        self.diag.pool_width.append(width)
        self.diag.T_probe.append(float(self.T[ix, iy, iz]))
        self.diag.heat_in.append(self._Q_cum)
        self.diag.heat_loss.append(self._Q_loss_cum)
        self.diag.energy_store.append(self._total_enthalpy() - self._H0)
        self.diag.max_speed.append(float(np.max(speed)))
        self.diag.pe_number.append(self.peclet())
        if record_fields:
            self.diag.T_max_frames.append(self.T_max.copy())
            self.diag.T_frames.append(self.T.copy())
            self.diag.fL_frames.append(self.fL.copy())
            self.diag.speed_frames.append(speed.copy())
            self.diag.arc_x_frames.append(self.arc_centre())

    def heat_balance_ratio(self) -> float:
        dH = self._total_enthalpy() - self._H0
        num = abs(self._Q_cum - self._Q_loss_cum - dH)
        den = max(abs(self._Q_cum), 1.0)
        return float(num / den)

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
        if self.diag.time:
            next_rec = self.diag.time[-1] + record_every
        else:
            next_rec = self.t
        wall0 = time.perf_counter()
        while self.t < t_end - 1e-15:
            dt = min(self.suggest_dt(), t_end - self.t)
            t_step0 = time.perf_counter()
            self.step(dt)
            dt_wall = time.perf_counter() - t_step0
            if self.t >= next_rec - 1e-15:
                self.record(record_fields=record_fields)
                next_rec += record_every
                if progress:
                    d, L, W = self.pool_metrics()
                    print(
                        f"t={self.t:.3f}s  Tmax={self.T.max():.0f}K  "
                        f"depth={1e3 * d:.2f}mm  width={1e3 * W:.2f}mm  "
                        f"|u|max={self.diag.max_speed[-1]:.4f}m/s  "
                        f"step={1e3 * dt_wall:.1f}ms  "
                        f"wall={time.perf_counter() - wall0:.1f}s",
                        flush=True,
                    )
                if callback is not None:
                    callback(self)
        if not self.diag.time or self.diag.time[-1] < self.t - 1e-12:
            self.record(record_fields=record_fields)
        return self.diag


def quick_process(**overrides) -> Process:
    base = dict(
        Lx=0.020,
        Ly=0.008,
        Lz=0.004,
        nx=32,
        ny=12,
        nz=10,
        P_arc=2000.0,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.005,
        t_end=0.4,
        dt_max=2.0e-3,
        cfl=0.25,
    )
    base.update(overrides)
    return Process(**base)
