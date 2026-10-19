# 2D longitudinal weld-pool model: heat, melt convection, solidification.
# No EM. Moving Gaussian surface flux. Darcy mushy zone (He 2003).
# Default properties: Al 5182 (Zhao & DebRoy 2001).

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

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
    Lx: float = 0.040
    Lz: float = 0.006
    nx: int = 120
    nz: int = 30
    q_max: float = 6.0e7
    r_b: float = 0.003
    v_weld: float = 0.005
    x0: float = 0.008
    T0: float = 300.0
    Ta: float = 300.0
    h_conv: float = 20.0
    h_bottom: float = 50.0
    emissivity: float = 0.4
    g: float = 9.81
    enable_flow: bool = True
    cfl: float = 0.25
    dt_max: float = 2.0e-3
    t_end: float = 4.0


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
    T_max_frames: list = field(default_factory=list)


class WeldPool2D:
    def __init__(self, material: Optional[Material] = None, process: Optional[Process] = None):
        self.mat = material or Material()
        self.proc = process or Process()
        self._build_grid()
        self.reset()

    def _build_grid(self) -> None:
        p = self.proc
        self.dx = p.Lx / p.nx
        self.dz = p.Lz / p.nz
        self.xc = (np.arange(p.nx) + 0.5) * self.dx
        self.zc = (np.arange(p.nz) + 0.5) * self.dz
        self.xu = np.arange(p.nx + 1) * self.dx
        self.zw = np.arange(p.nz + 1) * self.dz
        self.X, self.Z = np.meshgrid(self.xc, self.zc, indexing="ij")

    def reset(self) -> None:
        p = self.proc
        self.t = 0.0
        self.T = np.full((p.nx, p.nz), p.T0, dtype=np.float64)
        self.fL = np.zeros((p.nx, p.nz), dtype=np.float64)
        self.u = np.zeros((p.nx + 1, p.nz), dtype=np.float64)
        self.w = np.zeros((p.nx, p.nz + 1), dtype=np.float64)
        self.p = np.zeros((p.nx, p.nz), dtype=np.float64)
        self.T_max = self.T.copy()
        self.diag = Diagnostics()
        self._H0 = self._total_enthalpy()
        self._Q_cum = 0.0
        self._Q_loss_cum = 0.0
        ix = int(np.clip(np.searchsorted(self.xc, p.x0), 0, p.nx - 1))
        iz = p.nz // 2
        self.probe_ij = (ix, iz)
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
        p = self.proc
        if t is None:
            t = self.t
        x_c = self.arc_centre(t)
        q_in = p.q_max * np.exp(-3.0 * (self.xc - x_c) ** 2 / p.r_b**2)
        T_top = self.T[:, -1]
        q_loss = p.h_conv * (T_top - p.Ta) + p.emissivity * SIGMA_SB * (T_top**4 - p.Ta**4)
        return q_in - q_loss, q_in, q_loss

    def _total_enthalpy(self) -> float:
        m = self.mat
        p = self.proc
        fL = self.liquid_fraction(self.T)
        cp_ref = 0.5 * (m.cp_s + m.cp_l)
        H = m.rho * (cp_ref * (self.T - p.T0) + fL * m.latent)
        return float(np.sum(H) * self.dx * self.dz)

    def _cell_velocity(self):
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        w_c = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        return u_c, w_c

    def pool_metrics(self):
        melt = self.T >= self.mat.T_liquidus
        if not np.any(melt):
            return 0.0, 0.0
        melted_cols = np.any(melt, axis=1)
        depth = 0.0
        for i in np.where(melted_cols)[0]:
            ks = np.where(melt[i, :])[0]
            if ks.size:
                depth = max(depth, self.proc.Lz - float(self.zc[ks.min()]))
        length = float(np.sum(melted_cols) * self.dx)
        return depth, length

    def peclet(self) -> float:
        u_c, w_c = self._cell_velocity()
        speed = np.sqrt(u_c**2 + w_c**2)
        melt = self.fL > 0.5
        if not np.any(melt):
            return 0.0
        U = float(np.max(speed[melt]))
        depth, length = self.pool_metrics()
        Lchar = max(depth, 0.5 * length, self.dx)
        alpha = self.mat.k_l / (self.mat.rho * self.mat.cp_l)
        return U * Lchar / alpha

    def suggest_dt(self) -> float:
        p = self.proc
        m = self.mat
        umax = max(float(np.max(np.abs(self.u))), float(np.max(np.abs(self.w))), 1e-6)
        dt_cfl = p.cfl * min(self.dx, self.dz) / umax
        alpha = m.k_s / (m.rho * m.cp_s)
        dt_diff = 0.15 * min(self.dx, self.dz) ** 2 / alpha
        nu = m.mu / m.rho
        dt_visc = 0.15 * min(self.dx, self.dz) ** 2 / max(nu, 1e-12)
        return float(min(p.dt_max, dt_cfl, dt_diff, dt_visc))

    def step(self, dt: Optional[float] = None) -> float:
        if dt is None:
            dt = self.suggest_dt()
        if self.proc.enable_flow:
            self._step_momentum(dt)
        else:
            self.u.fill(0.0)
            self.w.fill(0.0)
        self._step_energy(dt)
        self.fL = self.liquid_fraction(self.T)
        solid = self.fL < 1e-6
        solid_u = np.zeros_like(self.u, dtype=bool)
        solid_u[1:-1, :] = solid[:-1, :] & solid[1:, :]
        solid_u[0, :] = solid[0, :]
        solid_u[-1, :] = solid[-1, :]
        self.u[solid_u] = 0.0
        solid_w = np.zeros_like(self.w, dtype=bool)
        solid_w[:, 1:-1] = solid[:, :-1] & solid[:, 1:]
        solid_w[:, 0] = True
        solid_w[:, -1] = True
        self.w[solid_w] = 0.0
        self.T_max = np.maximum(self.T_max, self.T)
        self.t += dt
        return dt

    def _step_momentum(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        nx, nz = p.nx, p.nz
        dx, dz = self.dx, self.dz
        rho, mu = m.rho, m.mu
        fL = self.fL
        A = self._darcy_coeff(fL)
        u, w = self.u, self.w

        u_star = u.copy()
        if nx >= 2:
            ui = u[1:-1, :]
            ue = u[2:, :]
            uw = u[:-2, :]
            w_u_n = 0.5 * (w[:-1, 1:] + w[1:, 1:])
            w_u_s = 0.5 * (w[:-1, :-1] + w[1:, :-1])
            w_adv = 0.5 * (w_u_n[: nx - 1, :] + w_u_s[: nx - 1, :])

            dudx = np.where(ui >= 0.0, (ui - uw) / dx, (ue - ui) / dx)
            un = np.empty_like(ui)
            us = np.empty_like(ui)
            un[:, :-1] = u[1:-1, 1:]
            un[:, -1] = u[1:-1, -1]
            us[:, 1:] = u[1:-1, :-1]
            us[:, 0] = u[1:-1, 0]
            dudz = np.where(w_adv >= 0.0, (ui - us) / dz, (un - ui) / dz)
            adv = ui * dudx + w_adv * dudz

            d2x = (ue - 2.0 * ui + uw) / dx**2
            d2z = np.zeros_like(ui)
            if nz >= 3:
                d2z[:, 1:-1] = (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dz**2
            if nz >= 2:
                d2z[:, 0] = (u[1:-1, 1] - 3.0 * u[1:-1, 0]) / dz**2
            fL_top = 0.5 * (fL[:-1, -1] + fL[1:, -1])
            dTdx = (self.T[1:, -1] - self.T[:-1, -1]) / dx
            tau = fL_top * m.dsigma_dT * dTdx
            u_g = u[1:-1, -1] + dz * tau / mu
            if nz >= 2:
                d2z[:, -1] = (u_g - 2.0 * u[1:-1, -1] + u[1:-1, -2]) / dz**2
            else:
                d2z[:, 0] = (u_g - 3.0 * u[1:-1, 0]) / dz**2

            visc = (mu / rho) * (d2x + d2z)
            A_f = 0.5 * (A[:-1, :] + A[1:, :])
            u_star[1:-1, :] = (u[1:-1, :] + dt * (-adv + visc)) / (1.0 + dt * A_f / rho)

        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0

        w_star = w.copy()
        if nz >= 2:
            wi = w[:, 1:-1]
            wn = w[:, 2:]
            ws = w[:, :-2]
            we = np.empty_like(wi)
            ww = np.empty_like(wi)
            we[:-1, :] = w[1:, 1:-1]
            we[-1, :] = w[-1, 1:-1]
            ww[1:, :] = w[:-1, 1:-1]
            ww[0, :] = w[0, 1:-1]

            u_east = 0.5 * (u[1:, :-1] + u[1:, 1:])
            u_west = 0.5 * (u[:-1, :-1] + u[:-1, 1:])
            u_adv = 0.5 * (u_east[:, : nz - 1] + u_west[:, : nz - 1])

            dwdx = np.where(u_adv >= 0.0, (wi - ww) / dx, (we - wi) / dx)
            dwdz = np.where(wi >= 0.0, (wi - ws) / dz, (wn - wi) / dz)
            adv = u_adv * dwdx + wi * dwdz

            d2z = (wn - 2.0 * wi + ws) / dz**2
            d2x = (we - 2.0 * wi + ww) / dx**2
            if nx >= 2:
                d2x[0, :] = (w[1, 1:-1] - 3.0 * w[0, 1:-1]) / dx**2
                d2x[-1, :] = (w[-2, 1:-1] - 3.0 * w[-1, 1:-1]) / dx**2

            visc = (mu / rho) * (d2x + d2z)
            T_f = 0.5 * (self.T[:, :-1] + self.T[:, 1:])
            buoy = p.g * m.beta * (T_f - m.T_ref)
            A_f = 0.5 * (A[:, :-1] + A[:, 1:])
            w_star[:, 1:-1] = (w[:, 1:-1] + dt * (-adv + visc + buoy)) / (1.0 + dt * A_f / rho)

        w_star[:, 0] = 0.0
        w_star[:, -1] = 0.0

        self._project(u_star, w_star, dt)
        self.u = u_star
        self.w = w_star

    def _build_poisson(self) -> None:
        nx, nz = self.proc.nx, self.proc.nz
        dx2, dz2 = self.dx**2, self.dz**2
        N = nx * nz
        rows, cols, data = [], [], []

        def idx(i, k):
            return i * nz + k

        for i in range(nx):
            for k in range(nz):
                r = idx(i, k)
                if i == 0 and k == 0:
                    rows.append(r)
                    cols.append(r)
                    data.append(1.0)
                    continue
                diag = 0.0
                if i + 1 < nx:
                    rows.append(r)
                    cols.append(idx(i + 1, k))
                    data.append(1.0 / dx2)
                    diag -= 1.0 / dx2
                if i - 1 >= 0:
                    rows.append(r)
                    cols.append(idx(i - 1, k))
                    data.append(1.0 / dx2)
                    diag -= 1.0 / dx2
                if k + 1 < nz:
                    rows.append(r)
                    cols.append(idx(i, k + 1))
                    data.append(1.0 / dz2)
                    diag -= 1.0 / dz2
                if k - 1 >= 0:
                    rows.append(r)
                    cols.append(idx(i, k - 1))
                    data.append(1.0 / dz2)
                    diag -= 1.0 / dz2
                rows.append(r)
                cols.append(r)
                data.append(diag if diag != 0.0 else 1.0)

        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        self._poisson_solve = factorized(A.tocsc())

    def _project(self, u_star, w_star, dt):
        m = self.mat
        nx, nz = self.proc.nx, self.proc.nz
        dx, dz = self.dx, self.dz
        div = (u_star[1:, :] - u_star[:-1, :]) / dx + (w_star[:, 1:] - w_star[:, :-1]) / dz
        rhs = (m.rho / dt) * div.ravel(order="C")
        rhs[0] = 0.0
        phi = self._poisson_solve(rhs).reshape((nx, nz), order="C")
        self.p = phi
        u_star[1:-1, :] -= (dt / m.rho) * (phi[1:, :] - phi[:-1, :]) / dx
        w_star[:, 1:-1] -= (dt / m.rho) * (phi[:, 1:] - phi[:, :-1]) / dz
        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0
        w_star[:, 0] = 0.0
        w_star[:, -1] = 0.0

    def _step_energy(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        nx, nz = p.nx, p.nz
        dx, dz = self.dx, self.dz
        T = self.T
        fL = self.liquid_fraction(T)
        u_c, w_c = self._cell_velocity()
        cp_ref = 0.5 * (m.cp_s + m.cp_l)

        q_net, q_in, q_loss = self.surface_heat_flux(self.t)
        self._Q_cum += float(np.sum(q_in) * dx) * dt
        self._Q_loss_cum += float(np.sum(q_loss) * dx) * dt

        k = self._k(fL)
        ke = np.zeros((nx + 1, nz))
        ke[1:-1, :] = 0.5 * (k[:-1, :] + k[1:, :])
        ke[0, :] = k[0, :]
        ke[-1, :] = k[-1, :]
        kn = np.zeros((nx, nz + 1))
        kn[:, 1:-1] = 0.5 * (k[:, :-1] + k[:, 1:])
        kn[:, 0] = k[:, 0]
        kn[:, -1] = k[:, -1]

        Fx = np.zeros((nx + 1, nz))
        Fx[1:-1, :] = ke[1:-1, :] * (T[1:, :] - T[:-1, :]) / dx
        Fx[0, :] = 0.0
        Fx[-1, :] = 0.0

        Fz = np.zeros((nx, nz + 1))
        Fz[:, 1:-1] = kn[:, 1:-1] * (T[:, 1:] - T[:, :-1]) / dz
        q_bot = p.h_bottom * (T[:, 0] - p.Ta)
        Fz[:, 0] = q_bot
        Fz[:, -1] = q_net
        div_cond = (Fx[1:, :] - Fx[:-1, :]) / dx + (Fz[:, 1:] - Fz[:, :-1]) / dz

        Te = np.empty_like(T)
        Tw = np.empty_like(T)
        Tn = np.empty_like(T)
        Ts = np.empty_like(T)
        Te[:-1, :] = T[1:, :]
        Te[-1, :] = T[-1, :]
        Tw[1:, :] = T[:-1, :]
        Tw[0, :] = T[0, :]
        Tn[:, :-1] = T[:, 1:]
        Tn[:, -1] = T[:, -1]
        Ts[:, 1:] = T[:, :-1]
        Ts[:, 0] = T[:, 0]
        dTdx = np.where(u_c >= 0.0, (T - Tw) / dx, (Te - T) / dx)
        dTdz = np.where(w_c >= 0.0, (T - Ts) / dz, (Tn - T) / dz)
        adv = u_c * dTdx + w_c * dTdz

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
        self._Q_loss_cum += float(np.sum(np.maximum(q_bot, 0.0)) * dx) * dt

    def record(self, record_fields: bool = False) -> None:
        depth, length = self.pool_metrics()
        u_c, w_c = self._cell_velocity()
        speed = np.sqrt(u_c**2 + w_c**2)
        ix, iz = self.probe_ij
        self.diag.time.append(self.t)
        self.diag.pool_depth.append(depth)
        self.diag.pool_length.append(length)
        self.diag.T_probe.append(float(self.T[ix, iz]))
        self.diag.heat_in.append(self._Q_cum)
        self.diag.heat_loss.append(self._Q_loss_cum)
        self.diag.energy_store.append(self._total_enthalpy() - self._H0)
        self.diag.max_speed.append(float(np.max(speed)))
        self.diag.pe_number.append(self.peclet())
        if record_fields:
            self.diag.T_max_frames.append(self.T_max.copy())

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
        while self.t < t_end - 1e-15:
            dt = min(self.suggest_dt(), t_end - self.t)
            self.step(dt)
            if self.t >= next_rec - 1e-15:
                self.record(record_fields=record_fields)
                next_rec += record_every
                if progress:
                    d, _ = self.pool_metrics()
                    print(
                        f"t={self.t:.3f}s  Tmax={self.T.max():.0f}K  "
                        f"depth={1e3 * d:.2f}mm  |u|max={self.diag.max_speed[-1]:.4f}m/s",
                        flush=True,
                    )
                if callback is not None:
                    callback(self)
        if not self.diag.time or self.diag.time[-1] < self.t - 1e-12:
            self.record(record_fields=record_fields)
        return self.diag


def quick_process(**overrides) -> Process:
    base = dict(
        Lx=0.030,
        Lz=0.005,
        nx=60,
        nz=20,
        q_max=8.0e7,
        r_b=0.0025,
        v_weld=0.008,
        x0=0.006,
        t_end=1.5,
        dt_max=2.5e-3,
        cfl=0.3,
    )
    base.update(overrides)
    return Process(**base)
