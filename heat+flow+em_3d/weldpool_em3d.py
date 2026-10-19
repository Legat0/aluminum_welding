# 3D half-plate electrode–droplet–weld-pool model.
# Single-grid VOF: moving consumable wire, hanging drop, free-surface pool.
# EM in metal only (pinch in wire/drop, potential in pool). No arc plasma.
# Default properties: Al 5182 (Zhao & DebRoy 2001).
# Domain: x along weld, y >= 0 half-plane (symmetry at y=0), z from plate bottom upward.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


MU0 = 4.0e-7 * np.pi
SIGMA_SB = 5.670374419e-8


@dataclass
class Material:
    T_solidus: float = 850.0
    T_liquidus: float = 911.0
    rho: float = 2300.0
    rho_g: float = 46.0  # elevated gas density (~rho/50) for explicit stability
    cp_s: float = 900.0
    cp_l: float = 1200.0
    cp_g: float = 1000.0
    k_s: float = 168.0
    k_l: float = 108.0
    k_g: float = 0.05
    mu: float = 1.1e-3
    mu_g: float = 5.0e-5
    sigma: float = 0.90
    dsigma_dT: float = -3.5e-4
    latent: float = 3.95e5
    beta: float = 1.0e-4
    T_ref: float = 911.0
    mush_C: float = 1.0e7
    mush_B: float = 1.0e-3
    sigma_e: float = 4.0e6
    sigma_e_solid: float = 1.5e7
    sigma_e_gas: float = 1.0e-6


@dataclass
class Process:
    Lx: float = 0.010
    Ly: float = 0.003
    Lz: float = 0.007
    nx: int = 40
    ny: int = 12
    nz: int = 28
    H_plate: float = 0.003
    R_wire: float = 0.0006
    z_wire_tip: float = 0.0055
    R_drop0: float = 0.0011
    v_feed: float = 0.05
    v_weld: float = 0.12  # m/s — elevated in QUICK so travel is visible
    x0: float = 0.003
    I_arc: float = 160.0
    U_anode: float = 4.0
    P_arc: float = 1800.0
    r_b: float = 0.0025
    p_arc_max: float = 300.0
    T0: float = 300.0
    Ta: float = 300.0
    T_wire: float = 900.0
    T_drop: float = 1200.0
    h_conv: float = 20.0
    h_bottom: float = 50.0
    emissivity: float = 0.4
    g: float = 9.81
    enable_flow: bool = True
    enable_em: bool = True
    cfl: float = 0.20
    dt_max: float = 5.0e-5
    t_end: float = 0.015
    em_every: int = 4
    alpha_metal: float = 0.5
    u_clip: float = 5.0
    F_em_clip: float = 5.0e6


@dataclass
class Diagnostics:
    time: list = field(default_factory=list)
    metal_volume: list = field(default_factory=list)
    wire_volume: list = field(default_factory=list)
    pool_volume: list = field(default_factory=list)
    n_components: list = field(default_factory=list)
    neck_radius: list = field(default_factory=list)
    electrode_x: list = field(default_factory=list)
    T_max: list = field(default_factory=list)
    max_speed: list = field(default_factory=list)
    detached: list = field(default_factory=list)
    alpha_frames: list = field(default_factory=list)
    T_frames: list = field(default_factory=list)
    speed_frames: list = field(default_factory=list)
    time_frames: list = field(default_factory=list)
    xe_frames: list = field(default_factory=list)


class WeldPoolEM3D:
    """3D half-plate electrode–droplet–pool VOF solver with moving torch."""

    def __init__(self, material: Optional[Material] = None, process: Optional[Process] = None):
        self.mat = material or Material()
        self.proc = process or Process()
        self._build_grid()
        self.reset()

    # ------------------------------------------------------------------ grid
    def _build_grid(self) -> None:
        p = self.proc
        self.dx = p.Lx / p.nx
        self.dy = p.Ly / p.ny
        self.dz = p.Lz / p.nz
        self.xc = (np.arange(p.nx) + 0.5) * self.dx
        self.yc = (np.arange(p.ny) + 0.5) * self.dy
        self.zc = (np.arange(p.nz) + 0.5) * self.dz
        self.xu = np.arange(p.nx + 1) * self.dx
        self.yv = np.arange(p.ny + 1) * self.dy
        self.zw = np.arange(p.nz + 1) * self.dz
        self.X, self.Y, self.Z = np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")
        self.cell_vol = self.dx * self.dy * self.dz
        # half-plate volume factor: report full-plate equivalent = 2 * half
        self._vol_scale = 2.0

    def electrode_x(self, t: Optional[float] = None) -> float:
        if t is None:
            t = self.t
        return self.proc.x0 + self.proc.v_weld * t

    def reset(self) -> None:
        p = self.proc
        self.t = 0.0
        self._step_count = 0
        self.alpha = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.T = np.full((p.nx, p.ny, p.nz), p.T0, dtype=np.float64)
        self.fL = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.u = np.zeros((p.nx + 1, p.ny, p.nz), dtype=np.float64)
        self.v = np.zeros((p.nx, p.ny + 1, p.nz), dtype=np.float64)
        self.w = np.zeros((p.nx, p.ny, p.nz + 1), dtype=np.float64)
        self.p = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.phi = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Jx = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Jy = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Jz = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Fx = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Fy = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.Fz = np.zeros((p.nx, p.ny, p.nz), dtype=np.float64)
        self.wire_mask = np.zeros((p.nx, p.ny, p.nz), dtype=bool)
        self.pool_mask = np.zeros((p.nx, p.ny, p.nz), dtype=bool)
        self.drop_mask = np.zeros((p.nx, p.ny, p.nz), dtype=bool)
        self._detached = False
        self._V_metal0 = 0.0
        self._V_feed_cum = 0.0
        self.diag = Diagnostics()
        self._init_geometry()
        self.fL = self.liquid_fraction(self.T)
        self._V_metal0 = self.metal_volume()
        self._update_masks()

    def _r_cyl(self, xe: Optional[float] = None) -> np.ndarray:
        if xe is None:
            xe = self.electrode_x()
        return np.sqrt((self.X - xe) ** 2 + self.Y**2)

    def _init_geometry(self) -> None:
        p = self.proc
        m = self.mat
        xe = self.electrode_x(0.0)
        R = self._r_cyl(xe)

        plate = self.Z <= p.H_plate
        self.alpha[plate] = 1.0
        self.T[plate] = p.T0

        wire = (R <= p.R_wire) & (self.Z >= p.z_wire_tip)
        self.alpha[wire] = 1.0
        self.T[wire] = p.T_wire

        # impose initial wire velocity on faces near cylinder
        self.u[:] = 0.0
        self.v[:] = 0.0
        self.w[:] = 0.0
        for i in range(p.nx):
            for j in range(p.ny):
                for k in range(p.nz + 1):
                    zf = self.zw[k]
                    if zf < p.z_wire_tip:
                        continue
                    # cell centres adjacent
                    xc = self.xc[i]
                    yc = self.yc[j]
                    if (xc - xe) ** 2 + yc**2 <= (p.R_wire + 0.5 * self.dx) ** 2:
                        self.w[i, j, k] = -p.v_feed
                        # u on faces i and i+1 later via feed

        z_c = p.z_wire_tip - 0.15 * p.R_drop0
        drop = (R**2 + (self.Z - z_c) ** 2) <= p.R_drop0**2
        drop &= self.Z > p.H_plate
        drop &= self.Z < p.z_wire_tip + 0.5 * p.R_wire
        self.alpha[drop] = np.maximum(self.alpha[drop], 1.0)
        self.T[drop] = p.T_drop

        pool_seed = (
            (self.Z <= p.H_plate)
            & (self.Z >= p.H_plate - 0.8e-3)
            & (R <= 2.0 * p.r_b)
        )
        self.T[pool_seed] = m.T_liquidus + 80.0
        self.alpha = np.clip(self.alpha, 0.0, 1.0)
        self.fL = self.liquid_fraction(self.T)

    # ----------------------------------------------------------- properties
    def liquid_fraction(self, T: np.ndarray) -> np.ndarray:
        m = self.mat
        return np.clip((T - m.T_solidus) / (m.T_liquidus - m.T_solidus), 0.0, 1.0)

    def _rho(self, alpha: Optional[np.ndarray] = None) -> np.ndarray:
        if alpha is None:
            alpha = self.alpha
        m = self.mat
        return m.rho * alpha + m.rho_g * (1.0 - alpha)

    def _mu(self, alpha: Optional[np.ndarray] = None) -> np.ndarray:
        if alpha is None:
            alpha = self.alpha
        m = self.mat
        return m.mu * alpha + m.mu_g * (1.0 - alpha)

    def _k(self, alpha: Optional[np.ndarray] = None, fL: Optional[np.ndarray] = None) -> np.ndarray:
        if alpha is None:
            alpha = self.alpha
        if fL is None:
            fL = self.fL
        m = self.mat
        k_m = m.k_s * (1.0 - fL) + m.k_l * fL
        return k_m * alpha + m.k_g * (1.0 - alpha)

    def _sigma_e(self, alpha: Optional[np.ndarray] = None, fL: Optional[np.ndarray] = None) -> np.ndarray:
        if alpha is None:
            alpha = self.alpha
        if fL is None:
            fL = self.fL
        m = self.mat
        se_m = m.sigma_e * fL + m.sigma_e_solid * (1.0 - fL)
        return se_m * alpha + m.sigma_e_gas * (1.0 - alpha)

    def _darcy_coeff(self, fL: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        m = self.mat
        A = m.mush_C * (1.0 - fL) ** 2 / (fL**3 + m.mush_B)
        return A * alpha

    def metal_volume(self) -> float:
        """Full-plate equivalent metal volume (2 × half-domain)."""
        return float(np.sum(self.alpha) * self.cell_vol * self._vol_scale)

    # -------------------------------------------------------------- masks
    def _flood_fill(self, seed: np.ndarray) -> np.ndarray:
        metal = self.alpha >= self.proc.alpha_metal
        mask = np.zeros_like(metal, dtype=bool)
        stack = list(zip(*np.where(seed & metal)))
        for i, j, k in stack:
            mask[i, j, k] = True
        while stack:
            i, j, k = stack.pop()
            for di, dj, dk in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ):
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < self.proc.nx and 0 <= nj < self.proc.ny and 0 <= nk < self.proc.nz:
                    if metal[ni, nj, nk] and not mask[ni, nj, nk]:
                        mask[ni, nj, nk] = True
                        stack.append((ni, nj, nk))
        return mask

    def _flood_fill_sub(self, allowed: np.ndarray, i0: int, j0: int, k0: int) -> np.ndarray:
        mask = np.zeros_like(allowed, dtype=bool)
        stack = [(i0, j0, k0)]
        mask[i0, j0, k0] = True
        while stack:
            i, j, k = stack.pop()
            for di, dj, dk in (
                (1, 0, 0),
                (-1, 0, 0),
                (0, 1, 0),
                (0, -1, 0),
                (0, 0, 1),
                (0, 0, -1),
            ):
                ni, nj, nk = i + di, j + dj, k + dk
                if 0 <= ni < self.proc.nx and 0 <= nj < self.proc.ny and 0 <= nk < self.proc.nz:
                    if allowed[ni, nj, nk] and not mask[ni, nj, nk]:
                        mask[ni, nj, nk] = True
                        stack.append((ni, nj, nk))
        return mask

    def _wire_disk_mask(self) -> np.ndarray:
        """Top-layer cells belonging to the moving wire inlet disk."""
        p = self.proc
        xe = self.electrode_x()
        R = self._r_cyl(xe)
        seed = np.zeros_like(self.alpha, dtype=bool)
        k_top = p.nz - 1
        seed[:, :, k_top] = (R[:, :, k_top] <= p.R_wire + 0.6 * self.dx) & (
            self.alpha[:, :, k_top] >= p.alpha_metal
        )
        if p.nz >= 2:
            seed[:, :, k_top - 1] = (R[:, :, k_top - 1] <= p.R_wire + 0.6 * self.dx) & (
                self.alpha[:, :, k_top - 1] >= p.alpha_metal
            )
        return seed

    def _update_masks(self) -> None:
        p = self.proc
        thr = p.alpha_metal
        metal = self.alpha >= thr

        seed_w = self._wire_disk_mask()
        # also seed any metal near tip in wire radius
        xe = self.electrode_x()
        R = self._r_cyl(xe)
        near_tip = (R <= p.R_wire + self.dx) & (self.Z >= p.z_wire_tip - p.R_drop0) & metal
        seed_w |= near_tip & (self.Z >= p.z_wire_tip - 0.2 * p.R_drop0)
        self.wire_mask = self._flood_fill(seed_w)

        seed_p = np.zeros_like(metal)
        k_plate = int(np.clip(np.searchsorted(self.zc, p.H_plate) - 1, 0, p.nz - 1))
        seed_p[:, :, : max(k_plate + 1, 1)] = metal[:, :, : max(k_plate + 1, 1)]
        self.pool_mask = self._flood_fill(seed_p)

        self.drop_mask = metal & ~self.wire_mask & ~self.pool_mask
        n_comp = int(self.wire_mask.any()) + int(self.pool_mask.any())
        if np.any(self.drop_mask):
            visited = np.zeros_like(metal)
            for i, j, k in zip(*np.where(self.drop_mask)):
                if visited[i, j, k]:
                    continue
                blob = self._flood_fill_sub(self.drop_mask, i, j, k)
                visited |= blob
                n_comp += 1
        self._n_components = n_comp
        if np.any(self.drop_mask) and not self._detached:
            self._detached = True

    def neck_radius(self) -> float:
        """Min metal radius from electrode axis in wire-connected gap above plate."""
        p = self.proc
        thr = p.alpha_metal
        xe = self.electrode_x()
        metal = self.alpha >= thr
        k0 = int(np.searchsorted(self.zc, p.H_plate + 0.2e-3))
        k1 = int(np.searchsorted(self.zc, min(p.z_wire_tip + p.R_drop0, p.Lz - self.dz)))
        k0 = max(0, k0)
        k1 = min(p.nz, max(k0 + 1, k1))
        r_min = p.R_wire
        found = False
        for k in range(k0, k1):
            cols = np.where(
                (self.wire_mask[:, :, k] | self.drop_mask[:, :, k]) & metal[:, :, k]
            )
            if cols[0].size == 0:
                continue
            dx = self.xc[cols[0]] - xe
            dy = self.yc[cols[1]]
            r_outer = float(np.max(np.sqrt(dx**2 + dy**2)) + 0.5 * min(self.dx, self.dy))
            r_min = min(r_min, r_outer)
            found = True
        return float(r_min) if found else 0.0

    # ------------------------------------------------------------------- EM
    def _metal_radius_profile(self, mask: np.ndarray) -> np.ndarray:
        """R_m(z) from wire-connected metal about electrode axis."""
        p = self.proc
        thr = p.alpha_metal
        xe = self.electrode_x()
        Rm = np.zeros(p.nz)
        for k in range(p.nz):
            ii, jj = np.where(mask[:, :, k] & (self.alpha[:, :, k] >= thr))
            if ii.size:
                dx = self.xc[ii] - xe
                dy = self.yc[jj]
                Rm[k] = float(np.max(np.sqrt(dx**2 + dy**2)) + 0.5 * min(self.dx, self.dy))
            else:
                Rm[k] = 0.0
        return Rm

    def _compute_pinch(self) -> None:
        p = self.proc
        self.Fx[:] = 0.0
        self.Fy[:] = 0.0
        self.Fz[:] = 0.0
        self.Jx[:] = 0.0
        self.Jy[:] = 0.0
        self.Jz[:] = 0.0
        if not p.enable_em or p.I_arc <= 0.0:
            return
        mask = self.wire_mask
        if not np.any(mask):
            return
        Rm = self._metal_radius_profile(mask)
        Rm_min = max(0.75 * min(self.dx, self.dy), 0.15 * p.R_wire)
        I = p.I_arc
        xe = self.electrode_x()
        for k in range(p.nz):
            Rm_k = max(Rm[k], Rm_min) if Rm[k] > 0.0 else 0.0
            if Rm_k < 1e-6:
                continue
            ii, jj = np.where(mask[:, :, k])
            for i, j in zip(ii, jj):
                dx = self.xc[i] - xe
                dy = self.yc[j]
                r = np.sqrt(dx**2 + dy**2)
                if r > Rm_k:
                    continue
                Jz = I / (np.pi * Rm_k**2)
                Bth = MU0 * I * r / (2.0 * np.pi * Rm_k**2)
                Fr = -Jz * Bth
                Fr = float(np.clip(Fr, -p.F_em_clip, p.F_em_clip))
                # F_r e_r → (Fx, Fy); e_r = (dx, dy)/r
                if r < 1e-12:
                    Fx = Fy = 0.0
                else:
                    Fx = Fr * dx / r
                    Fy = Fr * dy / r
                self.Jz[i, j, k] = Jz
                self.Fx[i, j, k] = Fx
                self.Fy[i, j, k] = Fy

    def _compute_pool_em(self) -> None:
        p = self.proc
        if not p.enable_em or p.I_arc <= 0.0:
            return
        mask = self.pool_mask
        if not np.any(mask):
            return

        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        se = self._sigma_e()
        idx_map = -np.ones((nx, ny, nz), dtype=np.int32)
        active = mask & (self.alpha >= p.alpha_metal)
        n_act = int(np.sum(active))
        if n_act < 2:
            return
        idx_map[active] = np.arange(n_act, dtype=np.int32)

        dirichlet = np.zeros_like(active)
        dirichlet[:, :, 0] = active[:, :, 0]
        if not np.any(dirichlet):
            for k in range(nz):
                if np.any(active[:, :, k]):
                    dirichlet[:, :, k] = active[:, :, k]
                    break

        surf = np.zeros_like(active)
        for i, j, k in zip(*np.where(active)):
            if k + 1 >= nz or not active[i, j, k + 1]:
                if self.zc[k] >= p.H_plate - 2.0e-3:
                    surf[i, j, k] = True

        xe = self.electrode_x()
        src_vol = np.zeros((nx, ny, nz))
        if np.any(surf):
            wgt = np.zeros((nx, ny, nz))
            R2 = (self.X - xe) ** 2 + self.Y**2
            wgt[surf] = np.exp(-3.0 * R2[surf] / p.r_b**2)
            tot = float(np.sum(wgt[surf]) * self.cell_vol)
            if tot > 0.0:
                src_vol[surf] = p.I_arc * wgt[surf] / tot

        rows, cols, data = [], [], []
        rhs = np.zeros(n_act)

        def add(a, b, val):
            rows.append(a)
            cols.append(b)
            data.append(val)

        for i, j, k in zip(*np.where(active)):
            row = int(idx_map[i, j, k])
            if dirichlet[i, j, k]:
                add(row, row, 1.0)
                rhs[row] = 0.0
                continue
            diag = 0.0
            for di, dj, dk, h in (
                (1, 0, 0, dx),
                (-1, 0, 0, dx),
                (0, 1, 0, dy),
                (0, -1, 0, dy),
                (0, 0, 1, dz),
                (0, 0, -1, dz),
            ):
                ni, nj, nk = i + di, j + dj, k + dk
                if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
                    continue
                if not active[ni, nj, nk]:
                    continue
                se_f = 2.0 / (1.0 / (se[i, j, k] + 1e-30) + 1.0 / (se[ni, nj, nk] + 1e-30))
                coef = se_f / h**2
                add(row, int(idx_map[ni, nj, nk]), coef)
                diag -= coef
            add(row, row, diag if diag != 0.0 else 1.0)
            rhs[row] = -src_vol[i, j, k]

        A = sparse.csr_matrix((data, (rows, cols)), shape=(n_act, n_act))
        try:
            sol = spsolve(A, rhs)
        except Exception:
            return
        if sol is None or np.any(~np.isfinite(sol)):
            return
        self.phi[:] = 0.0
        self.phi[active] = sol

        # J = -σ ∇φ on active cells (central / one-sided)
        Jx = np.zeros_like(self.phi)
        Jy = np.zeros_like(self.phi)
        Jz = np.zeros_like(self.phi)
        for i, j, k in zip(*np.where(active)):
            # dφ/dx
            if i + 1 < nx and active[i + 1, j, k] and i - 1 >= 0 and active[i - 1, j, k]:
                dphidx = (self.phi[i + 1, j, k] - self.phi[i - 1, j, k]) / (2 * dx)
            elif i + 1 < nx and active[i + 1, j, k]:
                dphidx = (self.phi[i + 1, j, k] - self.phi[i, j, k]) / dx
            elif i - 1 >= 0 and active[i - 1, j, k]:
                dphidx = (self.phi[i, j, k] - self.phi[i - 1, j, k]) / dx
            else:
                dphidx = 0.0
            if j + 1 < ny and active[i, j + 1, k] and j - 1 >= 0 and active[i, j - 1, k]:
                dphidy = (self.phi[i, j + 1, k] - self.phi[i, j - 1, k]) / (2 * dy)
            elif j + 1 < ny and active[i, j + 1, k]:
                dphidy = (self.phi[i, j + 1, k] - self.phi[i, j, k]) / dy
            elif j - 1 >= 0 and active[i, j - 1, k]:
                dphidy = (self.phi[i, j, k] - self.phi[i, j - 1, k]) / dy
            else:
                dphidy = 0.0
            if k + 1 < nz and active[i, j, k + 1] and k - 1 >= 0 and active[i, j, k - 1]:
                dphidz = (self.phi[i, j, k + 1] - self.phi[i, j, k - 1]) / (2 * dz)
            elif k + 1 < nz and active[i, j, k + 1]:
                dphidz = (self.phi[i, j, k + 1] - self.phi[i, j, k]) / dz
            elif k - 1 >= 0 and active[i, j, k - 1]:
                dphidz = (self.phi[i, j, k] - self.phi[i, j, k - 1]) / dz
            else:
                dphidz = 0.0
            s = se[i, j, k]
            Jx[i, j, k] = -s * dphidx
            Jy[i, j, k] = -s * dphidy
            Jz[i, j, k] = -s * dphidz

        # Approximate B_θ from vertical current about arc axis; F = J × B
        for i, j, k in zip(*np.where(active)):
            dx_ = self.xc[i] - xe
            dy_ = self.yc[j]
            r = np.sqrt(dx_**2 + dy_**2) + 1e-14
            # enclosed current proxy: use local Jz * π Rm^2 style — B from total I * r/Rm^2 capped
            # simpler: B_θ ≈ (μ0 I r) / (2π r_b^2) inside effective radius, Ampere outside
            if r < p.r_b:
                Bth = MU0 * p.I_arc * r / (2.0 * np.pi * p.r_b**2)
            else:
                Bth = MU0 * p.I_arc / (2.0 * np.pi * r)
            # e_θ = (-dy/r, dx/r, 0); J×B: Fx = Jy*0 - Jz*(-Bθ *? wait)
            # B = Bθ e_θ = Bθ (-dy/r, dx/r, 0)
            Bx = -Bth * dy_ / r
            By = Bth * dx_ / r
            Bz = 0.0
            Fx = Jy[i, j, k] * Bz - Jz[i, j, k] * By
            Fy = Jz[i, j, k] * Bx - Jx[i, j, k] * Bz
            Fz = Jx[i, j, k] * By - Jy[i, j, k] * Bx
            self.Fx[i, j, k] += float(np.clip(Fx, -p.F_em_clip, p.F_em_clip))
            self.Fy[i, j, k] += float(np.clip(Fy, -p.F_em_clip, p.F_em_clip))
            self.Fz[i, j, k] += float(np.clip(Fz, -p.F_em_clip, p.F_em_clip))
            self.Jx[i, j, k] = Jx[i, j, k]
            self.Jy[i, j, k] = Jy[i, j, k]
            self.Jz[i, j, k] = Jz[i, j, k]

    def update_em(self) -> None:
        self._compute_pinch()
        self._compute_pool_em()

    # --------------------------------------------------------------- VOF
    def _grad(self, f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Central gradients with Neumann at boundaries (symmetry at y=0)."""
        nx, ny, nz = self.proc.nx, self.proc.ny, self.proc.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        gx = np.zeros_like(f)
        gy = np.zeros_like(f)
        gz = np.zeros_like(f)
        gx[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2 * dx)
        gx[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
        gx[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
        if ny >= 2:
            gy[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2 * dy)
            gy[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy  # symmetry: ghost = f[:,0]
            gy[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
        gz[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2 * dz)
        gz[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
        gz[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
        return gx, gy, gz

    def _curvature(self, alpha: np.ndarray) -> np.ndarray:
        ax, ay, az = self._grad(alpha)
        mag = np.sqrt(ax**2 + ay**2 + az**2) + 1e-14
        nx_, ny_, nz_ = ax / mag, ay / mag, az / mag
        dnx_dx, _, _ = self._grad(nx_)
        _, dny_dy, _ = self._grad(ny_)
        _, _, dnz_dz = self._grad(nz_)
        return -(dnx_dx + dny_dy + dnz_dz)

    def _face_alpha_donor(self, alpha: np.ndarray, vel_face: np.ndarray, axis: str) -> np.ndarray:
        if axis == "x":
            # vel_face: (nx+1, ny, nz); return alpha on faces
            a_f = np.zeros_like(vel_face)
            a_f[1:-1, :, :] = np.where(
                vel_face[1:-1, :, :] >= 0.0,
                alpha[:-1, :, :],
                alpha[1:, :, :],
            )
            a_f[0, :, :] = alpha[0, :, :]
            a_f[-1, :, :] = alpha[-1, :, :]
            return a_f
        if axis == "y":
            a_f = np.zeros_like(vel_face)
            a_f[:, 1:-1, :] = np.where(
                vel_face[:, 1:-1, :] >= 0.0,
                alpha[:, :-1, :],
                alpha[:, 1:, :],
            )
            a_f[:, 0, :] = alpha[:, 0, :]
            a_f[:, -1, :] = alpha[:, -1, :]
            return a_f
        # z
        a_f = np.zeros_like(vel_face)
        a_f[:, :, 1:-1] = np.where(
            vel_face[:, :, 1:-1] >= 0.0,
            alpha[:, :, :-1],
            alpha[:, :, 1:],
        )
        a_f[:, :, 0] = alpha[:, :, 0]
        a_f[:, :, -1] = alpha[:, :, -1]
        return a_f

    def _advect_vof(self, dt: float) -> None:
        p = self.proc
        alpha = self.alpha
        dx, dy, dz = self.dx, self.dy, self.dz

        ax = self._face_alpha_donor(alpha, self.u, "x")
        flux_x = self.u * ax
        dadt = -(flux_x[1:, :, :] - flux_x[:-1, :, :]) / dx
        alpha1 = np.clip(alpha + dt * dadt, 0.0, 1.0)

        ay = self._face_alpha_donor(alpha1, self.v, "y")
        flux_y = self.v * ay
        dadt = -(flux_y[:, 1:, :] - flux_y[:, :-1, :]) / dy
        alpha2 = np.clip(alpha1 + dt * dadt, 0.0, 1.0)

        az = self._face_alpha_donor(alpha2, self.w, "z")
        flux_z = self.w * az
        dadt = -(flux_z[:, :, 1:] - flux_z[:, :, :-1]) / dz
        alpha3 = np.clip(alpha2 + dt * dadt, 0.0, 1.0)

        # moving wire inlet: force alpha=1 in top disk
        xe = self.electrode_x()
        R = np.sqrt((self.X[:, :, -1] - xe) ** 2 + self.Y[:, :, -1] ** 2)
        top_wire = R <= p.R_wire + 0.5 * self.dx
        sl = alpha3[:, :, -1]
        sl[top_wire] = 1.0
        if p.nz >= 2:
            R2 = np.sqrt((self.X[:, :, -2] - xe) ** 2 + self.Y[:, :, -2] ** 2)
            tw2 = R2 <= p.R_wire + 0.5 * self.dx
            sl2 = alpha3[:, :, -2]
            sl2[tw2] = np.maximum(sl2[tw2], 0.99)

        self.alpha = alpha3

    def _apply_wire_feed(self, dt: float) -> None:
        """Impose (v_weld, 0, -v_feed) in solid wire; accumulate fed volume."""
        p = self.proc
        m = self.mat
        xe = self.electrode_x()
        solid_wire = self.wire_mask & (self.T < m.T_solidus + 5.0) & (self.alpha >= p.alpha_metal)
        R = self._r_cyl(xe)
        inlet = (R <= p.R_wire + 0.5 * self.dx) & (self.Z >= p.Lz - 2.5 * self.dz)
        feed_region = solid_wire | inlet

        # set face velocities for feed column
        for i in range(p.nx):
            for j in range(p.ny):
                if not np.any(feed_region[i, j, :]):
                    # still check radius for tip column above tip
                    if (self.xc[i] - xe) ** 2 + self.yc[j] ** 2 > (p.R_wire + self.dx) ** 2:
                        continue
                in_col = ((self.xc[i] - xe) ** 2 + self.yc[j] ** 2) <= (p.R_wire + 0.6 * self.dx) ** 2
                if not in_col:
                    continue
                for k in range(p.nz + 1):
                    k_lo = max(k - 1, 0)
                    k_hi = min(k, p.nz - 1)
                    if feed_region[i, j, k_lo] or feed_region[i, j, k_hi] or self.zw[k] >= p.z_wire_tip:
                        if in_col:
                            self.w[i, j, k] = -p.v_feed
                for k in range(p.nz):
                    if feed_region[i, j, k] or (in_col and self.zc[k] >= p.z_wire_tip):
                        # u faces
                        self.u[i, j, k] = p.v_weld
                        self.u[i + 1, j, k] = p.v_weld
                        self.v[i, j, k] = 0.0
                        self.v[i, j + 1, k] = 0.0

        dV = 0.5 * np.pi * p.R_wire**2 * p.v_feed * dt  # half-disk feed; full equiv uses scale
        self._V_feed_cum += dV * self._vol_scale

    # -------------------------------------------------------- momentum
    def _cell_velocity(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u_c = 0.5 * (self.u[:-1, :, :] + self.u[1:, :, :])
        v_c = 0.5 * (self.v[:, :-1, :] + self.v[:, 1:, :])
        w_c = 0.5 * (self.w[:, :, :-1] + self.w[:, :, 1:])
        return u_c, v_c, w_c

    def _surface_forces(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.proc
        m = self.mat
        alpha = self.alpha
        ax, ay, az = self._grad(alpha)
        kappa = self._curvature(alpha)
        Fx = m.sigma * kappa * ax
        Fy = m.sigma * kappa * ay
        Fz = m.sigma * kappa * az
        Fcap = 2.0 * p.F_em_clip
        Fx = np.clip(Fx, -Fcap, Fcap)
        Fy = np.clip(Fy, -Fcap, Fcap)
        Fz = np.clip(Fz, -Fcap, Fcap)

        fL = self.fL
        Tx, Ty, Tz = self._grad(self.T)
        mag = np.sqrt(ax**2 + ay**2 + az**2) + 1e-14
        nx_, ny_, nz_ = ax / mag, ay / mag, az / mag
        dT_n = Tx * nx_ + Ty * ny_ + Tz * nz_
        Tx_t = Tx - dT_n * nx_
        Ty_t = Ty - dT_n * ny_
        Tz_t = Tz - dT_n * nz_
        mar = m.dsigma_dT * fL * alpha
        Fx += mar * Tx_t * mag
        Fy += mar * Ty_t * mag
        Fz += mar * Tz_t * mag

        if p.p_arc_max > 0.0:
            xe = self.electrode_x()
            R2 = (self.X - xe) ** 2 + self.Y**2
            p_arc = p.p_arc_max * np.exp(-3.0 * R2 / p.r_b**2)
            near_pool = self.pool_mask | (
                (self.Z <= p.H_plate + 1.5e-3) & (self.Z >= p.H_plate - 2e-3)
            )
            Fx += p_arc * ax * near_pool
            Fy += p_arc * ay * near_pool
            Fz += p_arc * az * near_pool

        return Fx, Fy, Fz

    def _step_momentum(self, dt: float) -> None:
        p = self.proc
        m = self.mat
        if not p.enable_flow:
            return
        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        alpha = self.alpha
        rho = self._rho(alpha)
        mu = self._mu(alpha)
        fL = self.fL
        A_darcy = self._darcy_coeff(fL, alpha)
        Fx_s, Fy_s, Fz_s = self._surface_forces()

        u, v, w = self.u, self.v, self.w
        u_star = u.copy()
        v_star = v.copy()
        w_star = w.copy()

        # ----- u -----
        if nx >= 2:
            for i in range(1, nx):
                rho_f = 0.5 * (rho[i - 1, :, :] + rho[i, :, :])
                mu_f = 0.5 * (mu[i - 1, :, :] + mu[i, :, :])
                A_f = 0.5 * (A_darcy[i - 1, :, :] + A_darcy[i, :, :])
                Fx_f = (
                    0.5 * (Fx_s[i - 1, :, :] + Fx_s[i, :, :])
                    + 0.5 * (self.Fx[i - 1, :, :] + self.Fx[i, :, :])
                )
                ui = u[i, :, :]
                ue = u[i + 1, :, :] if i + 1 <= nx else ui
                uw = u[i - 1, :, :]
                v_adv = 0.5 * (
                    0.5 * (v[i - 1, 1:, :] + v[i, 1:, :])
                    + 0.5 * (v[i - 1, :-1, :] + v[i, :-1, :])
                )
                w_adv = 0.5 * (
                    0.5 * (w[i - 1, :, 1:] + w[i, :, 1:])
                    + 0.5 * (w[i - 1, :, :-1] + w[i, :, :-1])
                )
                dudx = np.where(ui >= 0.0, (ui - uw) / dx, (ue - ui) / dx)
                # ui shape (ny, nz): y-neighbors along axis 0, z along axis 1
                un = np.empty_like(ui)
                us = np.empty_like(ui)
                un[:-1, :] = u[i, 1:, :]
                un[-1, :] = u[i, -1, :]
                us[1:, :] = u[i, :-1, :]
                us[0, :] = u[i, 0, :]
                dudy = np.where(v_adv >= 0.0, (ui - us) / dy, (un - ui) / dy)
                ut = np.empty_like(ui)
                ub = np.empty_like(ui)
                ut[:, :-1] = u[i, :, 1:]
                ut[:, -1] = u[i, :, -1]
                ub[:, 1:] = u[i, :, :-1]
                ub[:, 0] = u[i, :, 0]
                dudz = np.where(w_adv >= 0.0, (ui - ub) / dz, (ut - ui) / dz)
                adv = ui * dudx + v_adv * dudy + w_adv * dudz
                d2x = (ue - 2.0 * ui + uw) / dx**2
                d2y = (un - 2.0 * ui + us) / dy**2
                d2z = (ut - 2.0 * ui + ub) / dz**2
                visc = (mu_f / (rho_f + 1e-30)) * (d2x + d2y + d2z)
                body = Fx_f / (rho_f + 1e-30)
                u_star[i, :, :] = (ui + dt * (-adv + visc + body)) / (1.0 + dt * A_f / (rho_f + 1e-30))

        u_star[0, :, :] = 0.0
        u_star[-1, :, :] = 0.0

        # ----- v -----
        if ny >= 2:
            for j in range(1, ny):
                rho_f = 0.5 * (rho[:, j - 1, :] + rho[:, j, :])
                mu_f = 0.5 * (mu[:, j - 1, :] + mu[:, j, :])
                A_f = 0.5 * (A_darcy[:, j - 1, :] + A_darcy[:, j, :])
                Fy_f = (
                    0.5 * (Fy_s[:, j - 1, :] + Fy_s[:, j, :])
                    + 0.5 * (self.Fy[:, j - 1, :] + self.Fy[:, j, :])
                )
                vi = v[:, j, :]
                vn = v[:, j + 1, :] if j + 1 <= ny else vi
                vs = v[:, j - 1, :]
                u_adv = 0.5 * (
                    0.5 * (u[1:, j - 1, :] + u[1:, j, :])
                    + 0.5 * (u[:-1, j - 1, :] + u[:-1, j, :])
                )
                w_adv = 0.5 * (
                    0.5 * (w[:, j - 1, 1:] + w[:, j, 1:])
                    + 0.5 * (w[:, j - 1, :-1] + w[:, j, :-1])
                )
                ve = np.empty_like(vi)
                vw = np.empty_like(vi)
                ve[:-1, :] = v[1:, j, :]
                ve[-1, :] = v[-1, j, :]
                vw[1:, :] = v[:-1, j, :]
                vw[0, :] = v[0, j, :]
                dvdx = np.where(u_adv >= 0.0, (vi - vw) / dx, (ve - vi) / dx)
                dvdy = np.where(vi >= 0.0, (vi - vs) / dy, (vn - vi) / dy)
                vt = np.empty_like(vi)
                vb = np.empty_like(vi)
                vt[:, :-1] = v[:, j, 1:]
                vt[:, -1] = v[:, j, -1]
                vb[:, 1:] = v[:, j, :-1]
                vb[:, 0] = v[:, j, 0]
                dvdz = np.where(w_adv >= 0.0, (vi - vb) / dz, (vt - vi) / dz)
                adv = u_adv * dvdx + vi * dvdy + w_adv * dvdz
                d2y = (vn - 2.0 * vi + vs) / dy**2
                d2x = (ve - 2.0 * vi + vw) / dx**2
                d2z = (vt - 2.0 * vi + vb) / dz**2
                visc = (mu_f / (rho_f + 1e-30)) * (d2x + d2y + d2z)
                body = Fy_f / (rho_f + 1e-30)
                v_star[:, j, :] = (vi + dt * (-adv + visc + body)) / (1.0 + dt * A_f / (rho_f + 1e-30))

        v_star[:, 0, :] = 0.0
        v_star[:, -1, :] = 0.0

        # ----- w -----
        if nz >= 2:
            for k in range(1, nz):
                rho_f = 0.5 * (rho[:, :, k - 1] + rho[:, :, k])
                mu_f = 0.5 * (mu[:, :, k - 1] + mu[:, :, k])
                A_f = 0.5 * (A_darcy[:, :, k - 1] + A_darcy[:, :, k])
                Fz_f = (
                    0.5 * (Fz_s[:, :, k - 1] + Fz_s[:, :, k])
                    + 0.5 * (self.Fz[:, :, k - 1] + self.Fz[:, :, k])
                )
                wi = w[:, :, k]
                wt = w[:, :, k + 1] if k + 1 <= nz else wi
                wb = w[:, :, k - 1]
                u_adv = 0.5 * (
                    0.5 * (u[1:, :, k - 1] + u[1:, :, k])
                    + 0.5 * (u[:-1, :, k - 1] + u[:-1, :, k])
                )
                v_adv = 0.5 * (
                    0.5 * (v[:, 1:, k - 1] + v[:, 1:, k])
                    + 0.5 * (v[:, :-1, k - 1] + v[:, :-1, k])
                )
                we = np.empty_like(wi)
                ww = np.empty_like(wi)
                we[:-1, :] = w[1:, :, k]
                we[-1, :] = w[-1, :, k]
                ww[1:, :] = w[:-1, :, k]
                ww[0, :] = w[0, :, k]
                wn = np.empty_like(wi)
                ws = np.empty_like(wi)
                wn[:, :-1] = w[:, 1:, k]
                wn[:, -1] = w[:, -1, k]
                ws[:, 1:] = w[:, :-1, k]
                ws[:, 0] = w[:, 0, k]
                dwdx = np.where(u_adv >= 0.0, (wi - ww) / dx, (we - wi) / dx)
                dwdy = np.where(v_adv >= 0.0, (wi - ws) / dy, (wn - wi) / dy)
                dwdz = np.where(wi >= 0.0, (wi - wb) / dz, (wt - wi) / dz)
                adv = u_adv * dwdx + v_adv * dwdy + wi * dwdz
                d2z = (wt - 2.0 * wi + wb) / dz**2
                d2x = (we - 2.0 * wi + ww) / dx**2
                d2y = (wn - 2.0 * wi + ws) / dy**2
                visc = (mu_f / (rho_f + 1e-30)) * (d2x + d2y + d2z)
                grav = -p.g
                buoy = p.g * m.beta * (0.5 * (self.T[:, :, k - 1] + self.T[:, :, k]) - m.T_ref)
                fL_f = 0.5 * (fL[:, :, k - 1] + fL[:, :, k])
                a_f = 0.5 * (alpha[:, :, k - 1] + alpha[:, :, k])
                body = Fz_f / (rho_f + 1e-30) + grav + buoy * fL_f * a_f
                w_star[:, :, k] = (wi + dt * (-adv + visc + body)) / (1.0 + dt * A_f / (rho_f + 1e-30))

        w_star[:, :, 0] = 0.0
        w_star[:, :, -1] = w_star[:, :, -2]

        self._project(u_star, v_star, w_star, dt)
        uc = p.u_clip
        u_star[:] = np.clip(u_star, -uc, uc)
        v_star[:] = np.clip(v_star, -uc, uc)
        w_star[:] = np.clip(w_star, -uc, uc)
        self.u = u_star
        self.v = v_star
        self.w = w_star

    def _project(self, u_star, v_star, w_star, dt) -> None:
        p = self.proc
        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        rho = self._rho()
        N = nx * ny * nz

        def idx(i, j, k):
            return (i * ny + j) * nz + k

        div = (
            (u_star[1:, :, :] - u_star[:-1, :, :]) / dx
            + (v_star[:, 1:, :] - v_star[:, :-1, :]) / dy
            + (w_star[:, :, 1:] - w_star[:, :, :-1]) / dz
        )

        def rho_x(i, j, k):
            if i <= 0:
                return rho[0, j, k]
            if i >= nx:
                return rho[nx - 1, j, k]
            return 2.0 / (1.0 / rho[i - 1, j, k] + 1.0 / rho[i, j, k])

        def rho_y(i, j, k):
            if j <= 0:
                return rho[i, 0, k]
            if j >= ny:
                return rho[i, ny - 1, k]
            return 2.0 / (1.0 / rho[i, j - 1, k] + 1.0 / rho[i, j, k])

        def rho_z(i, j, k):
            if k <= 0:
                return rho[i, j, 0]
            if k >= nz:
                return rho[i, j, nz - 1]
            return 2.0 / (1.0 / rho[i, j, k - 1] + 1.0 / rho[i, j, k])

        rows, cols, data = [], [], []
        rhs = (div / dt).ravel(order="C")

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = idx(i, j, k)
                    if i == 0 and j == 0 and k == 0:
                        rows.append(r)
                        cols.append(r)
                        data.append(1.0)
                        rhs[r] = 0.0
                        continue
                    diag = 0.0
                    if i + 1 < nx:
                        coef = (1.0 / rho_x(i + 1, j, k)) / dx**2
                        rows.append(r)
                        cols.append(idx(i + 1, j, k))
                        data.append(coef)
                        diag -= coef
                    if i - 1 >= 0:
                        coef = (1.0 / rho_x(i, j, k)) / dx**2
                        rows.append(r)
                        cols.append(idx(i - 1, j, k))
                        data.append(coef)
                        diag -= coef
                    if j + 1 < ny:
                        coef = (1.0 / rho_y(i, j + 1, k)) / dy**2
                        rows.append(r)
                        cols.append(idx(i, j + 1, k))
                        data.append(coef)
                        diag -= coef
                    if j - 1 >= 0:
                        coef = (1.0 / rho_y(i, j, k)) / dy**2
                        rows.append(r)
                        cols.append(idx(i, j - 1, k))
                        data.append(coef)
                        diag -= coef
                    if k + 1 < nz:
                        coef = (1.0 / rho_z(i, j, k + 1)) / dz**2
                        rows.append(r)
                        cols.append(idx(i, j, k + 1))
                        data.append(coef)
                        diag -= coef
                    if k - 1 >= 0:
                        coef = (1.0 / rho_z(i, j, k)) / dz**2
                        rows.append(r)
                        cols.append(idx(i, j, k - 1))
                        data.append(coef)
                        diag -= coef
                    rows.append(r)
                    cols.append(r)
                    data.append(diag if diag != 0.0 else 1.0)

        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        try:
            phi = spsolve(A, rhs).reshape((nx, ny, nz), order="C")
        except Exception:
            return
        if phi is None or np.any(~np.isfinite(phi)):
            return
        self.p = phi

        for i in range(1, nx):
            for j in range(ny):
                for k in range(nz):
                    rf = rho_x(i, j, k)
                    u_star[i, j, k] -= (dt / rf) * (phi[i, j, k] - phi[i - 1, j, k]) / dx
        for i in range(nx):
            for j in range(1, ny):
                for k in range(nz):
                    rf = rho_y(i, j, k)
                    v_star[i, j, k] -= (dt / rf) * (phi[i, j, k] - phi[i, j - 1, k]) / dy
        for i in range(nx):
            for j in range(ny):
                for k in range(1, nz):
                    rf = rho_z(i, j, k)
                    w_star[i, j, k] -= (dt / rf) * (phi[i, j, k] - phi[i, j, k - 1]) / dz

        u_star[0, :, :] = 0.0
        u_star[-1, :, :] = 0.0
        v_star[:, 0, :] = 0.0
        v_star[:, -1, :] = 0.0
        w_star[:, :, 0] = 0.0

    # ----------------------------------------------------------- energy
    def _heat_sources(self) -> np.ndarray:
        p = self.proc
        ax, ay, az = self._grad(self.alpha)
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        q = np.zeros_like(self.alpha)
        xe = self.electrode_x()
        R2 = (self.X - xe) ** 2 + self.Y**2
        q_arc = p.P_arc * (3.0 / (np.pi * p.r_b**2)) * np.exp(-3.0 * R2 / p.r_b**2)
        hmin = min(self.dx, self.dy, self.dz)
        near_pool_iface = mag > 0.1 / hmin
        near_pool_iface &= (self.Z <= p.H_plate + 2e-3) & (self.Z >= p.H_plate - 2.5e-3)
        q += q_arc * mag * near_pool_iface

        tip_iface = mag > 0.1 / hmin
        tip_iface &= self.wire_mask | (
            (self.Z > p.H_plate + 0.5e-3) & (np.sqrt(R2) <= 3 * p.R_wire)
        )
        if np.any(tip_iface):
            area_w = float(np.sum(mag[tip_iface]) * self.cell_vol)
            if area_w > 0:
                # half-domain: use half of anode power for half-wire surface
                q_anode = 0.5 * p.I_arc * p.U_anode
                q += (q_anode / area_w) * mag * tip_iface

        se = self._sigma_e()
        J2 = self.Jx**2 + self.Jy**2 + self.Jz**2
        q += J2 / (se + 1e-30)

        T = self.T
        q_loss = p.h_conv * (T - p.Ta) + p.emissivity * SIGMA_SB * (T**4 - p.Ta**4)
        q -= q_loss * mag * (self.alpha > 0.05) * (self.alpha < 0.95)
        return q

    def _step_energy(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        nx, ny, nz = p.nx, p.ny, p.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        T = self.T
        alpha = self.alpha
        fL = self.liquid_fraction(T)
        k = self._k(alpha, fL)
        rho = self._rho(alpha)
        cp_ref = 0.5 * (m.cp_s + m.cp_l)
        cp = cp_ref * alpha + m.cp_g * (1.0 - alpha)
        u_c, v_c, w_c = self._cell_velocity()

        Fx = np.zeros((nx + 1, ny, nz))
        Fx[1:-1, :, :] = 0.5 * (k[:-1, :, :] + k[1:, :, :]) * (T[1:, :, :] - T[:-1, :, :]) / dx
        Fy = np.zeros((nx, ny + 1, nz))
        Fy[:, 1:-1, :] = 0.5 * (k[:, :-1, :] + k[:, 1:, :]) * (T[:, 1:, :] - T[:, :-1, :]) / dy
        # symmetry: zero flux at y=0
        Fy[:, 0, :] = 0.0
        Fz = np.zeros((nx, ny, nz + 1))
        Fz[:, :, 1:-1] = 0.5 * (k[:, :, :-1] + k[:, :, 1:]) * (T[:, :, 1:] - T[:, :, :-1]) / dz
        Fz[:, :, 0] = p.h_bottom * (T[:, :, 0] - p.Ta)
        Fz[:, :, -1] = 0.0

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
        Ts[:, 0, :] = T[:, 0, :]
        Tt[:, :, :-1] = T[:, :, 1:]
        Tt[:, :, -1] = T[:, :, -1]
        Tb[:, :, 1:] = T[:, :, :-1]
        Tb[:, :, 0] = T[:, :, 0]
        dTdx = np.where(u_c >= 0.0, (T - Tw) / dx, (Te - T) / dx)
        dTdy = np.where(v_c >= 0.0, (T - Ts) / dy, (Tn - T) / dy)
        dTdz = np.where(w_c >= 0.0, (T - Tb) / dz, (Tt - T) / dz)
        adv = u_c * dTdx + v_c * dTdy + w_c * dTdz

        q_src = self._heat_sources()
        dT_mush = m.T_liquidus - m.T_solidus
        in_mush = (T > m.T_solidus) & (T < m.T_liquidus) & (alpha > 0.5)
        cp_eff = cp + np.where(in_mush, m.latent / dT_mush, 0.0)
        rhs = div_cond - rho * cp * adv + q_src
        T_star = T + dt / (rho * cp_eff + 1e-30) * rhs

        will_mush = (
            ((T <= m.T_solidus) & (T_star > m.T_solidus))
            | ((T >= m.T_liquidus) & (T_star < m.T_liquidus))
        ) & (alpha > 0.5)
        if np.any(will_mush):
            cp2 = cp + m.latent / dT_mush
            T_star = np.where(will_mush, T + dt / (rho * cp2 + 1e-30) * rhs, T_star)

        xe = self.electrode_x()
        R = np.sqrt((self.X[:, :, -1] - xe) ** 2 + self.Y[:, :, -1] ** 2)
        top_wire = R <= p.R_wire + 0.5 * self.dx
        T_star[:, :, -1][top_wire] = p.T_wire

        self.T = np.clip(T_star, 200.0, 3500.0)
        self.fL = self.liquid_fraction(self.T)

    # -------------------------------------------------------------- time
    def suggest_dt(self) -> float:
        p = self.proc
        m = self.mat
        u_c, v_c, w_c = self._cell_velocity()
        umax = float(np.max(np.abs(u_c))) + 1e-9
        vmax = float(np.max(np.abs(v_c))) + 1e-9
        wmax = float(np.max(np.abs(w_c))) + 1e-9
        dt_cfl = p.cfl / (umax / self.dx + vmax / self.dy + wmax / self.dz)
        h = min(self.dx, self.dy, self.dz)
        dt_cap = p.cfl * np.sqrt(m.rho * h**3 / (np.pi * m.sigma + 1e-30))
        dt_visc = 0.2 * m.rho * h**2 / (m.mu + 1e-30)
        return float(min(p.dt_max, dt_cfl, dt_cap, dt_visc))

    def step(self, dt: Optional[float] = None) -> float:
        if dt is None:
            dt = self.suggest_dt()
        self._update_masks()
        if self._step_count % max(1, self.proc.em_every) == 0:
            self.update_em()
        self._step_momentum(dt)
        self._apply_wire_feed(dt)
        self._advect_vof(dt)
        self._step_energy(dt)
        self._update_masks()
        self.t += dt
        self._step_count += 1
        return dt

    def record(self, record_fields: bool = False) -> None:
        u_c, v_c, w_c = self._cell_velocity()
        speed = np.sqrt(u_c**2 + v_c**2 + w_c**2)
        V_m = self.metal_volume()
        V_w = float(np.sum(self.alpha * self.wire_mask) * self.cell_vol * self._vol_scale)
        V_p = float(np.sum(self.alpha * self.pool_mask) * self.cell_vol * self._vol_scale)
        self.diag.time.append(self.t)
        self.diag.metal_volume.append(V_m)
        self.diag.wire_volume.append(V_w)
        self.diag.pool_volume.append(V_p)
        self.diag.n_components.append(getattr(self, "_n_components", 1))
        self.diag.neck_radius.append(self.neck_radius())
        self.diag.electrode_x.append(self.electrode_x())
        self.diag.T_max.append(float(self.T.max()))
        self.diag.max_speed.append(float(np.max(speed)))
        self.diag.detached.append(self._detached)
        if record_fields:
            # store midplane y≈0 slice for compact frames
            self.diag.alpha_frames.append(self.alpha[:, 0, :].copy())
            self.diag.T_frames.append(self.T[:, 0, :].copy())
            self.diag.speed_frames.append(speed[:, 0, :].copy())
            self.diag.time_frames.append(self.t)
            self.diag.xe_frames.append(self.electrode_x())

    def volume_balance_ratio(self) -> float:
        dV = self.metal_volume() - self._V_metal0
        den = max(self._V_feed_cum, self._V_metal0, 1e-12)
        return float(abs(dV - self._V_feed_cum) / den)

    def slice_y0(self):
        """Return (x_mm, z_mm, alpha, T) on first y-cell (near symmetry plane)."""
        return (
            1e3 * self.xc,
            1e3 * self.zc,
            self.alpha[:, 0, :],
            self.T[:, 0, :],
        )

    def run(
        self,
        t_end: Optional[float] = None,
        record_every: float = 0.001,
        callback: Optional[Callable] = None,
        progress: bool = True,
        record_fields: bool = False,
    ) -> Diagnostics:
        if t_end is None:
            t_end = self.proc.t_end
        if self.diag.time:
            next_rec = self.diag.time[-1] + record_every
        else:
            self.record(record_fields=record_fields)
            next_rec = self.t + record_every
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
                    print(
                        f"t={1e3*self.t:.2f}ms  xe={1e3*self.electrode_x():.2f}mm  "
                        f"Tmax={self.T.max():.0f}K  "
                        f"Rneck={1e3*self.neck_radius():.3f}mm  "
                        f"comp={getattr(self, '_n_components', 1)}  "
                        f"|u|max={self.diag.max_speed[-1]:.3f}m/s  "
                        f"det={self._detached}  "
                        f"step={1e3*dt_wall:.0f}ms  "
                        f"wall={time.perf_counter()-wall0:.1f}s",
                        flush=True,
                    )
                if callback is not None:
                    callback(self)
        if not self.diag.time or self.diag.time[-1] < self.t - 1e-12:
            self.record(record_fields=record_fields)
        return self.diag


def quick_process(**overrides) -> Process:
    base = dict(
        Lx=0.010,
        Ly=0.003,
        Lz=0.007,
        nx=40,
        ny=12,
        nz=28,
        H_plate=0.003,
        R_wire=0.0006,
        z_wire_tip=0.0055,
        R_drop0=0.00115,
        v_feed=0.04,
        v_weld=0.12,
        x0=0.003,
        I_arc=170.0,
        P_arc=1600.0,
        r_b=0.0025,
        p_arc_max=250.0,
        t_end=0.012,
        dt_max=4.0e-5,
        cfl=0.2,
        em_every=4,
        u_clip=4.0,
    )
    base.update(overrides)
    return Process(**base)
