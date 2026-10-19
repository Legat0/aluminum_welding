# Axisymmetric (r–z) electrode–droplet–weld-pool model.
# Single-grid VOF: consumable wire, hanging drop, free-surface pool.
# EM in metal only (pinch in wire/drop, potential in pool). No arc plasma.
# Default properties: Al 5182 (Zhao & DebRoy 2001).
# Domain: r from axis, z from plate bottom (z=0) upward.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import time
import warnings

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
    sigma: float = 0.90  # N/m surface tension
    dsigma_dT: float = -3.5e-4
    latent: float = 3.95e5
    beta: float = 1.0e-4
    T_ref: float = 911.0
    mush_C: float = 1.0e7
    mush_B: float = 1.0e-3
    sigma_e: float = 4.0e6  # S/m liquid Al electrical conductivity
    sigma_e_solid: float = 1.5e7
    sigma_e_gas: float = 1.0e-6


@dataclass
class Process:
    Lr: float = 0.006
    Lz: float = 0.010
    nr: int = 48
    nz: int = 80
    H_plate: float = 0.003
    R_wire: float = 0.0006
    z_wire_tip: float = 0.0075  # initial solid tip z (above plate)
    R_drop0: float = 0.0011  # initial hanging-drop radius (near critical)
    v_feed: float = 0.05  # m/s wire feed (downward)
    I_arc: float = 160.0  # A
    U_anode: float = 4.0  # V equivalent anode fall (heat on drop surface)
    P_arc: float = 1800.0  # W Gaussian heat on pool surface
    r_b: float = 0.0025
    p_arc_max: float = 300.0  # Pa peak arc pressure on pool
    T0: float = 300.0
    Ta: float = 300.0
    T_wire: float = 900.0  # wire / drop initial temperature (near melt)
    T_drop: float = 1200.0
    h_conv: float = 20.0
    h_bottom: float = 50.0
    emissivity: float = 0.4
    g: float = 9.81
    enable_flow: bool = True
    enable_em: bool = True
    cfl: float = 0.20
    dt_max: float = 5.0e-5
    t_end: float = 0.020
    em_every: int = 3  # EM solve every N hydro steps
    alpha_metal: float = 0.5  # threshold for metal mask
    u_clip: float = 5.0  # m/s hard velocity limiter (demo stability)
    F_em_clip: float = 5.0e6  # N/m³ Lorentz force cap


@dataclass
class Diagnostics:
    time: list = field(default_factory=list)
    metal_volume: list = field(default_factory=list)
    wire_volume: list = field(default_factory=list)
    pool_volume: list = field(default_factory=list)
    n_components: list = field(default_factory=list)
    neck_radius: list = field(default_factory=list)
    pool_depression: list = field(default_factory=list)
    T_max: list = field(default_factory=list)
    max_speed: list = field(default_factory=list)
    detached: list = field(default_factory=list)
    alpha_frames: list = field(default_factory=list)
    T_frames: list = field(default_factory=list)
    speed_frames: list = field(default_factory=list)
    Fr_frames: list = field(default_factory=list)
    time_frames: list = field(default_factory=list)


class WeldPoolAxi:
    """Axisymmetric electrode–droplet–pool VOF solver."""

    def __init__(self, material: Optional[Material] = None, process: Optional[Process] = None):
        self.mat = material or Material()
        self.proc = process or Process()
        self._build_grid()
        self.reset()

    # ------------------------------------------------------------------ grid
    def _build_grid(self) -> None:
        p = self.proc
        self.dr = p.Lr / p.nr
        self.dz = p.Lz / p.nz
        self.rc = (np.arange(p.nr) + 0.5) * self.dr
        self.zc = (np.arange(p.nz) + 0.5) * self.dz
        self.ru = np.arange(p.nr + 1) * self.dr  # radial faces
        self.zw = np.arange(p.nz + 1) * self.dz  # axial faces
        self.R, self.Z = np.meshgrid(self.rc, self.zc, indexing="ij")
        # cell "volume" factor 2π r Δr Δz — use 2π r for integrals
        self.cell_area = 2.0 * np.pi * self.rc[:, None] * self.dr * self.dz  # (nr,1)*(1)->broadcast
        self.cell_area = 2.0 * np.pi * self.R * self.dr * self.dz

    def reset(self) -> None:
        p = self.proc
        m = self.mat
        self.t = 0.0
        self._step_count = 0
        self.alpha = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.T = np.full((p.nr, p.nz), p.T0, dtype=np.float64)
        self.fL = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.u = np.zeros((p.nr + 1, p.nz), dtype=np.float64)  # u_r
        self.w = np.zeros((p.nr, p.nz + 1), dtype=np.float64)  # u_z
        self.p = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.phi = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.Jr = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.Jz = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.Bth = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.Fr = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.Fz = np.zeros((p.nr, p.nz), dtype=np.float64)
        self.wire_mask = np.zeros((p.nr, p.nz), dtype=bool)
        self.pool_mask = np.zeros((p.nr, p.nz), dtype=bool)
        self.drop_mask = np.zeros((p.nr, p.nz), dtype=bool)
        self._detached = False
        self._V_metal0 = 0.0
        self._V_feed_cum = 0.0
        self.diag = Diagnostics()
        self._init_geometry()
        self.fL = self.liquid_fraction(self.T)
        self._V_metal0 = self.metal_volume()
        self._update_masks()

    def _init_geometry(self) -> None:
        """Plate + wire + hanging spherical drop nearly critical."""
        p = self.proc
        m = self.mat
        R, Z = self.R, self.Z

        # plate
        plate = Z <= p.H_plate
        self.alpha[plate] = 1.0
        self.T[plate] = p.T0

        # solid wire column from tip up to domain top
        wire = (R <= p.R_wire) & (Z >= p.z_wire_tip)
        self.alpha[wire] = 1.0
        self.T[wire] = p.T_wire
        # feed velocity in solid wire
        self.w[:, :] = 0.0
        for j in range(p.nz + 1):
            z_face = self.zw[j]
            if z_face >= p.z_wire_tip:
                self.w[R[:, 0] <= p.R_wire, j] = -p.v_feed

        # hanging drop: sphere centred on axis below tip
        z_c = p.z_wire_tip - 0.15 * p.R_drop0
        drop = (R**2 + (Z - z_c) ** 2) <= p.R_drop0**2
        drop &= Z > p.H_plate  # not overlapping solid plate top
        drop &= Z < p.z_wire_tip + 0.5 * p.R_wire
        self.alpha[drop] = np.maximum(self.alpha[drop], 1.0)
        self.T[drop] = p.T_drop

        # smooth plate top free surface slightly recessed under arc (optional pool seed)
        # leave plate surface flat; melting creates free surface motion later
        self.alpha = np.clip(self.alpha, 0.0, 1.0)

        # melt a shallow pool seed under the arc so free surface can deform
        pool_seed = (Z <= p.H_plate) & (Z >= p.H_plate - 0.8e-3) & (R <= 2.0 * p.r_b)
        self.T[pool_seed] = m.T_liquidus + 80.0
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
        return A * alpha  # only in metal

    def metal_volume(self) -> float:
        return float(np.sum(self.alpha * self.cell_area))

    # -------------------------------------------------------- connectivity
    def _flood_fill(self, seed: np.ndarray) -> np.ndarray:
        """4-connected flood fill on metal cells (alpha > threshold)."""
        thr = self.proc.alpha_metal
        metal = self.alpha >= thr
        mask = np.zeros_like(metal, dtype=bool)
        stack = list(zip(*np.where(seed & metal)))
        for i, j in stack:
            mask[i, j] = True
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.proc.nr and 0 <= nj < self.proc.nz:
                    if metal[ni, nj] and not mask[ni, nj]:
                        mask[ni, nj] = True
                        stack.append((ni, nj))
        return mask

    def _update_masks(self) -> None:
        p = self.proc
        thr = p.alpha_metal
        metal = self.alpha >= thr

        # wire seed: top boundary cells with r < R_wire
        seed_w = np.zeros_like(metal)
        j_top = p.nz - 1
        seed_w[(self.rc <= p.R_wire) & metal[:, j_top], j_top] = True
        # also any metal in top few rows in wire radius
        for jj in range(max(0, p.nz - 3), p.nz):
            seed_w[(self.rc <= p.R_wire) & metal[:, jj], jj] = True
        self.wire_mask = self._flood_fill(seed_w)

        # pool seed: bottom plate region
        seed_p = np.zeros_like(metal)
        j_plate = int(np.clip(np.searchsorted(self.zc, p.H_plate) - 1, 0, p.nz - 1))
        seed_p[:, : max(j_plate + 1, 1)] = metal[:, : max(j_plate + 1, 1)]
        self.pool_mask = self._flood_fill(seed_p)

        # free / flying metal = metal not connected to wire or pool
        self.drop_mask = metal & ~self.wire_mask & ~self.pool_mask
        # hanging drop still on wire is part of wire_mask
        n_comp = int(self.wire_mask.any()) + int(self.pool_mask.any())
        if np.any(self.drop_mask):
            # count flying blobs roughly
            visited = np.zeros_like(metal)
            for i, j in zip(*np.where(self.drop_mask)):
                if visited[i, j]:
                    continue
                blob = self._flood_fill_sub(self.drop_mask, i, j)
                visited |= blob
                n_comp += 1
        self._n_components = n_comp
        if np.any(self.drop_mask) and not self._detached:
            self._detached = True

    def _flood_fill_sub(self, allowed: np.ndarray, i0: int, j0: int) -> np.ndarray:
        mask = np.zeros_like(allowed, dtype=bool)
        stack = [(i0, j0)]
        mask[i0, j0] = True
        while stack:
            i, j = stack.pop()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < self.proc.nr and 0 <= nj < self.proc.nz:
                    if allowed[ni, nj] and not mask[ni, nj]:
                        mask[ni, nj] = True
                        stack.append((ni, nj))
        return mask

    def neck_radius(self) -> float:
        """Minimum metal radius along wire+drop column above the plate."""
        p = self.proc
        thr = p.alpha_metal
        metal = self.alpha >= thr
        # look in gap between plate and top
        j0 = int(np.searchsorted(self.zc, p.H_plate + 0.2e-3))
        j1 = int(np.searchsorted(self.zc, min(p.z_wire_tip + p.R_drop0, p.Lz - self.dz)))
        j0 = max(0, j0)
        j1 = min(p.nz, max(j0 + 1, j1))
        r_min = p.R_wire
        found = False
        for j in range(j0, j1):
            cols = np.where(metal[:, j])[0]
            if cols.size == 0:
                continue
            # only consider if connected to wire path (wire_mask or nearby)
            if not (self.wire_mask[:, j].any() or self.drop_mask[:, j].any()):
                continue
            r_outer = float(self.rc[cols.max()] + 0.5 * self.dr)
            r_min = min(r_min, r_outer)
            found = True
        return float(r_min) if found else 0.0

    def pool_depression(self) -> float:
        """Depth of free-surface dip on axis relative to H_plate."""
        p = self.proc
        thr = p.alpha_metal
        # find top metal cell on axis (i=0)
        metal_axis = self.alpha[0, :] >= thr
        if not np.any(metal_axis):
            return 0.0
        j_top = int(np.where(metal_axis)[0].max())
        z_top = float(self.zc[j_top] + 0.5 * self.dz * (2.0 * self.alpha[0, j_top] - 1.0))
        return float(max(0.0, p.H_plate - z_top))

    # ------------------------------------------------------------------- EM
    def _metal_radius_profile(self, mask: np.ndarray) -> np.ndarray:
        """R_m(z) from alpha for cells in mask; shape (nz,)."""
        thr = self.proc.alpha_metal
        Rm = np.zeros(self.proc.nz)
        for j in range(self.proc.nz):
            cols = np.where(mask[:, j] & (self.alpha[:, j] >= thr))[0]
            if cols.size:
                Rm[j] = float(self.rc[cols.max()] + 0.5 * self.dr)
            else:
                Rm[j] = 0.0
        return Rm

    def _compute_pinch(self) -> None:
        """Analytical pinch force in wire-connected metal."""
        p = self.proc
        self.Fr[:] = 0.0
        self.Fz[:] = 0.0
        self.Jr[:] = 0.0
        self.Jz[:] = 0.0
        self.Bth[:] = 0.0
        if not p.enable_em or p.I_arc <= 0.0:
            return

        mask = self.wire_mask
        if not np.any(mask):
            return
        Rm = self._metal_radius_profile(mask)
        Rm_min = max(0.75 * self.dr, 0.15 * p.R_wire)
        I = p.I_arc
        for j in range(p.nz):
            Rm_j = max(Rm[j], Rm_min) if Rm[j] > 0.0 else 0.0
            if Rm_j < 1e-6:
                continue
            for i in range(p.nr):
                if not mask[i, j]:
                    continue
                r = self.rc[i]
                if r > Rm_j:
                    continue
                Jz = I / (np.pi * Rm_j**2)
                Bth = MU0 * I * r / (2.0 * np.pi * Rm_j**2)
                Fr = -Jz * Bth
                Fr = float(np.clip(Fr, -p.F_em_clip, p.F_em_clip))
                self.Jz[i, j] = Jz
                self.Bth[i, j] = Bth
                self.Fr[i, j] = Fr

    def _compute_pool_em(self) -> None:
        """Potential solve in pool metal grounded at the bottom; add J×B.

        Only the metal component connected to the ground plane enters the
        linear system — floating fragments (spray / VOF debris) are skipped
        so the matrix stays non-singular.
        """
        p = self.proc
        if not p.enable_em or p.I_arc <= 0.0:
            return
        mask = self.pool_mask
        if not np.any(mask):
            return

        nr, nz = p.nr, p.nz
        dr, dz = self.dr, self.dz
        se = self._sigma_e()
        metal = mask & (self.alpha >= p.alpha_metal)

        # Ground seeds on the bottom row, then keep only the connected component
        seed = np.zeros_like(metal)
        seed[:, 0] = metal[:, 0]
        if not np.any(seed):
            for j in range(nz):
                if np.any(metal[:, j]):
                    seed[:, j] = metal[:, j]
                    break
        if not np.any(seed):
            return
        active = self._flood_fill(seed) & metal
        n_act = int(np.sum(active))
        if n_act < 2:
            return

        idx_map = -np.ones((nr, nz), dtype=np.int32)
        idx_map[active] = np.arange(n_act, dtype=np.int32)

        dirichlet = np.zeros_like(active)
        dirichlet[:, 0] = active[:, 0]
        if not np.any(dirichlet):
            for j in range(nz):
                if np.any(active[:, j]):
                    dirichlet[:, j] = active[:, j]
                    break

        surf = np.zeros_like(active)
        for i, j in zip(*np.where(active)):
            if (j + 1 >= nz or not active[i, j + 1]) and self.zc[j] >= p.H_plate - 2.0e-3:
                surf[i, j] = True

        src_vol = np.zeros((nr, nz))
        if np.any(surf):
            wgt = np.zeros((nr, nz))
            wgt[surf] = np.exp(-3.0 * (self.R[surf] / p.r_b) ** 2)
            tot = float(np.sum(wgt[surf] * self.cell_area[surf]))
            if tot > 0.0:
                src_vol[surf] = p.I_arc * wgt[surf] / tot

        rows, cols, data = [], [], []
        rhs = np.zeros(n_act)

        def add(a, b, val):
            rows.append(a)
            cols.append(b)
            data.append(val)

        for i, j in zip(*np.where(active)):
            row = int(idx_map[i, j])
            if dirichlet[i, j]:
                add(row, row, 1.0)
                rhs[row] = 0.0
                continue

            diag = 0.0
            r_c = self.rc[i]
            for ni, face_r in (
                (i + 1, r_c + 0.5 * dr),
                (i - 1, max(r_c - 0.5 * dr, 0.0)),
            ):
                if ni < 0 or ni >= nr or not active[ni, j]:
                    continue
                se_f = 0.5 * (se[i, j] + se[ni, j])
                area = 2.0 * np.pi * face_r * dz
                c = se_f * area / dr
                add(row, int(idx_map[ni, j]), c)
                diag -= c

            for nj in (j + 1, j - 1):
                if nj < 0 or nj >= nz or not active[i, nj]:
                    continue
                se_f = 0.5 * (se[i, j] + se[i, nj])
                area = 2.0 * np.pi * r_c * dr
                c = se_f * area / dz
                add(row, int(idx_map[i, nj]), c)
                diag -= c

            # mild ground leak so sparse islands cannot float
            diag -= 1.0e-6 * (2.0 * np.pi * r_c * dr * dz)
            if abs(diag) < 1e-30:
                add(row, row, 1.0)
                rhs[row] = 0.0
            else:
                add(row, row, diag)
                rhs[row] = -src_vol[i, j] * self.cell_area[i, j]

        A = sparse.csr_matrix((data, (rows, cols)), shape=(n_act, n_act))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*singular.*")
            try:
                phi_vec = spsolve(A, rhs)
            except Exception:
                return
        if phi_vec is None or np.any(~np.isfinite(phi_vec)):
            return
        if float(np.max(np.abs(phi_vec))) > 1.0e3:
            return

        self.phi[:] = 0.0
        self.phi[active] = phi_vec

        Jr = np.zeros((nr, nz))
        Jz = np.zeros((nr, nz))
        for i, j in zip(*np.where(active)):
            if i + 1 < nr and active[i + 1, j] and i > 0 and active[i - 1, j]:
                dphdr = (self.phi[i + 1, j] - self.phi[i - 1, j]) / (2 * dr)
            elif i + 1 < nr and active[i + 1, j]:
                dphdr = (self.phi[i + 1, j] - self.phi[i, j]) / dr
            elif i > 0 and active[i - 1, j]:
                dphdr = (self.phi[i, j] - self.phi[i - 1, j]) / dr
            else:
                dphdr = 0.0
            if j + 1 < nz and active[i, j + 1] and j > 0 and active[i, j - 1]:
                dphdz = (self.phi[i, j + 1] - self.phi[i, j - 1]) / (2 * dz)
            elif j + 1 < nz and active[i, j + 1]:
                dphdz = (self.phi[i, j + 1] - self.phi[i, j]) / dz
            elif j > 0 and active[i, j - 1]:
                dphdz = (self.phi[i, j] - self.phi[i, j - 1]) / dz
            else:
                dphdz = 0.0
            Jr[i, j] = -se[i, j] * dphdr
            Jz[i, j] = -se[i, j] * dphdz

        Bth = np.zeros((nr, nz))
        for j in range(nz):
            integ = 0.0
            for i in range(nr):
                integ += Jz[i, j] * self.rc[i] * dr
                Bth[i, j] = MU0 * integ / (self.rc[i] + 1e-16)

        pool_only = active & ~self.wire_mask
        self.Jr[pool_only] = Jr[pool_only]
        self.Jz[pool_only] = Jz[pool_only]
        self.Bth[pool_only] = Bth[pool_only]
        Fr_add = np.clip(-Jz[pool_only] * Bth[pool_only], -p.F_em_clip, p.F_em_clip)
        Fz_add = np.clip(Jr[pool_only] * Bth[pool_only], -p.F_em_clip, p.F_em_clip)
        self.Fr[pool_only] += Fr_add
        self.Fz[pool_only] += Fz_add

    def update_em(self) -> None:
        self._compute_pinch()
        self._compute_pool_em()

    # -------------------------------------------------------- interface geo
    def _youngs_normal(self, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Outward-from-metal normal ≈ -∇α / |∇α| direction stored as ∇α components."""
        dr, dz = self.dr, self.dz
        nr, nz = alpha.shape
        ar = np.zeros_like(alpha)
        az = np.zeros_like(alpha)
        ar[1:-1, :] = (alpha[2:, :] - alpha[:-2, :]) / (2 * dr)
        ar[0, :] = (alpha[1, :] - alpha[0, :]) / dr
        ar[-1, :] = (alpha[-1, :] - alpha[-2, :]) / dr
        az[:, 1:-1] = (alpha[:, 2:] - alpha[:, :-2]) / (2 * dz)
        az[:, 0] = (alpha[:, 1] - alpha[:, 0]) / dz
        az[:, -1] = (alpha[:, -1] - alpha[:, -2]) / dz
        return ar, az

    def _curvature(self, alpha: np.ndarray) -> np.ndarray:
        """κ = ∇·n, n = ∇α / |∇α|; axisymmetric divergence."""
        ar, az = self._youngs_normal(alpha)
        mag = np.sqrt(ar**2 + az**2) + 1e-14
        nr_c = ar / mag
        nz_c = az / mag
        dr, dz = self.dr, self.dz
        # ∂(r n_r)/∂r / r + ∂n_z/∂z
        r = self.rc[:, None]
        # pad for derivatives
        d_rnr = np.zeros_like(alpha)
        d_rnr[1:-1, :] = (r[2:] * nr_c[2:, :] - r[:-2] * nr_c[:-2, :]) / (2 * dr)
        d_rnr[0, :] = ((r[1] * nr_c[1, :]) - 0.0) / dr  # r=0 → n_r=0
        d_rnr[-1, :] = (r[-1] * nr_c[-1, :] - r[-2] * nr_c[-2, :]) / dr
        dnz = np.zeros_like(alpha)
        dnz[:, 1:-1] = (nz_c[:, 2:] - nz_c[:, :-2]) / (2 * dz)
        dnz[:, 0] = (nz_c[:, 1] - nz_c[:, 0]) / dz
        dnz[:, -1] = (nz_c[:, -1] - nz_c[:, -2]) / dz
        kappa = d_rnr / (r + 1e-14) + dnz
        # limit extreme curvature (CSF stability on coarse grids)
        kappa = np.clip(kappa, -1.0 / min(dr, dz), 1.0 / min(dr, dz))
        return kappa

    # -------------------------------------------------------- VOF advection
    def _face_alpha_donor(self, alpha: np.ndarray, vel_face: np.ndarray, axis: str) -> np.ndarray:
        """Upwind volume fraction on faces."""
        if axis == "r":
            # vel_face: (nr+1, nz)
            af = np.zeros_like(vel_face)
            af[1:-1, :] = np.where(vel_face[1:-1, :] >= 0.0, alpha[:-1, :], alpha[1:, :])
            af[0, :] = alpha[0, :]
            af[-1, :] = alpha[-1, :]
            return af
        # axis == "z", vel_face (nr, nz+1)
        af = np.zeros_like(vel_face)
        af[:, 1:-1] = np.where(vel_face[:, 1:-1] >= 0.0, alpha[:, :-1], alpha[:, 1:])
        af[:, 0] = alpha[:, 0]
        af[:, -1] = alpha[:, -1]
        return af

    def _advect_vof(self, dt: float) -> None:
        """Directional-split Weymouth–Yue style VOF (upwind donor-acceptor)."""
        p = self.proc
        alpha = self.alpha.copy()
        u, w = self.u, self.w
        dr, dz = self.dr, self.dz
        nr, nz = p.nr, p.nz

        # r-sweep
        af = self._face_alpha_donor(alpha, u, "r")
        # flux through radial face: u * α * (2π r Δz) / (2π r_c Δr Δz) = u α r_face / (r_c Δr)
        flux_r = np.zeros((nr + 1, nz))
        for i in range(nr + 1):
            r_f = self.ru[i]
            if i == 0:
                flux_r[i, :] = 0.0
            else:
                # volume flux / cell volume for adjacent cells differs; use conservative form
                flux_r[i, :] = u[i, :] * af[i, :] * r_f  # will divide by r_c dr
        dadt = np.zeros_like(alpha)
        for i in range(nr):
            r_c = self.rc[i]
            dadt[i, :] = -(flux_r[i + 1, :] - flux_r[i, :]) / (r_c * dr)
        alpha1 = np.clip(alpha + dt * dadt, 0.0, 1.0)

        # z-sweep
        afz = self._face_alpha_donor(alpha1, w, "z")
        flux_z = w * afz  # / dz later
        dadt_z = -(flux_z[:, 1:] - flux_z[:, :-1]) / dz
        alpha2 = np.clip(alpha1 + dt * dadt_z, 0.0, 1.0)

        # wire feed inlet: force alpha=1 in wire inlet cells
        top_wire = (self.rc <= p.R_wire)
        alpha2[top_wire, -1] = 1.0
        if nz >= 2:
            alpha2[top_wire, -2] = np.maximum(alpha2[top_wire, -2], 0.99)

        self.alpha = alpha2

    def _apply_wire_feed(self, dt: float) -> None:
        """Impose downward feed velocity in solid wire; accumulate fed volume."""
        p = self.proc
        m = self.mat
        solid_wire = self.wire_mask & (self.T < m.T_solidus + 5.0) & (self.alpha >= p.alpha_metal)
        # also always feed at top inlet
        inlet = (self.R <= p.R_wire) & (self.Z >= p.Lz - 2.5 * self.dz)
        feed_region = solid_wire | inlet
        # set w on faces inside feed column
        for i in range(p.nr):
            if self.rc[i] > p.R_wire:
                continue
            for j in range(p.nz + 1):
                j_lo = max(j - 1, 0)
                j_hi = min(j, p.nz - 1)
                if feed_region[i, j_lo] or feed_region[i, j_hi] or self.zw[j] >= p.z_wire_tip:
                    self.w[i, j] = -p.v_feed
            for j in range(p.nz):
                if feed_region[i, j]:
                    self.u[i, j] = 0.0
                    self.u[i + 1, j] = 0.0

        dV = np.pi * p.R_wire**2 * p.v_feed * dt
        self._V_feed_cum += dV

    # -------------------------------------------------------- momentum
    def _cell_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        u_c = 0.5 * (self.u[:-1, :] + self.u[1:, :])
        w_c = 0.5 * (self.w[:, :-1] + self.w[:, 1:])
        return u_c, w_c

    def _surface_forces(self) -> Tuple[np.ndarray, np.ndarray]:
        """CSF + Marangoni + arc pressure on interface."""
        p = self.proc
        m = self.mat
        alpha = self.alpha
        ar, az = self._youngs_normal(alpha)
        kappa = self._curvature(alpha)
        # CSF: σ κ ∇α (scaled for coarse-grid stability)
        Fr = m.sigma * kappa * ar
        Fz = m.sigma * kappa * az
        Fcap = 2.0 * p.F_em_clip
        Fr = np.clip(Fr, -Fcap, Fcap)
        Fz = np.clip(Fz, -Fcap, Fcap)

        # Marangoni: (dσ/dT)(∇_s T)|∇α| approximately as (dσ/dT) ∇T projected
        fL = self.fL
        Tr, Tz = self._youngs_normal(self.T)  # reuse gradient helper
        # tangential ≈ ∇T - (∇T·n) n
        mag = np.sqrt(ar**2 + az**2) + 1e-14
        nr_c, nz_c = ar / mag, az / mag
        dT_n = Tr * nr_c + Tz * nz_c
        Tr_t = Tr - dT_n * nr_c
        Tz_t = Tz - dT_n * nz_c
        mar = m.dsigma_dT * fL * alpha
        Fr += mar * Tr_t * mag
        Fz += mar * Tz_t * mag

        # Arc pressure on pool free surface: p_arc n |∇α|
        if p.p_arc_max > 0.0:
            p_arc = p.p_arc_max * np.exp(-3.0 * (self.R / p.r_b) ** 2)
            # only near plate free surface
            near_pool = self.pool_mask | (
                (self.Z <= p.H_plate + 1.5e-3) & (self.Z >= p.H_plate - 2e-3)
            )
            Fr += p_arc * ar * near_pool
            Fz += p_arc * az * near_pool

        return Fr, Fz

    def _step_momentum(self, dt: float) -> None:
        p = self.proc
        m = self.mat
        if not p.enable_flow:
            return
        nr, nz = p.nr, p.nz
        dr, dz = self.dr, self.dz
        alpha = self.alpha
        rho = self._rho(alpha)
        mu = self._mu(alpha)
        fL = self.fL
        A_darcy = self._darcy_coeff(fL, alpha)
        Fr_s, Fz_s = self._surface_forces()

        u = self.u
        w = self.w
        u_star = u.copy()
        w_star = w.copy()

        # --- u_r faces (i=1..nr-1 interior; i=0 axis = 0; i=nr wall = 0)
        for i in range(1, nr):
            r_f = self.ru[i]
            # stagger density / viscosity / forces
            rho_f = 0.5 * (rho[i - 1, :] + rho[i, :])
            mu_f = 0.5 * (mu[i - 1, :] + mu[i, :])
            A_f = 0.5 * (A_darcy[i - 1, :] + A_darcy[i, :])
            Fr_f = 0.5 * (Fr_s[i - 1, :] + Fr_s[i, :]) + 0.5 * (self.Fr[i - 1, :] + self.Fr[i, :])

            ui = u[i, :]
            # neighbours
            ue = u[i + 1, :] if i + 1 <= nr else 0.0
            uw = u[i - 1, :]
            # upwind advection (axisymmetric)
            u_c = ui
            # approximate cell-centre velocities at face
            w_adv = 0.5 * (
                0.5 * (w[i - 1, :-1] + w[i - 1, 1:]) + 0.5 * (w[i, :-1] + w[i, 1:])
            )
            # radial upwind
            dudr = np.where(u_c >= 0.0, (ui - uw) / dr, (ue - ui) / dr)
            # axial
            ut = np.empty(nz)
            ub = np.empty(nz)
            ut[:-1] = ui[1:]
            ut[-1] = ui[-1]
            ub[1:] = ui[:-1]
            ub[0] = ui[0]
            dudz = np.where(w_adv >= 0.0, (ui - ub) / dz, (ut - ui) / dz)
            adv = u_c * dudr + w_adv * dudz

            # viscosity laplacian (simplified)
            d2r = (ue - 2.0 * ui + uw) / dr**2
            d2z = (ut - 2.0 * ui + ub) / dz**2
            # geometric term -u/r^2 for axisymmetric
            visc = (mu_f / rho_f) * (d2r + d2z - ui / (r_f**2 + 1e-16))

            body = Fr_f / rho_f
            u_star[i, :] = (ui + dt * (-adv + visc + body)) / (1.0 + dt * A_f / rho_f)

        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0

        # --- u_z faces
        for j in range(1, nz):
            rho_f = 0.5 * (rho[:, j - 1] + rho[:, j])
            mu_f = 0.5 * (mu[:, j - 1] + mu[:, j])
            A_f = 0.5 * (A_darcy[:, j - 1] + A_darcy[:, j])
            Fz_f = 0.5 * (Fz_s[:, j - 1] + Fz_s[:, j]) + 0.5 * (self.Fz[:, j - 1] + self.Fz[:, j])

            wi = w[:, j]
            wt = w[:, j + 1] if j + 1 <= nz else w[:, j]
            wb = w[:, j - 1]
            u_adv = 0.5 * (
                0.5 * (u[:-1, j - 1] + u[1:, j - 1]) + 0.5 * (u[:-1, j] + u[1:, j])
            )
            dwdz = np.where(wi >= 0.0, (wi - wb) / dz, (wt - wi) / dz)
            we = np.empty(nr)
            ww = np.empty(nr)
            we[:-1] = wi[1:]
            we[-1] = wi[-1]
            ww[1:] = wi[:-1]
            ww[0] = wi[0]
            dwdr = np.where(u_adv >= 0.0, (wi - ww) / dr, (we - wi) / dr)
            adv = u_adv * dwdr + wi * dwdz

            d2z = (wt - 2.0 * wi + wb) / dz**2
            d2r = (we - 2.0 * wi + ww) / dr**2
            # axisymmetric: (1/r)∂/∂r(r ∂w/∂r) ≈ d2r + (1/r) dw/dr
            inv_r = 1.0 / (self.rc + 1e-16)
            visc = (mu_f / rho_f) * (d2r + inv_r * (we - ww) / (2 * dr) + d2z)

            grav = -p.g  # z upward
            buoy = p.g * m.beta * (0.5 * (self.T[:, j - 1] + self.T[:, j]) - m.T_ref)
            # buoyancy only in liquid metal
            fL_f = 0.5 * (fL[:, j - 1] + fL[:, j])
            a_f = 0.5 * (alpha[:, j - 1] + alpha[:, j])
            body = Fz_f / rho_f + grav + buoy * fL_f * a_f
            w_star[:, j] = (wi + dt * (-adv + visc + body)) / (1.0 + dt * A_f / rho_f)

        w_star[:, 0] = 0.0
        w_star[:, -1] = 0.0
        # outlet / top BC: zero gradient for gas, feed handled later
        w_star[:, -1] = w_star[:, -2]

        self._project(u_star, w_star, dt)
        # Magnitude clip so |u|_max in the log cannot exceed u_clip
        # (component-wise ±u_clip allowed √2·u_clip on the diagonal).
        uc = p.u_clip
        u_c = 0.5 * (u_star[:-1, :] + u_star[1:, :])
        w_c = 0.5 * (w_star[:, :-1] + w_star[:, 1:])
        spd = np.sqrt(u_c**2 + w_c**2)
        fac = np.minimum(1.0, uc / (spd + 1e-30))
        u_star[1:-1, :] *= 0.5 * (fac[:-1, :] + fac[1:, :])
        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0
        w_star[:, 1:-1] *= 0.5 * (fac[:, :-1] + fac[:, 1:])
        w_star[:, 0] = 0.0
        self.u = u_star
        self.w = w_star

    def _project(self, u_star: np.ndarray, w_star: np.ndarray, dt: float) -> None:
        """Variable-density pressure projection, axisymmetric."""
        p = self.proc
        nr, nz = p.nr, p.nz
        dr, dz = self.dr, self.dz
        rho = self._rho()
        N = nr * nz

        def idx(i, j):
            return i * nz + j

        # divergence of u* (axisymmetric)
        div = np.zeros((nr, nz))
        for i in range(nr):
            r_c = self.rc[i]
            r_e = self.ru[i + 1]
            r_w = self.ru[i]
            div[i, :] = (r_e * u_star[i + 1, :] - r_w * u_star[i, :]) / (r_c * dr) + (
                w_star[i, 1:] - w_star[i, :-1]
            ) / dz

        # harmonic density on faces
        def rho_r(i, j):
            if i <= 0:
                return rho[0, j]
            if i >= nr:
                return rho[nr - 1, j]
            return 2.0 / (1.0 / rho[i - 1, j] + 1.0 / rho[i, j])

        def rho_z(i, j):
            if j <= 0:
                return rho[i, 0]
            if j >= nz:
                return rho[i, nz - 1]
            return 2.0 / (1.0 / rho[i, j - 1] + 1.0 / rho[i, j])

        rows, cols, data = [], [], []
        rhs = (div / dt).ravel(order="C")

        for i in range(nr):
            for j in range(nz):
                r = idx(i, j)
                if i == 0 and j == 0:
                    rows.append(r)
                    cols.append(r)
                    data.append(1.0)
                    rhs[r] = 0.0
                    continue
                diag = 0.0
                r_c = self.rc[i]
                # east
                if i + 1 < nr:
                    rf = self.ru[i + 1]
                    coef = (rf / (r_c * dr)) * (1.0 / rho_r(i + 1, j)) / dr
                    rows.append(r)
                    cols.append(idx(i + 1, j))
                    data.append(coef)
                    diag -= coef
                # west
                if i - 1 >= 0:
                    rf = self.ru[i]
                    coef = (rf / (r_c * dr)) * (1.0 / rho_r(i, j)) / dr
                    rows.append(r)
                    cols.append(idx(i - 1, j))
                    data.append(coef)
                    diag -= coef
                # north
                if j + 1 < nz:
                    coef = (1.0 / rho_z(i, j + 1)) / dz**2
                    rows.append(r)
                    cols.append(idx(i, j + 1))
                    data.append(coef)
                    diag -= coef
                # south
                if j - 1 >= 0:
                    coef = (1.0 / rho_z(i, j)) / dz**2
                    rows.append(r)
                    cols.append(idx(i, j - 1))
                    data.append(coef)
                    diag -= coef
                rows.append(r)
                cols.append(r)
                data.append(diag if diag != 0.0 else 1.0)

        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        try:
            phi = spsolve(A, rhs).reshape((nr, nz), order="C")
        except Exception:
            return
        if phi is None or np.any(~np.isfinite(phi)):
            return
        self.p = phi

        # correct velocities
        for i in range(1, nr):
            rho_f = np.array([rho_r(i, j) for j in range(nz)])
            u_star[i, :] -= (dt / rho_f) * (phi[i, :] - phi[i - 1, :]) / dr
        for j in range(1, nz):
            rho_f = np.array([rho_z(i, j) for i in range(nr)])
            w_star[:, j] -= (dt / rho_f) * (phi[:, j] - phi[:, j - 1]) / dz

        u_star[0, :] = 0.0
        u_star[-1, :] = 0.0
        w_star[:, 0] = 0.0
        # top: allow outflow
        w_star[self.rc > self.proc.R_wire, -1] = w_star[self.rc > self.proc.R_wire, -2]

    # ----------------------------------------------------------- energy
    def _heat_sources(self) -> np.ndarray:
        """Volumetric heating: arc on pool interface, anode on drop/wire tip, Joule."""
        p = self.proc
        m = self.mat
        ar, az = self._youngs_normal(self.alpha)
        mag = np.sqrt(ar**2 + az**2)
        q = np.zeros_like(self.alpha)

        # Gaussian arc heat on pool free surface
        q_arc = p.P_arc * (3.0 / (np.pi * p.r_b**2)) * np.exp(-3.0 * (self.R / p.r_b) ** 2)
        near_pool_iface = mag > 0.1 / min(self.dr, self.dz)
        near_pool_iface &= (self.Z <= p.H_plate + 2e-3) & (self.Z >= p.H_plate - 2.5e-3)
        # distribute as q * |∇α|
        q += q_arc * mag * near_pool_iface

        # Anode heating on wire/drop surface: I * U_anode spread on interface area
        tip_iface = mag > 0.1 / min(self.dr, self.dz)
        tip_iface &= self.wire_mask | (
            (self.Z > p.H_plate + 0.5e-3) & (self.R <= 3 * p.R_wire)
        )
        if np.any(tip_iface):
            area_w = float(np.sum(mag[tip_iface] * self.cell_area[tip_iface]))
            if area_w > 0:
                q_anode = p.I_arc * p.U_anode  # Watts
                q += (q_anode / area_w) * mag * tip_iface

        # Joule heating
        se = self._sigma_e()
        J2 = self.Jr**2 + self.Jz**2
        q += J2 / (se + 1e-30)

        # losses on interface
        T = self.T
        q_loss = p.h_conv * (T - p.Ta) + p.emissivity * SIGMA_SB * (T**4 - p.Ta**4)
        q -= q_loss * mag * (self.alpha > 0.05) * (self.alpha < 0.95)

        return q

    def _step_energy(self, dt: float) -> None:
        m = self.mat
        p = self.proc
        nr, nz = p.nr, p.nz
        dr, dz = self.dr, self.dz
        T = self.T
        alpha = self.alpha
        fL = self.liquid_fraction(T)
        k = self._k(alpha, fL)
        rho = self._rho(alpha)
        cp_ref = 0.5 * (m.cp_s + m.cp_l)
        cp = (cp_ref * alpha + m.cp_g * (1.0 - alpha))
        u_c, w_c = self._cell_velocity()

        # conduction fluxes
        Fr = np.zeros((nr + 1, nz))
        Fr[1:-1, :] = 0.5 * (k[:-1, :] + k[1:, :]) * (T[1:, :] - T[:-1, :]) / dr
        Fz = np.zeros((nr, nz + 1))
        Fz[:, 1:-1] = 0.5 * (k[:, :-1] + k[:, 1:]) * (T[:, 1:] - T[:, :-1]) / dz
        # bottom loss
        Fz[:, 0] = p.h_bottom * (T[:, 0] - p.Ta)
        Fz[:, -1] = 0.0

        div_cond = np.zeros_like(T)
        for i in range(nr):
            r_c = self.rc[i]
            r_e = self.ru[i + 1]
            r_w = self.ru[i]
            div_cond[i, :] = (r_e * Fr[i + 1, :] - r_w * Fr[i, :]) / (r_c * dr) + (
                Fz[i, 1:] - Fz[i, :-1]
            ) / dz

        # upwind advection
        Te = np.empty_like(T)
        Tw = np.empty_like(T)
        Tt = np.empty_like(T)
        Tb = np.empty_like(T)
        Te[:-1, :] = T[1:, :]
        Te[-1, :] = T[-1, :]
        Tw[1:, :] = T[:-1, :]
        Tw[0, :] = T[0, :]
        Tt[:, :-1] = T[:, 1:]
        Tt[:, -1] = T[:, -1]
        Tb[:, 1:] = T[:, :-1]
        Tb[:, 0] = T[:, 0]
        dTdr = np.where(u_c >= 0.0, (T - Tw) / dr, (Te - T) / dr)
        dTdz = np.where(w_c >= 0.0, (T - Tb) / dz, (Tt - T) / dz)
        adv = u_c * dTdr + w_c * dTdz

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

        # wire inlet temperature
        top_wire = self.rc <= p.R_wire
        T_star[top_wire, -1] = p.T_wire

        self.T = np.clip(T_star, 200.0, 3500.0)
        self.fL = self.liquid_fraction(self.T)

    # -------------------------------------------------------------- time
    def suggest_dt(self) -> float:
        p = self.proc
        m = self.mat
        u_c, w_c = self._cell_velocity()
        umax = float(np.max(np.abs(u_c))) + 1e-9
        wmax = float(np.max(np.abs(w_c))) + 1e-9
        dt_cfl = p.cfl / (umax / self.dr + wmax / self.dz)
        # capillary
        dx = min(self.dr, self.dz)
        dt_cap = p.cfl * np.sqrt(m.rho * dx**3 / (np.pi * m.sigma + 1e-30))
        # viscous
        dt_visc = 0.2 * m.rho * dx**2 / (m.mu + 1e-30)
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
        u_c, w_c = self._cell_velocity()
        speed = np.sqrt(u_c**2 + w_c**2)
        V_m = self.metal_volume()
        V_w = float(np.sum(self.alpha * self.wire_mask * self.cell_area))
        V_p = float(np.sum(self.alpha * self.pool_mask * self.cell_area))
        self.diag.time.append(self.t)
        self.diag.metal_volume.append(V_m)
        self.diag.wire_volume.append(V_w)
        self.diag.pool_volume.append(V_p)
        self.diag.n_components.append(getattr(self, "_n_components", 1))
        self.diag.neck_radius.append(self.neck_radius())
        self.diag.pool_depression.append(self.pool_depression())
        self.diag.T_max.append(float(self.T.max()))
        self.diag.max_speed.append(float(np.max(speed)))
        self.diag.detached.append(self._detached)
        if record_fields:
            self.diag.alpha_frames.append(self.alpha.copy())
            self.diag.T_frames.append(self.T.copy())
            self.diag.speed_frames.append(speed.copy())
            self.diag.Fr_frames.append(self.Fr.copy())
            self.diag.time_frames.append(self.t)

    def volume_balance_ratio(self) -> float:
        """|ΔV_metal - V_fed| / max(V_fed, V0)."""
        dV = self.metal_volume() - self._V_metal0
        den = max(self._V_feed_cum, self._V_metal0, 1e-12)
        return float(abs(dV - self._V_feed_cum) / den)

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
            next_rec = self.t
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
                        f"t={1e3*self.t:.2f}ms  Tmax={self.T.max():.0f}K  "
                        f"Rneck={1e3*self.neck_radius():.3f}mm  "
                        f"comp={getattr(self, '_n_components', 1)}  "
                        f"|u|max={self.diag.max_speed[-1]:.3f}m/s  "
                        f"det={self._detached}  "
                        f"step={1e3*dt_wall:.1f}ms  "
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
        Lr=0.006,
        Lz=0.010,
        nr=28,
        nz=48,
        H_plate=0.003,
        R_wire=0.0006,
        z_wire_tip=0.0072,
        R_drop0=0.00115,
        v_feed=0.04,
        I_arc=170.0,
        P_arc=1600.0,
        r_b=0.0025,
        p_arc_max=250.0,
        t_end=0.010,
        dt_max=4.0e-5,
        cfl=0.2,
        em_every=4,
        u_clip=4.0,
    )
    base.update(overrides)
    return Process(**base)
