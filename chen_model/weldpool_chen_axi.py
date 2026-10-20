# Axisymmetric LTE arc + weld pool (Chen/Murphy style, I=95 A BOP).
# Domain (r,z): plate metal, argon plasma, wire tip. Free surface Kim–Na φ(r).
# Not full 3D ArcWeld; no Al vapour; droplet influence time-averaged.
# Refs: Chen et al. AMM 81 (2020); Murphy JPhysD 44 (2011) §2.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import argon_lte as ar

MU0 = 4.0e-7 * np.pi
SIGMA_SB = 5.670374419e-8


@dataclass
class MetalProps:
    name: str = "AA5754"
    T_solidus: float = 880.0
    T_liquidus: float = 920.0
    rho: float = 2700.0
    cp_s: float = 900.0
    cp_l: float = 1170.0
    k_s: float = 150.0
    k_l: float = 95.0
    mu: float = 1.2e-3
    latent: float = 3.9e5
    beta: float = 1.0e-4
    T_ref: float = 920.0
    mush_C: float = 1.0e7
    mush_B: float = 1.0e-3
    sigma_e_l: float = 3.5e6
    sigma_e_s: float = 1.5e7
    sigma0: float = 0.90  # N/m surface tension
    dsigma_dT: float = -3.5e-4
    work_fn: float = 4.0  # V (eV-scale, used as volts in sheath heat)
    cathode_fall: float = 15.0  # V
    anode_work: float = 4.0  # V
    emissivity: float = 0.4


@dataclass
class ProcessAxi:
    # Geometry
    Lr: float = 0.008
    H_plate: float = 0.003
    Lz_gas: float = 0.014  # plasma height above z=0 (CTWD ~12 mm + tip)
    nr: int = 48
    nz: int = 72
    R_wire: float = 0.0006
    z_wire_tip: float = 0.012  # tip above plate (CTWD ~12 mm)
    # Welding (Table 1 BOP)
    I_arc: float = 95.0
    v_feed: float = 4.3 / 60.0
    v_weld: float = 0.9 / 60.0
    f_drop: float = 93.0
    v_drop: float = 0.6
    T_drop: float = 1200.0
    T0: float = 300.0
    Ta: float = 300.0
    h_bottom: float = 50.0
    h_side: float = 20.0
    g: float = 9.81
    # Numerics
    n_outer: int = 80
    n_energy: int = 40
    n_flow: int = 30
    relax_T: float = 0.35
    relax_phi_surf: float = 0.25
    T_plasma_max: float = 18000.0
    T_metal_max: float = 1800.0
    surface_every: int = 5
    enable_flow: bool = True
    enable_plasma_flow: bool = True
    F_em_clip: float = 5.0e6
    u_clip_plasma: float = 200.0
    u_clip_metal: float = 1.5
    # Characteristic length for axi bead volume = A * L_char
    L_char: float = 0.006
    p_drop_scale: float = 1.0  # scales droplet pressure

    @property
    def bead_area(self) -> float:
        return np.pi * self.R_wire**2 * self.v_feed / self.v_weld

    @property
    def V_bead(self) -> float:
        return self.bead_area * self.L_char

    @property
    def m_dot(self) -> float:
        return 2700.0 * np.pi * self.R_wire**2 * self.v_feed


@dataclass
class Diagnostics:
    iter: list = field(default_factory=list)
    T_arc_max: list = field(default_factory=list)
    T_pool_max: list = field(default_factory=list)
    RH: list = field(default_factory=list)
    WD: list = field(default_factory=list)
    BW: list = field(default_factory=list)
    residual_T: list = field(default_factory=list)
    voltage: list = field(default_factory=list)


class WeldPoolChenAxi:
    """Quasi-steady axisymmetric LTE arc + pool + Kim–Na surface."""

    def __init__(
        self,
        metal: Optional[MetalProps] = None,
        process: Optional[ProcessAxi] = None,
    ):
        self.mat = metal or MetalProps()
        self.proc = process or ProcessAxi()
        self._build_grid()
        self.reset()

    def _build_grid(self) -> None:
        p = self.proc
        self.z_min = -p.H_plate
        self.z_max = p.Lz_gas
        Lz = p.H_plate + p.Lz_gas
        self.dr = p.Lr / p.nr
        self.dz = Lz / p.nz
        self.rc = (np.arange(p.nr) + 0.5) * self.dr
        self.zc = self.z_min + (np.arange(p.nz) + 0.5) * self.dz
        self.ru = np.arange(p.nr + 1) * self.dr
        self.zw = self.z_min + np.arange(p.nz + 1) * self.dz
        self.R, self.Z = np.meshgrid(self.rc, self.zc, indexing="ij")
        self.cell_vol = 2.0 * np.pi * self.R * self.dr * self.dz  # m³

    def reset(self) -> None:
        p = self.proc
        m = self.mat
        nr, nz = p.nr, p.nz
        self.T = np.full((nr, nz), p.T0, dtype=np.float64)
        self.u = np.zeros((nr + 1, nz), dtype=np.float64)  # u_r
        self.w = np.zeros((nr, nz + 1), dtype=np.float64)  # u_z
        self.pot = np.zeros((nr, nz), dtype=np.float64)
        self.Jr = np.zeros((nr, nz), dtype=np.float64)
        self.Jz = np.zeros((nr, nz), dtype=np.float64)
        self.Bth = np.zeros((nr, nz), dtype=np.float64)
        self.Fr = np.zeros((nr, nz), dtype=np.float64)
        self.Fz = np.zeros((nr, nz), dtype=np.float64)
        self.phi_surf = np.zeros(nr, dtype=np.float64)  # height φ(r)
        self.fL = np.zeros((nr, nz), dtype=np.float64)
        self.diag = Diagnostics()
        self._iter = 0

        # Initial masks: plate metal z<=0; wire; plasma elsewhere above
        self.wire = (self.R <= p.R_wire) & (self.Z >= p.z_wire_tip)
        self.metal = (self.Z <= 0.0) | self.wire
        self.plasma = ~self.metal

        # Seed hot plasma column and warm surface
        col = self.plasma & (self.R < 2.5e-3)
        self.T[col] = 8000.0 + 6000.0 * np.exp(-((self.R[col] / 1.5e-3) ** 2))
        tip = self.wire & (self.Z < p.z_wire_tip + 1.5e-3)
        self.T[tip] = 2000.0
        seed = (self.Z <= 0.0) & (self.Z >= -0.5e-3) & (self.R <= 2.5e-3)
        self.T[seed] = m.T_liquidus + 50.0
        self.fL = self.liquid_fraction(self.T)
        self._rebuild_masks()

    def liquid_fraction(self, T: np.ndarray) -> np.ndarray:
        m = self.mat
        dT = max(m.T_liquidus - m.T_solidus, 1.0)
        fl = np.clip((T - m.T_solidus) / dT, 0.0, 1.0)
        fl = np.where(self.metal & ~self.wire, fl, 0.0)
        # wire tip treated as hot metal (liquid-capable for EM)
        fl = np.where(self.wire & (T > m.T_solidus), np.maximum(fl, 0.5), fl)
        return fl

    def _rebuild_masks(self) -> None:
        """Metal = plate below φ(r) or original plate + wire; plasma = rest above φ."""
        p = self.proc
        phi = self.phi_surf
        # Interpolate φ(r) onto cells
        phi_c = np.interp(self.rc, self.rc, phi)
        Phi = phi_c[:, None] * np.ones((1, p.nz))
        plate_body = self.Z <= Phi
        # Keep solid plate below z=0 always metal; reinforcement 0<z<φ also metal
        self.wire = (self.R <= p.R_wire) & (self.Z >= p.z_wire_tip)
        self.metal = plate_body | self.wire
        # Plasma only above surface and outside wire
        self.plasma = (~self.metal) & (self.Z > Phi - 0.5 * self.dz)
        # Ensure no plasma inside plate bottom
        self.plasma &= self.Z > -p.H_plate + 0.5 * self.dz
        self.fL = self.liquid_fraction(self.T)

    # ------------------------------------------------------------------ props
    def _sigma_e(self) -> np.ndarray:
        m = self.mat
        p = self.proc
        se = np.zeros_like(self.T)
        se[self.plasma] = ar.sigma(self.T[self.plasma])
        # LTE-diffusion style: keep a warm conductive channel so the arc can attach
        channel = self.plasma & (self.R < 3.5e-3)
        se_floor = 800.0 * np.exp(-((self.R / 1.8e-3) ** 2))
        se[channel] = np.maximum(se[channel], se_floor[channel])
        fl = self.fL
        se_m = m.sigma_e_l * fl + m.sigma_e_s * (1.0 - fl)
        se[self.metal] = se_m[self.metal]
        return np.maximum(se, 1.0e-6)

    def _k(self) -> np.ndarray:
        m = self.mat
        k = np.zeros_like(self.T)
        k[self.plasma] = ar.k_th(self.T[self.plasma])
        fl = self.fL
        k_m = m.k_s * (1.0 - fl) + m.k_l * fl
        k[self.metal] = k_m[self.metal]
        return np.maximum(k, 1.0e-3)

    def _cp(self) -> np.ndarray:
        m = self.mat
        cp = np.zeros_like(self.T)
        cp[self.plasma] = ar.cp(self.T[self.plasma])
        fl = self.fL
        cp_m = m.cp_s * (1.0 - fl) + m.cp_l * fl
        cp[self.metal] = cp_m[self.metal]
        return np.maximum(cp, 100.0)

    def _rho(self) -> np.ndarray:
        m = self.mat
        rho = np.zeros_like(self.T)
        rho[self.plasma] = ar.rho(self.T[self.plasma])
        rho[self.metal] = m.rho
        return np.maximum(rho, 1.0e-4)

    def _mu(self) -> np.ndarray:
        m = self.mat
        mu = np.zeros_like(self.T)
        mu[self.plasma] = ar.mu(self.T[self.plasma])
        mu[self.metal] = m.mu
        return np.maximum(mu, 1.0e-6)

    # ------------------------------------------------------------------ EM
    def solve_em(self) -> float:
        """Electric potential; return approximate arc voltage."""
        p = self.proc
        m = self.mat
        nr, nz = p.nr, p.nz
        se = self._sigma_e()
        N = nr * nz
        rows, cols, data = [], [], []
        rhs = np.zeros(N)

        def idx(i, j):
            return i * nz + j

        # Dirichlet: bottom of plate φ=0; top of wire fixed current via Neumann
        bottom = self.Z < self.z_min + 1.5 * self.dz
        wire_top = self.wire & (self.Z > p.z_wire_tip + 0.5 * (self.z_max - p.z_wire_tip))

        for i in range(nr):
            for j in range(nz):
                r = idx(i, j)
                if bottom[i, j] and self.metal[i, j]:
                    rows.append(r)
                    cols.append(r)
                    data.append(1.0)
                    rhs[r] = 0.0
                    continue

                diag = 0.0
                # r-neighbors
                for ni in (i - 1, i + 1):
                    if ni < 0 or ni >= nr:
                        continue
                    se_f = 0.5 * (se[i, j] + se[ni, j])
                    face_r = 0.5 * (self.rc[i] + self.rc[ni])
                    area = 2.0 * np.pi * face_r * self.dz
                    c = se_f * area / self.dr
                    rows.append(r)
                    cols.append(idx(ni, j))
                    data.append(c)
                    diag -= c
                # z-neighbors
                for nj in (j - 1, j + 1):
                    if nj < 0 or nj >= nz:
                        continue
                    se_f = 0.5 * (se[i, j] + se[i, nj])
                    area = 2.0 * np.pi * self.rc[i] * self.dr
                    c = se_f * area / self.dz
                    rows.append(r)
                    cols.append(idx(i, nj))
                    data.append(c)
                    diag -= c

                # Current injection at wire top: -σ ∂φ/∂z = j_in
                if wire_top[i, j] and j == nz - 1:
                    area = 2.0 * np.pi * self.rc[i] * self.dr
                    # distribute I over wire cross-section cells at top
                    j_in = p.I_arc / (np.pi * p.R_wire**2)
                    rhs[r] = -j_in * area
                elif abs(diag) < 1e-30:
                    diag = -1.0e-12

                rows.append(r)
                cols.append(r)
                data.append(diag if abs(diag) > 1e-30 else 1.0)

        A = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                pot = spsolve(A, rhs).reshape((nr, nz))
            except Exception:
                return 0.0
        if pot is None or np.any(~np.isfinite(pot)):
            return 0.0
        # Normalize: shift so mean bottom = 0 already; clip runaway
        pot = pot - np.mean(pot[bottom & self.metal]) if np.any(bottom & self.metal) else pot
        self.pot = np.clip(pot, -200.0, 200.0)

        Jr = np.zeros((nr, nz))
        Jz = np.zeros((nr, nz))
        for i in range(nr):
            for j in range(nz):
                if i == 0:
                    dpr = (self.pot[min(1, nr - 1), j] - self.pot[0, j]) / self.dr
                elif i == nr - 1:
                    dpr = (self.pot[i, j] - self.pot[i - 1, j]) / self.dr
                else:
                    dpr = (self.pot[i + 1, j] - self.pot[i - 1, j]) / (2 * self.dr)
                if j == 0:
                    dpz = (self.pot[i, min(1, nz - 1)] - self.pot[i, 0]) / self.dz
                elif j == nz - 1:
                    dpz = (self.pot[i, j] - self.pot[i, j - 1]) / self.dz
                else:
                    dpz = (self.pot[i, j + 1] - self.pot[i, j - 1]) / (2 * self.dz)
                Jr[i, j] = -se[i, j] * dpr
                Jz[i, j] = -se[i, j] * dpz
        self.Jr, self.Jz = Jr, Jz

        # Ampere: B_θ from enclosed axial current
        I_enc = np.cumsum(Jz * 2.0 * np.pi * self.rc[:, None] * self.dr, axis=0)
        self.Bth = MU0 * I_enc / (2.0 * np.pi * (self.rc[:, None] + 1e-16))
        Fr = np.clip(-Jz * self.Bth, -p.F_em_clip, p.F_em_clip)
        Fz = np.clip(Jr * self.Bth, -p.F_em_clip, p.F_em_clip)
        self.Fr, self.Fz = Fr, Fz

        # Voltage estimate: pot at wire tip minus ground
        tip = self.wire & (np.abs(self.Z - p.z_wire_tip) < 2 * self.dz)
        V = float(np.mean(self.pot[tip])) if np.any(tip) else float(np.max(self.pot))
        return abs(V)

    # -------------------------------------------------------------- energy
    def _sheath_heat(self) -> np.ndarray:
        """Volumetric sheath heating near plasma–metal interfaces."""
        p = self.proc
        m = self.mat
        Q = np.zeros_like(self.T)
        jmag = np.sqrt(self.Jr**2 + self.Jz**2)
        # Workpiece cathode cells adjacent to plasma
        metal_surf = self.metal & ~self.wire
        # neighbor plasma?
        plas_nb = np.zeros_like(metal_surf, dtype=bool)
        plas_nb[1:, :] |= self.plasma[:-1, :]
        plas_nb[:-1, :] |= self.plasma[1:, :]
        plas_nb[:, 1:] |= self.plasma[:, :-1]
        plas_nb[:, :-1] |= self.plasma[:, 1:]
        cat = metal_surf & plas_nb & (self.R < 4.0e-3)
        # Sw = j (Vw - φ_w) / thickness ~ j * V / dz
        Q[cat] += jmag[cat] * (m.cathode_fall - m.work_fn) / self.dz
        # Wire anode
        wire_surf = self.wire & plas_nb
        Q[wire_surf] += jmag[wire_surf] * m.anode_work / self.dz
        # Radiation sink on metal surface
        Q[cat] -= m.emissivity * SIGMA_SB * (self.T[cat] ** 4 - p.Ta**4) / self.dz
        return Q

    def _drop_heat(self) -> np.ndarray:
        """Averaged droplet enthalpy deposited under the arc in the pool."""
        p = self.proc
        m = self.mat
        Q = np.zeros_like(self.T)
        r_dep = 1.2e-3
        dep = self.metal & ~self.wire & (self.R <= r_dep) & (self.Z <= self.phi_surf.max() + 1e-3) & (
            self.Z >= -1.0e-3
        )
        if not np.any(dep):
            return Q
        V = float(np.sum(self.cell_vol[dep]))
        if V < 1e-16:
            return Q
        # enthalpy rate of drops relative to local T
        h_drop = m.cp_l * (p.T_drop - m.T_ref) + m.latent
        Q[dep] = p.m_dot * h_drop / V
        return Q

    def relax_energy(self, n_iter: Optional[int] = None) -> float:
        """Under-relaxed diffusion + Joule − radiation + sheath."""
        p = self.proc
        m = self.mat
        if n_iter is None:
            n_iter = p.n_energy
        nr, nz = p.nr, p.nz
        res = 0.0
        for _ in range(n_iter):
            k = self._k()
            se = self._sigma_e()
            joule = (self.Jr**2 + self.Jz**2) / np.maximum(se, 1.0)
            # Cap extreme Joule spikes at cold–hot interfaces
            joule = np.clip(joule, 0.0, 5.0e12)
            U = np.zeros_like(self.T)
            U[self.plasma] = 2.5 * ar.U_rad(self.T[self.plasma])  # stronger NEC (no metal vapour)
            Qsh = self._sheath_heat()
            Qd = self._drop_heat()

            # Effective heat capacity (latent in mush)
            cp = self._cp()
            rho = self._rho()
            fl = self.fL
            mush = self.metal & (self.T > m.T_solidus) & (self.T < m.T_liquidus)
            cp_eff = cp + np.where(mush, m.latent / max(m.T_liquidus - m.T_solidus, 1.0), 0.0)

            T = self.T
            ke = 0.5 * (k[1:, :] + k[:-1, :])  # faces i+1/2 between i and i+1 → shape (nr-1, nz)
            kt = 0.5 * (k[:, 1:] + k[:, :-1])  # (nr, nz-1)
            # fluxes
            Fr = ke * (T[1:, :] - T[:-1, :]) / self.dr  # (nr-1, nz)
            Fz = kt * (T[:, 1:] - T[:, :-1]) / self.dz  # (nr, nz-1)
            r_face = self.ru[1:-1]  # length nr-1 interior faces? ru has nr+1
            # div at interior cells i=1..nr-2, j=1..nz-2
            div = np.zeros_like(T)
            # radial: (1/r) ∂(r Fr)/∂r
            rF = self.ru[1:-1, None] * Fr  # (nr-1, nz) with ru[1:-1] = faces between cells 0..nr-1
            # Fr[i] is flux between cell i and i+1, located at ru[i+1]
            # for cell i: (r_e Fr_e - r_w Fr_w) / (r_c dr)
            div[1:-1, :] = (
                self.ru[2:-1, None] * Fr[1:, :] - self.ru[1:-2, None] * Fr[:-1, :]
            ) / (self.rc[1:-1, None] * self.dr + 1e-16)
            div[:, 1:-1] += (Fz[:, 1:] - Fz[:, :-1]) / self.dz

            src = joule - U + Qsh + Qd
            dT = (div + src) / (rho * cp_eff + 1e-6)
            alpha_diff = k / (rho * cp_eff + 1e-6)
            h2 = min(self.dr, self.dz) ** 2
            alpha = p.relax_T * h2 / np.maximum(alpha_diff, 1e-12)
            alpha = np.minimum(alpha, 5e-5)
            # plasma is stiffer: smaller steps
            alpha = np.where(self.plasma, np.minimum(alpha, 1.5e-5), alpha)
            # only update interior
            T_new = T.copy()
            T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha[1:-1, 1:-1] * dT[1:-1, 1:-1]

            T_new[0, :] = T_new[1, :]
            T_new[-1, :] = p.T0
            bot = self.metal & (self.Z < self.z_min + 1.5 * self.dz)
            T_new[bot] -= (
                p.h_bottom * (T[bot] - p.Ta) / (m.rho * m.cp_s * self.dz + 1e-6) * 2e-4
            )

            T_new = np.where(self.plasma, np.clip(T_new, p.T0, p.T_plasma_max), T_new)
            T_new = np.where(self.metal, np.clip(T_new, p.T0, p.T_metal_max), T_new)
            far = self.plasma & (self.R > 0.7 * p.Lr)
            T_new[far] = 0.9 * T_new[far] + 0.1 * p.T0

            res = float(np.max(np.abs(T_new - self.T)))
            self.T = T_new
            self.fL = self.liquid_fraction(self.T)
        return res

    # --------------------------------------------------------------- flow
    def step_flow(self, n_iter: Optional[int] = None) -> None:
        """Simple explicit momentum with Lorentz + buoyancy; no pressure projection (demo)."""
        p = self.proc
        m = self.mat
        if n_iter is None:
            n_iter = p.n_flow
        if not p.enable_flow:
            return
        rho = self._rho()
        mu = self._mu()
        nr, nz = p.nr, p.nz
        dt = 2e-6
        for _ in range(n_iter):
            u, w = self.u, self.w
            # buoyancy in metal liquid
            buoy = np.zeros((nr, nz))
            liq = self.metal & (self.fL > 0.5)
            buoy[liq] = -p.g * m.beta * (self.T[liq] - m.T_ref)

            Fr = self.Fr / np.maximum(rho, 1e-3)
            Fz = self.Fz / np.maximum(rho, 1e-3) + buoy

            # update cell-centered then average to faces (very simplified)
            ur = 0.5 * (u[:-1, :] + u[1:, :])
            wz = 0.5 * (w[:, :-1] + w[:, 1:])
            ur = ur + dt * Fr
            wz = wz + dt * Fz

            # Darcy in mush / solid
            A = m.mush_C * (1.0 - self.fL) ** 2 / (self.fL**3 + m.mush_B)
            damp = 1.0 / (1.0 + dt * A / np.maximum(rho, 1.0))
            ur *= damp
            wz *= damp

            # zero in solid metal and far cold plasma weakly
            solid = self.metal & (self.fL < 0.05)
            ur[solid] = 0.0
            wz[solid] = 0.0
            ur[self.wire] = 0.0
            wz[self.wire] = 0.0

            if not p.enable_plasma_flow:
                ur[self.plasma] *= 0.0
                wz[self.plasma] *= 0.0
            else:
                # clip plasma velocities
                ur[self.plasma] = np.clip(ur[self.plasma], -p.u_clip_plasma, p.u_clip_plasma)
                wz[self.plasma] = np.clip(wz[self.plasma], -p.u_clip_plasma, p.u_clip_plasma)
            ur[liq] = np.clip(ur[liq], -p.u_clip_metal, p.u_clip_metal)
            wz[liq] = np.clip(wz[liq], -p.u_clip_metal, p.u_clip_metal)

            # map back to faces
            self.u[1:-1, :] = 0.5 * (ur[:-1, :] + ur[1:, :])
            self.u[0, :] = 0.0
            self.u[-1, :] = 0.0
            self.w[:, 1:-1] = 0.5 * (wz[:, :-1] + wz[:, 1:])
            self.w[:, 0] = 0.0
            self.w[:, -1] = 0.0

            # Marangoni on pool surface: approximate τ = dσ/dT * dT/dr on top liquid cells
            surf_j = []
            for i in range(nr):
                # highest metal cell in column
                metal_col = np.where(self.metal[i, :] & ~self.wire[i, :])[0]
                if metal_col.size == 0:
                    continue
                j_s = metal_col.max()
                if self.fL[i, j_s] < 0.5:
                    continue
                if i > 0 and i < nr - 1:
                    dTdr = (self.T[i + 1, j_s] - self.T[i - 1, j_s]) / (2 * self.dr)
                    tau = m.dsigma_dT * dTdr
                    self.u[i, j_s] += dt * tau / (m.mu + 1e-8) * 0.1

    # ----------------------------------------------------------- Kim–Na φ
    def _arc_pressure_on_surface(self) -> np.ndarray:
        """Pa(r) from plasma dynamic + magnetic pinch estimate near φ."""
        p = self.proc
        Pa = np.zeros(p.nr)
        for i in range(p.nr):
            # sample plasma just above surface
            z_s = self.phi_surf[i] + self.dz
            j = int(np.clip((z_s - self.z_min) / self.dz, 0, p.nz - 1))
            # magnetic pressure ~ B²/(2μ0)
            Bm = abs(self.Bth[i, j])
            Pm = Bm**2 / (2 * MU0)
            # stagnation ~ ρ v²
            ur = 0.5 * (self.u[i, j] + self.u[i + 1, j]) if i < p.nr else 0.0
            wz = 0.5 * (self.w[i, j] + self.w[i, min(j + 1, p.nz)])
            rho_p = float(ar.rho(self.T[i, j])) if self.plasma[i, j] else 0.05
            Pdyn = rho_p * (ur**2 + wz**2)
            Pa[i] = Pm + Pdyn
        # smooth + floor Gaussian fallback so first iters have pressure
        r = self.rc
        g = 150.0 * np.exp(-3.0 * (r / 0.0025) ** 2)
        Pa = 0.6 * Pa + 0.4 * g
        return Pa

    def _drop_pressure(self) -> np.ndarray:
        p = self.proc
        r_spot = 1.2e-3
        A = np.pi * r_spot**2
        p0 = p.p_drop_scale * p.m_dot * p.v_drop / max(A, 1e-10)
        return p0 * np.exp(-3.0 * (self.rc / r_spot) ** 2)

    def update_surface(self) -> None:
        """Relax φ(r) to γ κ = ρg φ + Pa + Pd + λ with volume constraint."""
        p = self.proc
        m = self.mat
        nr = p.nr
        Pa = self._arc_pressure_on_surface()
        Pd = self._drop_pressure()
        # liquid radius: molten near axis only (avoid whole-plate melt on coarse demo)
        liquid_r = np.zeros(nr, dtype=bool)
        for i in range(nr):
            if self.rc[i] > 4.0e-3:
                continue
            near = self.metal[i, :] & ~self.wire[i, :] & (self.Z[i, :] <= max(self.phi_surf[i], 0.0) + 2 * self.dz)
            if np.any(near) and np.max(self.T[i, near]) >= m.T_liquidus - 5.0:
                liquid_r[i] = True
        if not np.any(liquid_r):
            liquid_r = self.rc < 2.5e-3

        # Find λ so volume matches V_bead
        def volume(phi):
            return float(np.sum(2.0 * np.pi * self.rc * np.maximum(phi, 0.0) * self.dr))

        def relax_phi(lam: float, n: int = 80) -> np.ndarray:
            phi = self.phi_surf.copy()
            for _ in range(n):
                phi_new = phi.copy()
                for i in range(1, nr - 1):
                    if not liquid_r[i]:
                        phi_new[i] = 0.0
                        continue
                    # discrete curvature for axisymmetric surface (approx)
                    d1 = (phi[i + 1] - phi[i - 1]) / (2 * self.dr)
                    d2 = (phi[i + 1] - 2 * phi[i] + phi[i - 1]) / self.dr**2
                    # κ ≈ d2 / (1+d1²)^{3/2} + (1/r) d1 / sqrt(1+d1²)
                    s = np.sqrt(1.0 + d1 * d1)
                    kappa = d2 / (s**3 + 1e-12) + (d1 / (self.rc[i] * s + 1e-12))
                    # surface tension at local T
                    j_s = int(np.clip((0.0 - self.z_min) / self.dz, 0, p.nz - 1))
                    Ts = float(self.T[i, j_s])
                    gamma = m.sigma0 + m.dsigma_dT * (Ts - m.T_ref)
                    gamma = max(gamma, 0.2)
                    rhs = m.rho * p.g * phi[i] + Pa[i] + Pd[i] + lam
                    # residual: gamma*kappa - rhs → 0; update phi via curvature drive
                    # Use: move phi to reduce imbalance (artificial mobility)
                    res = gamma * kappa - rhs
                    phi_new[i] = phi[i] - 0.15 * self.dr**2 / max(gamma, 0.2) * res
                phi_new[0] = phi_new[1]
                phi_new[-1] = 0.0
                phi_new = np.where(liquid_r, phi_new, 0.0)
                phi_new = np.clip(phi_new, -0.001, 0.003)
                phi = (1 - p.relax_phi_surf) * phi + p.relax_phi_surf * phi_new
            return phi

        # Secant on λ for volume, then hard rescale to V_bead
        lam = 0.0
        phi = relax_phi(lam)
        V = volume(phi)
        V_tgt = p.V_bead
        lam1 = -5000.0 if V > V_tgt else 5000.0
        phi1 = relax_phi(lam1)
        V1 = volume(phi1)
        for _ in range(10):
            if abs(V1 - V) < 1e-14:
                break
            lam2 = lam1 - (V1 - V_tgt) * (lam1 - lam) / (V1 - V + 1e-16)
            lam2 = float(np.clip(lam2, -2e4, 2e4))
            phi2 = relax_phi(lam2)
            V2 = volume(phi2)
            lam, V, phi = lam1, V1, phi1
            lam1, V1, phi1 = lam2, V2, phi2
            if abs(V1 - V_tgt) / max(V_tgt, 1e-12) < 0.08:
                break
        # Enforce volume by scaling positive φ (Kim–Na λ imperfect on coarse grid)
        V1 = volume(phi1)
        if V1 > 1e-16:
            phi1 = phi1 * (V_tgt / V1)
        phi1 = np.clip(phi1, -0.0005, 0.0025)
        # Restrict support to molten radius (~pool)
        r_pool = 0.0
        for i in range(nr):
            if liquid_r[i]:
                r_pool = self.rc[i]
        if r_pool < 1e-4:
            r_pool = 3e-3
        phi1 = np.where(self.rc <= r_pool + self.dr, phi1, 0.0)
        V1 = volume(phi1)
        if V1 > 1e-16:
            phi1 = phi1 * (V_tgt / V1)
        self.phi_surf = 0.5 * self.phi_surf + 0.5 * np.clip(phi1, -0.0005, 0.0025)
        self._rebuild_masks()
        # Fill reinforcement cells with liquid metal temperature
        for i in range(nr):
            for j in range(p.nz):
                if 0.0 < self.Z[i, j] <= self.phi_surf[i] and not self.wire[i, j]:
                    self.metal[i, j] = True
                    self.plasma[i, j] = False
                    if self.T[i, j] < m.T_liquidus:
                        self.T[i, j] = max(self.T[i, j], m.T_liquidus + 10.0)
        self.fL = self.liquid_fraction(self.T)

    # ----------------------------------------------------------- metrics
    def pool_metrics(self) -> Tuple[float, float, float]:
        m = self.mat
        melt = self.metal & ~self.wire & (self.T >= m.T_liquidus) & (self.R < 5e-3)
        if not np.any(melt):
            RH = float(np.max(self.phi_surf))
            # BW from φ support
            wet = self.phi_surf > 1e-5
            BW = 2.0 * float(np.max(self.rc[wet])) if np.any(wet) else 0.0
            return 0.0, BW, RH
        below = melt & (self.Z <= 0.0)
        WD = -float(np.min(self.Z[below])) if np.any(below) else 0.0
        surf = melt & (self.Z > -0.8e-3)
        BW = 2.0 * float(np.max(self.R[surf])) if np.any(surf) else 0.0
        if BW < 1e-6:
            wet = self.phi_surf > 1e-5
            BW = 2.0 * float(np.max(self.rc[wet])) if np.any(wet) else 0.0
        RH = float(np.max(self.phi_surf))
        return WD, BW, RH

    def summary(self) -> dict:
        WD, BW, RH = self.pool_metrics()
        return {
            "WD_mm": 1e3 * WD,
            "BW_mm": 1e3 * BW,
            "RH_mm": 1e3 * RH,
            "gamma": WD / self.proc.H_plate if self.proc.H_plate > 0 else 0.0,
            "T_arc_max": float(np.max(self.T[self.plasma])) if np.any(self.plasma) else 0.0,
            "T_pool_max": float(np.max(self.T[self.metal & ~self.wire])) if np.any(self.metal) else 0.0,
            "I_arc": self.proc.I_arc,
            "V_bead_mm3": 1e9 * self.proc.V_bead,
            "bead_area_mm2": 1e6 * self.proc.bead_area,
        }

    def outer_iteration(self) -> dict:
        V = self.solve_em()
        res = self.relax_energy()
        self.step_flow()
        if self._iter % max(self.proc.surface_every, 1) == 0:
            self.update_surface()
        self._iter += 1
        WD, BW, RH = self.pool_metrics()
        s = {
            "iter": self._iter,
            "V": V,
            "res_T": res,
            "T_arc": float(np.max(self.T[self.plasma])) if np.any(self.plasma) else 0.0,
            "T_pool": float(np.max(self.T[self.metal & ~self.wire])) if np.any(self.metal & ~self.wire) else 0.0,
            "RH": RH,
            "WD": WD,
            "BW": BW,
        }
        self.diag.iter.append(self._iter)
        self.diag.T_arc_max.append(s["T_arc"])
        self.diag.T_pool_max.append(s["T_pool"])
        self.diag.RH.append(RH)
        self.diag.WD.append(WD)
        self.diag.BW.append(BW)
        self.diag.residual_T.append(res)
        self.diag.voltage.append(V)
        return s

    def run(self, n_outer: Optional[int] = None, progress: bool = True) -> Diagnostics:
        if n_outer is None:
            n_outer = self.proc.n_outer
        wall0 = time.perf_counter()
        for _ in range(n_outer):
            s = self.outer_iteration()
            if progress and (self._iter % 5 == 0 or self._iter == 1):
                print(
                    f"it={s['iter']:3d}  V~{s['V']:.1f}V  "
                    f"Tarc={s['T_arc']:.0f}K  Tpool={s['T_pool']:.0f}K  "
                    f"RH={1e3*s['RH']:.2f}mm  WD={1e3*s['WD']:.2f}mm  "
                    f"BW={1e3*s['BW']:.2f}mm  dT={s['res_T']:.1f}  "
                    f"wall={time.perf_counter()-wall0:.1f}s",
                    flush=True,
                )
        return self.diag


def bop_95A_axi(**overrides) -> ProcessAxi:
    base = dict(
        Lr=0.008,
        H_plate=0.003,
        Lz_gas=0.014,
        nr=48,
        nz=72,
        R_wire=0.0006,
        z_wire_tip=0.012,
        I_arc=95.0,
        v_feed=4.3 / 60.0,
        v_weld=0.9 / 60.0,
        n_outer=60,
        n_energy=25,
        n_flow=15,
        L_char=0.006,
    )
    base.update(overrides)
    return ProcessAxi(**base)


def quick_bop_95A_axi(**overrides) -> ProcessAxi:
    base = dict(
        nr=32,
        nz=48,
        n_outer=25,
        n_energy=15,
        n_flow=8,
        enable_plasma_flow=False,
    )
    base.update(overrides)
    return bop_95A_axi(**base)
