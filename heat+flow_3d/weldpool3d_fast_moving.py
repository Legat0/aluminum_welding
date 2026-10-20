# Faster explicit variant of the moving-frame WeldPool3D:
# DCT Neumann projection + 3D diffusion CFL. Same physics as weldpool3d_moving.

from __future__ import annotations

import numpy as np
from scipy.fft import dctn, idctn

from weldpool3d_moving import Process, WeldPool3D


class WeldPool3DFast(WeldPool3D):
    """Moving-frame weld-pool solver with DCT pressure projection and a 3D heat-CFL."""

    def suggest_dt(self) -> float:
        p = self.proc
        m = self.mat
        umax = max(
            float(np.max(np.abs(self.u))),
            float(np.max(np.abs(self.v))),
            float(np.max(np.abs(self.w))),
            float(abs(p.v_weld)),
            1e-6,
        )
        hmin = min(self.dx, self.dy, self.dz)
        dt_cfl = p.cfl * hmin / umax
        alpha = m.k_s / (m.rho * m.cp_s)
        # Explicit 3D conduction: dt < 1 / (2 α Σ 1/h²); keep 0.8 safety factor.
        dt_diff = 0.8 / (2.0 * alpha * (1.0 / self.dx**2 + 1.0 / self.dy**2 + 1.0 / self.dz**2))
        nu = m.mu / m.rho
        dt_visc = 0.12 * hmin**2 / max(nu, 1e-12)
        return float(min(p.dt_max, dt_cfl, dt_diff, dt_visc))

    def _build_poisson(self) -> None:
        nx, ny, nz = self.proc.nx, self.proc.ny, self.proc.nz
        dx, dy, dz = self.dx, self.dy, self.dz
        ix = np.arange(nx)
        jy = np.arange(ny)
        kz = np.arange(nz)
        # Eigenvalues of the 7-point Neumann Laplacian under DCT-II.
        lam = (
            (2.0 * np.cos(np.pi * ix / nx) - 2.0) / dx**2
        )[:, None, None] + (
            (2.0 * np.cos(np.pi * jy / ny) - 2.0) / dy**2
        )[None, :, None] + (
            (2.0 * np.cos(np.pi * kz / nz) - 2.0) / dz**2
        )[None, None, :]
        self._poisson_lam = lam.copy()
        self._poisson_lam[0, 0, 0] = 1.0
        self._poisson_solve = None

    def _project(self, u_star, v_star, w_star, dt) -> None:
        m = self.mat
        dx, dy, dz = self.dx, self.dy, self.dz
        div = (
            (u_star[1:, :, :] - u_star[:-1, :, :]) / dx
            + (v_star[:, 1:, :] - v_star[:, :-1, :]) / dy
            + (w_star[:, :, 1:] - w_star[:, :, :-1]) / dz
        )
        rhs = (m.rho / dt) * div
        rhs = rhs - rhs.mean()
        xhat = dctn(rhs, type=2, norm="ortho") / self._poisson_lam
        xhat[0, 0, 0] = 0.0
        phi = idctn(xhat, type=2, norm="ortho")
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


def quick_process(**overrides) -> Process:
    from weldpool3d_moving import quick_process as _base_quick

    return _base_quick(**overrides)
