"""Fig.7-style and profile plots from results_axi_95A.npz."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
TABLE2 = {
    "measured": {"RH_mm": 1.47, "WD_mm": 1.85, "BW_mm": 5.92, "gamma": 0.62},
    "chen_paper": {"RH_mm": 1.54, "WD_mm": 1.76, "BW_mm": 5.54, "gamma": 0.59},
}


def main() -> None:
    data = np.load(ROOT / "results_axi_95A.npz", allow_pickle=True)
    summary = json.loads(str(data["summary_json"]))
    rc, zc = data["rc"], data["zc"]
    T = data["T"]
    ur, wz = data["u"], data["w"]
    phi = data["phi_surf"]
    metal = data["metal"].astype(bool)
    plasma = data["plasma"].astype(bool)
    wire = data["wire"].astype(bool)
    fL = data["fL"]

    # --- Fig.7 style: T + streamlines + surface ---
    fig, ax = plt.subplots(figsize=(6.5, 7.0))
    R, Z = np.meshgrid(rc, zc, indexing="ij")
    # mirror for full cross-section look
    Rm = np.concatenate([-rc[::-1], rc])
    Tm = np.concatenate([T[::-1, :], T], axis=0)  # (2*nr, nz)
    RR, ZZ = np.meshgrid(Rm, zc, indexing="xy")
    pcm = ax.pcolormesh(
        1e3 * RR,
        1e3 * ZZ,
        Tm.T,
        shading="auto",
        cmap="inferno",
        vmin=300,
        vmax=min(float(np.max(T)), 18000),
    )
    # surface φ mirrored
    y_phi = np.concatenate([-rc[::-1], rc])
    z_phi = np.concatenate([phi[::-1], phi])
    ax.plot(1e3 * y_phi, 1e3 * z_phi, "c-", lw=2.0, label="φ (Kim–Na)")
    ax.axhline(0.0, color="w", ls=":", lw=0.8)
    # streamlines in pool (metal liquid)
    speed = np.sqrt(ur**2 + wz**2)
    pool = metal & ~wire & (fL > 0.4) & (rc[:, None] < 4e-3)
    uu = np.where(pool, ur, 0.0)
    ww = np.where(pool, wz, 0.0)
    # subsample quiver on +r side
    si, sj = max(len(rc) // 12, 1), max(len(zc) // 10, 1)
    Rq, Zq = np.meshgrid(rc[::si], zc[::sj], indexing="ij")
    ax.quiver(
        1e3 * Rq,
        1e3 * Zq,
        uu[::si, ::sj],
        ww[::si, ::sj],
        color="w",
        scale=3.0,
        width=0.003,
        alpha=0.85,
    )
    # wire outline
    zw = float(data["z_wire_tip"])
    ax.plot(
        [0, 0],
        [1e3 * zw, 1e3 * float(np.max(zc))],
        color="0.8",
        lw=4,
        solid_capstyle="butt",
    )
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 14)
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("LTE axi I=95 A — T, pool flow, φ (Chen Fig. 7 style)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(pcm, ax=ax, label="T [K]", fraction=0.046)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_chen_axi_95A_fields.png", dpi=140)
    plt.close(fig)

    # --- Profile like Fig.5 ---
    # liquidus / solidus depth vs r
    Ts, Tl = 880.0, 920.0
    z_liq = np.full_like(rc, np.nan)
    z_sol = np.full_like(rc, np.nan)
    for i in range(len(rc)):
        col = metal[i, :] & ~wire[i, :]
        if not np.any(col):
            continue
        hit_l = col & (T[i, :] >= Tl)
        hit_s = col & (T[i, :] >= Ts)
        if np.any(hit_l):
            z_liq[i] = float(zc[np.where(hit_l)[0].min()])
        if np.any(hit_s):
            z_sol[i] = float(zc[np.where(hit_s)[0].min()])

    def mirror(y, z):
        msk = np.isfinite(z)
        yp, zp = y[msk], z[msk]
        if yp.size == 0:
            return yp, zp
        return np.concatenate([-yp[::-1], yp]), np.concatenate([zp[::-1], zp])

    yl, zl = mirror(rc, z_liq)
    ys, zs = mirror(rc, z_sol)
    yp, zp = mirror(rc, phi)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.axhline(0.0, color="0.5", lw=0.8)
    if yp.size:
        ax.plot(1e3 * yp, 1e3 * zp, "k--", lw=1.8, label="Reinforcement φ")
    if yl.size:
        ax.plot(1e3 * yl, 1e3 * zl, "C2--", lw=1.5, label="Liquidus")
    if ys.size:
        ax.plot(1e3 * ys, 1e3 * zs, color="limegreen", ls="--", lw=1.5, label="Solidus")
    meas, paper = TABLE2["measured"], TABLE2["chen_paper"]
    ax.plot(0, meas["RH_mm"], "ko", ms=7, label="Table 2 measured RH/WD")
    ax.plot(0, -meas["WD_mm"], "ko", ms=7)
    ax.plot(0, paper["RH_mm"], "bs", ms=6, label="Table 2 ArcWeld RH/WD")
    ax.plot(0, -paper["WD_mm"], "bs", ms=6)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("Distance perpendicular to welding direction (mm)")
    ax.set_ylabel("Height relative to surface (mm)")
    ax.set_title("LTE axi I=95 A — cross-section (Fig. 5 style)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_chen_axi_95A_profile.png", dpi=140)
    plt.close(fig)

    print("summary:", summary)
    print(
        f"{'Source':20s}  {'RH':>7s}  {'WD':>7s}  {'BW':>7s}  {'gamma':>7s}"
    )
    for name, r in (
        ("Measured", meas),
        ("ArcWeld paper", paper),
        (
            "This LTE axi",
            {
                "RH_mm": summary["RH_mm"],
                "WD_mm": summary["WD_mm"],
                "BW_mm": summary["BW_mm"],
                "gamma": summary["gamma"],
            },
        ),
    ):
        print(
            f"{name:20s}  {r['RH_mm']:7.2f}  {r['WD_mm']:7.2f}  "
            f"{r['BW_mm']:7.2f}  {r['gamma']:7.2f}"
        )
    print("saved fig_chen_axi_95A_fields.png, fig_chen_axi_95A_profile.png")


if __name__ == "__main__":
    main()
