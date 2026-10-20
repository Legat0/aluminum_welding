"""Plot metrics/fields/Fig.5-style profile from results_95A.npz."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match weldpool_chen AA5754 / AA4043 mix
TS_PLATE, TL_PLATE = 880.0, 920.0
TS_WIRE, TL_WIRE = 850.0, 905.0

ROOT = Path(__file__).resolve().parent

TABLE2 = {
    "measured": {"RH_mm": 1.47, "WD_mm": 1.85, "BW_mm": 5.92, "gamma": 0.62},
    "chen_paper": {"RH_mm": 1.54, "WD_mm": 1.76, "BW_mm": 5.54, "gamma": 0.59},
}


def local_Ts_Tl(Yd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Y = np.clip(Yd, 0.0, 1.0)
    Ts = (1.0 - Y) * TS_PLATE + Y * TS_WIRE
    Tl = (1.0 - Y) * TL_PLATE + Y * TL_WIRE
    return Ts, Tl


def deepest_contour(T_yz: np.ndarray, zc: np.ndarray, T_thr: np.ndarray) -> np.ndarray:
    """For each y-column, deepest z where T >= threshold (NaN if none)."""
    ny, nz = T_yz.shape
    z_out = np.full(ny, np.nan)
    for j in range(ny):
        hit = T_yz[j, :] >= T_thr[j, :]
        if np.any(hit):
            z_out[j] = float(zc[np.where(hit)[0].min()])
    return z_out


def mirror_y(y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-plate y>=0 → full cross-section; drop NaN edges cleanly."""
    mask = np.isfinite(z)
    y_pos, z_pos = y[mask], z[mask]
    if y_pos.size == 0:
        return np.array([]), np.array([])
    # skip y≈0 duplicate when mirroring
    y_neg = -y_pos[::-1]
    z_neg = z_pos[::-1]
    if y_pos[0] < 1e-9:
        return np.concatenate([y_neg[:-1], y_pos]), np.concatenate([z_neg[:-1], z_pos])
    return np.concatenate([y_neg, y_pos]), np.concatenate([z_neg, z_pos])


def parabolic_reinforcement(BW_m: float, RH_m: float, n: int = 81) -> tuple[np.ndarray, np.ndarray]:
    half = 0.5 * BW_m
    y = np.linspace(-half, half, n)
    z = RH_m * (1.0 - (y / half) ** 2)
    return y, z


def plot_fig5_profile(
    data,
    summary: dict,
    table2: dict,
    out_path: Path | None = None,
    *,
    show: bool = False,
) -> plt.Figure:
    """Cross-section like Chen Fig. 5: RH above z=0, liquidus/solidus below."""
    xc, yc, zc = data["xc"], data["yc"], data["zc"]
    T, Yd = data["T"], data["Yd"]
    x_arc = float(data["x_arc"])
    i_arc = int(np.argmin(np.abs(xc - x_arc)))

    T_yz = T[i_arc, :, :]  # (ny, nz)
    Y_yz = Yd[i_arc, :, :]
    Ts, Tl = local_Ts_Tl(Y_yz)

    z_liq = deepest_contour(T_yz, zc, Tl)
    z_sol = deepest_contour(T_yz, zc, Ts)

    y_liq, z_liq_full = mirror_y(yc, z_liq)
    y_sol, z_sol_full = mirror_y(yc, z_sol)

    RH = summary["RH_mm"] * 1e-3
    BW = summary["BW_mm"] * 1e-3
    y_rh, z_rh = parabolic_reinforcement(BW, RH)

    meas = table2["measured"]
    paper = table2["chen_paper"]

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.axhline(0.0, color="0.5", lw=0.8)

    # Model curves (dashed, Fig. 5 style)
    ax.plot(
        1e3 * y_rh,
        1e3 * z_rh,
        "k--",
        lw=1.8,
        label="Reinforcement (this model)",
    )
    if y_liq.size:
        ax.plot(
            1e3 * y_liq,
            1e3 * z_liq_full,
            color="C2",
            ls="--",
            lw=1.6,
            label="Liquidus (this model)",
        )
    if y_sol.size:
        ax.plot(
            1e3 * y_sol,
            1e3 * z_sol_full,
            color="limegreen",
            ls="--",
            lw=1.6,
            label="Solidus (this model)",
        )

    # Table 2 reference markers (Position curves not digitized)
    ax.plot(0.0, meas["RH_mm"], "ko", ms=7, label="Table 2 measured RH/WD")
    ax.plot(0.0, -meas["WD_mm"], "ko", ms=7)
    ax.plot(
        [-0.5 * meas["BW_mm"], 0.5 * meas["BW_mm"]],
        [0.0, 0.0],
        "k:",
        lw=1.2,
        label=f"Table 2 measured BW={meas['BW_mm']:.2f} mm",
    )
    ax.plot(0.0, paper["RH_mm"], "bs", ms=6, label="Table 2 ArcWeld RH/WD")
    ax.plot(0.0, -paper["WD_mm"], "bs", ms=6)
    ax.plot(
        [-0.5 * paper["BW_mm"], 0.5 * paper["BW_mm"]],
        [0.05, 0.05],
        "b:",
        lw=1.0,
        label=f"Table 2 ArcWeld BW={paper['BW_mm']:.2f} mm",
    )

    ax.set_xlabel("Distance perpendicular to welding direction (mm)")
    ax.set_ylabel("Height relative to surface (mm)")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("BOP I=95 A — cross-section (Chen Fig. 5 style)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=140)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main() -> None:
    data = np.load(ROOT / "results_95A.npz", allow_pickle=True)
    chen = TABLE2["chen_paper"]
    summary = json.loads(str(data["summary_json"]))

    t = data["time"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].plot(t, 1e3 * data["pool_depth"], "C0")
    ax[0].axhline(chen["WD_mm"], color="k", ls="--", lw=1, label="Chen WD")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("WD [mm]")
    ax[0].legend()
    ax[0].set_title("Depth")
    ax[1].plot(t, 1e3 * data["pool_width"], "C1")
    ax[1].axhline(chen["BW_mm"], color="k", ls="--", lw=1, label="Chen BW")
    ax[1].set_xlabel("t [s]")
    ax[1].set_ylabel("BW [mm]")
    ax[1].legend()
    ax[1].set_title("Width")
    ax[2].plot(t, 1e3 * data["RH"], "C2")
    ax[2].axhline(chen["RH_mm"], color="k", ls="--", lw=1, label="Chen RH")
    ax[2].set_xlabel("t [s]")
    ax[2].set_ylabel("RH [mm]")
    ax[2].legend()
    ax[2].set_title("Reinforcement")
    fig.tight_layout()
    fig.savefig(ROOT / "fig_chen_95A_metrics.png", dpi=140)
    plt.close(fig)

    xc, yc, zc = data["xc"], data["yc"], data["zc"]
    T, fL, speed = data["T"], data["fL"], data["speed"]
    u, w = data["u"], data["w"]
    x_arc = float(data["x_arc"])
    j0 = 0
    i_arc = int(np.argmin(np.abs(xc - x_arc)))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    pcm = ax.pcolormesh(
        1e3 * xc, 1e3 * zc, T[:, j0, :].T, shading="auto", cmap="inferno"
    )
    ax.contour(
        1e3 * xc,
        1e3 * zc,
        fL[:, j0, :].T,
        levels=[0.01, 0.99],
        colors=["w", "cyan"],
        linewidths=1,
    )
    sx, sz = max(len(xc) // 20, 1), max(len(zc) // 6, 1)
    Xq, Zq = np.meshgrid(xc[::sx], zc[::sz], indexing="ij")
    ax.quiver(
        1e3 * Xq,
        1e3 * Zq,
        u[::sx, j0, ::sz],
        w[::sx, j0, ::sz],
        color="w",
        scale=2.0,
        width=0.003,
    )
    ax.axvline(1e3 * x_arc, color="lime", ls=":", lw=1)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("T, fL, V  (y~0)")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, label="T [K]", fraction=0.046)

    ax = axes[0, 1]
    pcm = ax.pcolormesh(
        1e3 * yc, 1e3 * zc, T[i_arc, :, :].T, shading="auto", cmap="inferno"
    )
    ax.contour(
        1e3 * yc,
        1e3 * zc,
        fL[i_arc, :, :].T,
        levels=[0.01, 0.99],
        colors=["w", "cyan"],
        linewidths=1,
    )
    ax.set_xlabel("y [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("T, fL  (x=x_arc)")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, label="T [K]", fraction=0.046)

    ax = axes[1, 0]
    pcm = ax.pcolormesh(
        1e3 * xc, 1e3 * yc, T[:, :, -1].T, shading="auto", cmap="inferno"
    )
    ax.contour(
        1e3 * xc,
        1e3 * yc,
        fL[:, :, -1].T,
        levels=[0.99],
        colors=["cyan"],
        linewidths=1.2,
    )
    ax.plot(1e3 * x_arc, 0, "g+", ms=12)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Surface T (z=0)")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, label="T [K]", fraction=0.046)

    ax = axes[1, 1]
    pcm = ax.pcolormesh(
        1e3 * xc, 1e3 * yc, speed[:, :, -1].T, shading="auto", cmap="viridis"
    )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("|u - u_solid| surface")
    ax.set_aspect("equal")
    fig.colorbar(pcm, ax=ax, label="m/s", fraction=0.046)
    fig.tight_layout()
    fig.savefig(ROOT / "fig_chen_95A_fields.png", dpi=140)
    plt.close(fig)

    plot_fig5_profile(data, summary, TABLE2, ROOT / "fig_chen_95A_profile.png")

    kw = {k: data[k] for k in data.files if k != "table2_json"}
    kw["table2_json"] = json.dumps(TABLE2)
    np.savez_compressed(ROOT / "results_95A.npz", **kw)

    print("summary:", summary)
    hdr = f"{'Source':20s}  {'RH':>7s}  {'WD':>7s}  {'BW':>7s}  {'gamma':>7s}"
    print(hdr)
    for name, r in (
        ("Measured", TABLE2["measured"]),
        ("ArcWeld paper", TABLE2["chen_paper"]),
        (
            "This model",
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
    print(
        "saved fig_chen_95A_metrics.png, fig_chen_95A_fields.png, "
        "fig_chen_95A_profile.png"
    )


if __name__ == "__main__":
    main()
