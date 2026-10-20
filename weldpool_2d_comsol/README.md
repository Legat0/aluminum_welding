# COMSOL 6.2 — 2D weld pool

COMSOL Multiphysics counterpart of [`heat+flow_2d`](../heat+flow_2d): longitudinal plate, moving Gaussian surface flux, phase change (Al 5182), Marangoni, Boussinesq buoyancy, Darcy mush. No EM, arc pressure, free-surface deformation, or filler.

Coordinates: **x** along the weld, **y** through thickness (Python `z`). Gravity is \((0,-g)\).

## Requirements

- COMSOL Multiphysics **6.2**
- Licenses: **Heat Transfer Module**, **CFD Module**
- Path used by the builder: `%USERPROFILE%\.comsol\v62\applications\files\user\`  
  (COMSOL “Temporary and application files” security allows read/write there.)

## Files

| File | Role |
|---|---|
| [`weldpool_2d.mph`](weldpool_2d.mph) | Open in COMSOL Desktop; study `t_end = 4 s` |
| [`WeldPool2D.java`](WeldPool2D.java) | Model builder (Java API) |
| [`SmokeTest.java`](SmokeTest.java) | Short batch solve (Marangoni off for start-up stiffness) |
| [`build.ps1`](build.ps1) | Compile → write MPH → smoke solve |

## Open and run

1. Open `weldpool_2d.mph` in COMSOL 6.2.
2. Check **Parameters** (`q_max`, `v_weld`, `rb`, material data).
3. Compute **Transient weld pass** (Study 1). Full 4 s is expensive; start with a shorter `t_end` if needed.
4. Plots: Temperature (+ liquidus), liquid fraction `fL`, velocity + liquidus. Derived values: `Tmax`, `|u|max`.

## Rebuild

```powershell
cd weldpool_2d_comsol
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Smoke run: 0.02 s with full physics (ramped flux, capped Darcy, fL-scaled Marangoni shear). If the Desktop solve still fails at t≈0, check that you opened the rebuilt `weldpool_2d.mph` (re-run `build.ps1`).

## Physics map (vs `heat+flow_2d`)

| Python | COMSOL |
|---|---|
| Moving Gaussian `q_max`, `rb`, `v_weld` | Top `HeatFluxBoundary` + variables `q_arc`, `xc` |
| Top convection + radiation | Convective heat flux + surface-to-ambient radiation |
| Linear `fL`, apparent `cp` | `PhaseChangeMaterial` (`Ts`–`Tl`, `L_latent`) |
| Marangoni `μ ∂u/∂y = f_L (dσ/dT) ∂T/∂x` | Slip wall + weak contribution `test(u)*fL*dsigma_dT*Tx` (not Marangoni MP — that shears the solid and singularizes) |
| Buoyancy | Nonisothermal Flow, Boussinesq (`beta`, `Tref = Tl`) |
| Darcy–Carman–Kozeny | Volume force `F = -A_mush*u` |
| Grid ~120×30 | Mapped mesh (`Map` + size ≈ `Lx/nx_mesh`, `Lz/nz_mesh`) |

Do not name parameters `nx` / `ny` / `nz` — those clash with COMSOL boundary normals. Also avoid `g_const` (built-in).

### If you see “singularity / last time step not converged” at t≈0

That was the old Marangoni multiphysics shearing a fully solid top surface. Rebuild (`build.ps1`) or re-open the new `weldpool_2d.mph`. Stabilization in the current model:

- heat flux ramp `t_ramp`
- Darcy capped (`A_max`) with milder `mushC`/`mushB`
- Marangoni as weak form × `fL` (zero in solid)
- buoyancy only where `fL > 0`
