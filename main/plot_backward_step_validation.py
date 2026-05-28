import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pinn import (
    BackwardStepPINN,
    load_model_state,
    eval_uvp_batch_bs_raw,
    FourierBackwardStepPINN,
    load_fourier_model,
)
from src.backward_step import normalize_xy
from src.plotting import (
    plot_spatial_error_map,
    plot_backward_step_comparison,
)


# ═══════════════════════════════════════════════════════════════════════════
# Geometry & network (must match training)
# ═══════════════════════════════════════════════════════════════════════════

X_MIN = -2.0
H_STEP = 1.0
H_CHAN = 2.0
U_MEAN = 1.0
X_MAX = 20.0

ACTIVATION = jax.nn.tanh


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

RE_VALUES = [100, 200, 400]

PINN_RESULTS = "results/backward_step_v2/Re{re}/params.npz"
OF_CASES = {
    100: "openfoam/run/backwardStep_parabolic_re100",
    200: "openfoam/run/backwardStep_parabolic_re200",
    400: "openfoam/run/backwardStep_parabolic_re400",
}
OF_TIME = "20"

X_PROBE = 7.0  # x/h for vertical u(y) profiles
Y_PROBE = 0.1  # y/h for horizontal u(x) profiles

OUT_DIR = "figures"


# ═══════════════════════════════════════════════════════════════════════════
# PINN evaluation
# ═══════════════════════════════════════════════════════════════════════════


def load_pinn_model(path):
    """Load a BackwardStepPINN or FourierBackwardStepPINN from a .npz file.

    Automatically detects Fourier models (saved by save_fourier_model) by
    checking for the ``_n_fourier`` key and delegates to load_fourier_model.
    """
    data = np.load(path)

    # ── Fourier-feature model ─────────────────────────────────────────────
    if "_n_fourier" in data:
        return load_fourier_model(path, key=jax.random.key(0), activation=ACTIVATION)

    # ── Standard BackwardStepPINN ─────────────────────────────────────────
    if "_n" in data:
        if "_widths" in data:
            widths = [int(w) for w in data["_widths"]]
            n_inputs = int(data["_n_inputs"]) if "_n_inputs" in data else 2
        else:
            widths = [128, 128, 128, 128]
            n_inputs = 2
            print(
                f"  WARNING: {path} has no _widths metadata. Using fallback {widths}."
            )

        model = BackwardStepPINN(
            widths=widths,
            key=jax.random.key(0),
            activation=ACTIVATION,
            n_inputs=n_inputs,
        )
        load_model_state(model, path)
        return model

    # ── Legacy format (W0/b0 keys) ────────────────────────────────────────
    n = sum(1 for k in data if k.startswith("W"))
    if n == 0:
        raise ValueError(f"No weight arrays found in {path}")

    n_inputs = int(data["W0"].shape[0])
    widths_inferred = [int(data[f"W{i}"].shape[1]) for i in range(n - 1)]
    model = BackwardStepPINN(
        widths=widths_inferred,
        key=jax.random.key(0),
        activation=ACTIVATION,
        n_inputs=n_inputs,
    )
    for i in range(n):
        model.ws[i].value = jnp.array(data[f"W{i}"])
        model.bs[i].value = jnp.array(data[f"b{i}"])
    return model


RE_X_MAX = {100: 20.0, 200: 20.0, 400: 30.0}


def eval_pinn(model, x_arr, y_arr, x_max=X_MAX):
    """Evaluate PINN (raw output, no ansatz) at arrays of (x, y) coordinates."""
    u, v, p = eval_uvp_batch_bs_raw(
        model,
        jnp.array(x_arr, dtype=float),
        jnp.array(y_arr, dtype=float),
        X_MIN,
        x_max,
        H_CHAN,
    )
    return np.array(u), np.array(v), np.array(p)


def eval_wall_shear_pinn(model, x_arr, x_max=X_MAX):
    """Compute ∂u/∂y at y=0 (bottom wall) via JAX forward-mode AD.

    Returns the non-dimensional velocity gradient, which is proportional to
    the wall shear stress.  Positive = forward flow, negative = recirculation.
    """
    ey = jnp.array([0.0, 1.0])
    xy_batch = jnp.stack(
        [jnp.array(x_arr, dtype=float), jnp.zeros(len(x_arr), dtype=float)],
        axis=-1,
    )

    def dudy_single(xy):
        def net_fn(z):
            x_n, y_n = normalize_xy(z[0], z[1], X_MIN, x_max, H_CHAN)
            raw = model(jnp.stack([x_n, y_n])[None])[0]
            return raw[0]   # u only

        _, dudy = jax.jvp(net_fn, (xy,), (ey,))
        return dudy

    return np.array(jax.vmap(dudy_single)(xy_batch))


def of_wall_shear(case_dir, n_pts=300, time_dir=OF_TIME):
    """Estimate ∂u/∂y at y=0 from OpenFOAM cell-centre data.

    Uses u(x, dy)/dy with dy equal to half the first-cell height (~0.025h).
    OpenFOAM enforces no-slip exactly so u(y=0)=0.
    """
    x_of, y_of, n_cells = _of_cell_centers(case_dir, time_dir=time_dir)
    uvw = _parse_of_vector_field(os.path.join(case_dir, time_dir, "U"), n_cells)

    # Use a probe height just above the bottom wall; the first cell centre is
    # at y ≈ H/(2·NY) = 1/40 = 0.025 h, so 0.03 h sits safely inside it.
    dy = 0.03
    xx = np.linspace(0.05, 19.95, n_pts)
    q = np.column_stack([xx, np.full(n_pts, dy)])
    u_dy = griddata(np.column_stack([x_of, y_of]), uvw[:, 0], q, method="linear")
    return xx, u_dy / dy


# ═══════════════════════════════════════════════════════════════════════════
# OpenFOAM native reader
# ═══════════════════════════════════════════════════════════════════════════

_OF_H_M = 0.1
_OF_LIN_M = 0.2
_OF_H2_M = 0.2
_OF_NY = 20
_OF_NX1 = 20


def _of_detect_mesh(case_dir, time_dir=OF_TIME):
    """
    Auto-detect mesh parameters from a backwardStep case directory.
    Returns (n_cells, nx_exp, Ltot_m).
    """
    pts_file = os.path.join(case_dir, "constant", "polyMesh", "points")
    with open(pts_file) as fh:
        txt = fh.read()

    x_max_pts = 0.0
    reading = False
    for line in txt.split("\n"):
        s = line.strip()
        if s.isdigit() and int(s) > 100:
            reading = True
            continue
        if reading and s == "(":
            continue
        if reading and s.startswith(")"):
            break
        if reading and s:
            vals = s.strip("()").split()
            if len(vals) >= 1:
                x_max_pts = max(x_max_pts, float(vals[0]))
    Ltot_m = x_max_pts

    u_file = os.path.join(case_dir, time_dir, "U")
    n_cells = None
    with open(u_file) as fh:
        for line in fh:
            s = line.strip()
            if s.isdigit() and int(s) > 100:
                n_cells = int(s)
                break
    if n_cells is None:
        raise RuntimeError(f"Could not find cell count in {u_file}")

    nx_exp = (n_cells - _OF_NX1 * _OF_NY) // (2 * _OF_NY)
    return n_cells, nx_exp, Ltot_m


def _of_cell_centers(case_dir, time_dir=OF_TIME):
    """Return OpenFOAM cell-centre coordinates in PINN non-dimensional units."""
    n_cells, nx_exp, Ltot_m = _of_detect_mesh(case_dir, time_dir=time_dir)
    H, Lin, H2 = _OF_H_M, _OF_LIN_M, _OF_H2_M

    def _block_centres(x0, x1, nx, y0, y1, ny):
        xc = x0 + (x1 - x0) * (np.arange(nx) + 0.5) / nx
        yc = y0 + (y1 - y0) * (np.arange(ny) + 0.5) / ny
        Xi, Yi = np.meshgrid(xc, yc, indexing="xy")
        return Xi.ravel(), Yi.ravel()

    xb1, yb1 = _block_centres(0, Lin, _OF_NX1, H, H2, _OF_NY)
    xb2, yb2 = _block_centres(Lin, Ltot_m, nx_exp, H, H2, _OF_NY)
    xb3, yb3 = _block_centres(Lin, Ltot_m, nx_exp, 0, H, _OF_NY)

    x_phys = np.concatenate([xb1, xb2, xb3])
    y_phys = np.concatenate([yb1, yb2, yb3])

    x_nd = x_phys / H - Lin / H
    y_nd = y_phys / H
    return x_nd, y_nd, n_cells


def _parse_of_vector_field(filepath, n_cells):
    """Parse the internal field of a vector OpenFOAM file, for example U."""
    with open(filepath) as fh:
        txt = fh.read()

    values = []
    reading = False
    for line in txt.split("\n"):
        s = line.strip()
        if s == str(n_cells):
            reading = True
            continue
        if reading and s == "(":
            continue
        if reading and s.startswith(")"):
            break
        if reading and s:
            values.append([float(v) for v in s.strip("()").split()])

    arr = np.array(values)
    if len(arr) != n_cells:
        raise RuntimeError(
            f"Expected {n_cells} velocity vectors, got {len(arr)} in {filepath}"
        )
    return arr


def _parse_of_scalar_field(filepath, n_cells):
    """Parse the internal field of a scalar OpenFOAM file (e.g. p)."""
    with open(filepath) as fh:
        txt = fh.read()

    values = []
    reading = False
    for line in txt.split("\n"):
        s = line.strip()
        if s == str(n_cells):
            reading = True
            continue
        if reading and s == "(":
            continue
        if reading and s.startswith(")"):
            break
        if reading and s:
            try:
                values.append(float(s))
            except ValueError:
                pass

    arr = np.array(values)
    if len(arr) != n_cells:
        raise RuntimeError(
            f"Expected {n_cells} pressure values, got {len(arr)} in {filepath}"
        )
    return arr


def of_probe_line(case_dir, x_fixed=None, y_fixed=None, n_pts=300, time_dir=OF_TIME):
    """
    Extract a probe line from a native OpenFOAM case by interpolating
    cell-centred data with scipy griddata.

    Returns
    -------
    coord : y values for vertical line, x values for horizontal line
    ux    : streamwise velocity u/U_mean
    uy    : cross-stream velocity v/U_mean
    """
    u_path = os.path.join(case_dir, time_dir, "U")
    x_nd, y_nd, n_cells = _of_cell_centers(case_dir, time_dir=time_dir)
    uvw = _parse_of_vector_field(u_path, n_cells)
    pts = np.column_stack([x_nd, y_nd])

    if x_fixed is not None:
        y_lo = 0.0 if x_fixed >= 0 else H_STEP
        coord = np.linspace(y_lo, H_CHAN, n_pts)
        q = np.column_stack([np.full(n_pts, x_fixed), coord])
        ux = griddata(pts, uvw[:, 0], q, method="linear")
        uy = griddata(pts, uvw[:, 1], q, method="linear")
        return coord, ux, uy

    if y_fixed is not None:
        x_lo, x_hi = 0.05, 19.95
        coord = np.linspace(x_lo, x_hi, n_pts)
        q = np.column_stack([coord, np.full(n_pts, y_fixed)])
        ux = griddata(pts, uvw[:, 0], q, method="linear")
        uy = griddata(pts, uvw[:, 1], q, method="linear")
        return coord, ux, uy

    raise ValueError("Specify x_fixed or y_fixed")


# ═══════════════════════════════════════════════════════════════════════════
# Optional benchmark / reference data
# ═══════════════════════════════════════════════════════════════════════════

REF_U_PROFILE = {
    100: None,
    200: None,
    400: None,
}

# Bottom row is now u(x, y=Y_PROBE), not v(x, y=Y_PROBE).
REF_UX_LINE = {
    100: None,
    200: None,
    400: None,
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def draw_step_geometry(ax, orientation="vertical", x_fixed=None):
    """
    Add geometry markers without misleading shading.

    For a vertical profile downstream of the step, the whole range 0 < y < 2h
    is fluid, so we only mark the old step height y/h = 1. Shading y < 1
    would incorrectly imply that region is blocked.
    """
    if orientation == "vertical":
        if x_fixed is not None and x_fixed < 0.0:
            ax.axhspan(0, H_STEP, color="0.88", zorder=0, lw=0)
        ax.axhline(H_STEP, color="0.6", lw=0.8, ls="--")
    else:
        ax.axvline(0, color="0.6", lw=0.8, ls=":")


def first_negative_to_positive_crossing(x, u):
    """
    Estimate first x where u changes from negative to positive.

    This is only a probe-line estimate. The formal reattachment location is
    usually defined from the lower-wall shear stress, not from u at y=0.5h.
    """
    x = np.asarray(x)
    u = np.asarray(u)
    finite = np.isfinite(x) & np.isfinite(u)
    x = x[finite]
    u = u[finite]

    if len(x) < 2:
        return None

    for i in range(len(x) - 1):
        if u[i] < 0.0 <= u[i + 1]:
            dx = x[i + 1] - x[i]
            du = u[i + 1] - u[i]
            if abs(du) < 1e-14:
                return x[i]
            return x[i] - u[i] * dx / du
    return None


def make_field_comparison_plots(
    out_dir, pinn_results=None, label="v2", model_loader=None
):
    """For each Re, produce the 3×3 comparison figure (OF | PINN | error) for u, v, p.

    All Re values share the same colorbar range per field so plots are directly
    comparable: one range for u, one for v, one for p, one for each error field.
    """
    from scipy.interpolate import griddata

    if pinn_results is None:
        pinn_results = PINN_RESULTS
    if model_loader is None:
        model_loader = load_pinn_model

    # ── Pass 1: collect all fields, compute global colorbar limits ────────
    all_data = {}  # Re → dict of arrays
    for Re in RE_VALUES:
        pinn_path = pinn_results.format(re=int(Re))
        of_dir = OF_CASES.get(Re)
        if not os.path.exists(pinn_path) or of_dir is None or not os.path.isdir(of_dir):
            continue

        x_max_re = RE_X_MAX.get(Re, X_MAX)
        model = model_loader(pinn_path)

        Nx, Ny = 400, 80
        x1d = np.linspace(X_MIN, x_max_re, Nx)
        y1d = np.linspace(0.0, H_CHAN, Ny)
        Xg, Yg = np.meshgrid(x1d, y1d)
        fluid = ~((Xg < 0.0) & (Yg < H_STEP))

        u_pinn, v_pinn, p_pinn = eval_pinn(
            model, Xg.ravel(), Yg.ravel(), x_max=x_max_re
        )
        u_pinn = np.where(fluid, u_pinn.reshape(Ny, Nx), np.nan)
        v_pinn = np.where(fluid, v_pinn.reshape(Ny, Nx), np.nan)
        p_pinn = np.where(fluid, p_pinn.reshape(Ny, Nx), np.nan)
        p_pinn = p_pinn - np.nanmean(p_pinn)

        try:
            x_of, y_of, n_cells = _of_cell_centers(of_dir)
            q = np.column_stack([Xg.ravel(), Yg.ravel()])
            of_pts = np.column_stack([x_of, y_of])

            uvw = _parse_of_vector_field(os.path.join(of_dir, OF_TIME, "U"), n_cells)
            u_of = griddata(of_pts, uvw[:, 0], q, method="linear").reshape(Ny, Nx)
            v_of = griddata(of_pts, uvw[:, 1], q, method="linear").reshape(Ny, Nx)
            u_of = np.where(fluid, u_of, np.nan)
            v_of = np.where(fluid, v_of, np.nan)

            p_path = os.path.join(of_dir, OF_TIME, "p")
            if os.path.exists(p_path):
                p_raw = _parse_of_scalar_field(p_path, n_cells)
                p_of = griddata(of_pts, p_raw, q, method="linear").reshape(Ny, Nx)
                p_of = np.where(fluid, p_of, np.nan)
                p_of = p_of - np.nanmean(p_of)
            else:
                p_of = np.zeros_like(u_of)
        except Exception as exc:
            print(f"    WARNING: OF interpolation failed for Re={Re}: {exc}")
            continue

        all_data[Re] = dict(
            Xg=Xg,
            Yg=Yg,
            u_of=u_of,
            v_of=v_of,
            p_of=p_of,
            u_pinn=u_pinn,
            v_pinn=v_pinn,
            p_pinn=p_pinn,
        )

    if not all_data:
        print(f"  No data found for [{label}], skipping field comparison.")
        return

    # Global limits: take the symmetric range from OpenFOAM (reference) across all Re
    def _global_lim(arrays):
        lo = min(np.nanmin(a) for a in arrays)
        hi = max(np.nanmax(a) for a in arrays)
        return lo, hi

    vlims = {
        "u": _global_lim([d["u_of"] for d in all_data.values()]),
        "v": _global_lim([d["v_of"] for d in all_data.values()]),
        "p": _global_lim([d["p_of"] for d in all_data.values()]),
        "u_err": max(
            np.nanmax(np.abs(d["u_pinn"] - d["u_of"])) for d in all_data.values()
        ),
        "v_err": max(
            np.nanmax(np.abs(d["v_pinn"] - d["v_of"])) for d in all_data.values()
        ),
        "p_err": max(
            np.nanmax(np.abs(d["p_pinn"] - d["p_of"])) for d in all_data.values()
        ),
    }
    print(f"  [{label}] shared colorbar limits:")
    for k, v in vlims.items():
        print(f"    {k}: {v}")

    # ── Pass 2: plot with shared limits ───────────────────────────────────
    for Re, d in all_data.items():
        print(f"  Field comparison [{label}] Re={Re}...")
        plots_dir = f"plots/backward_step_{label}/Re{int(Re)}"
        os.makedirs(plots_dir, exist_ok=True)

        plot_backward_step_comparison(
            d["Xg"],
            d["Yg"],
            d["u_of"],
            d["v_of"],
            d["p_of"],
            d["u_pinn"],
            d["v_pinn"],
            d["p_pinn"],
            Re,
            h_step=H_STEP,
            plots_dir=plots_dir,
            filename="field_comparison.png",
            vlims=vlims,
        )
        plot_backward_step_comparison(
            d["Xg"],
            d["Yg"],
            d["u_of"],
            d["v_of"],
            d["p_of"],
            d["u_pinn"],
            d["v_pinn"],
            d["p_pinn"],
            Re,
            h_step=H_STEP,
            plots_dir=out_dir,
            filename=f"field_comparison_{label}_Re{int(Re)}.png",
            vlims=vlims,
        )


def make_spatial_error_plots(out_dir, pinn_results=None, label="v2", model_loader=None):
    """For each Re, produce a 2-D heatmap of |u_pinn − u_OF| over the domain."""
    from scipy.interpolate import griddata

    if pinn_results is None:
        pinn_results = PINN_RESULTS
    if model_loader is None:
        model_loader = load_pinn_model

    for Re in RE_VALUES:
        pinn_path = pinn_results.format(re=int(Re))
        of_dir = OF_CASES.get(Re)

        if not os.path.exists(pinn_path):
            print(f"  Spatial error [{label}] Re={Re}: PINN file not found, skipping.")
            continue
        if of_dir is None or not os.path.isdir(of_dir):
            print(
                f"  Spatial error [{label}] Re={Re}: OpenFOAM directory not found, skipping."
            )
            continue

        print(f"  Spatial error map [{label}] Re={Re}...")
        x_max_re = RE_X_MAX.get(Re, X_MAX)
        model = model_loader(pinn_path)

        # Regular grid over the fluid domain
        Nx, Ny = 400, 80
        x1d = np.linspace(X_MIN, x_max_re, Nx)
        y1d = np.linspace(0.0, H_CHAN, Ny)
        Xg, Yg = np.meshgrid(x1d, y1d)  # (Ny, Nx)
        fluid = ~((Xg < 0.0) & (Yg < H_STEP))  # mask out step block

        # PINN u on the grid
        u_pinn, _, _ = eval_pinn(model, Xg.ravel(), Yg.ravel(), x_max=x_max_re)
        u_pinn = u_pinn.reshape(Ny, Nx)
        u_pinn = np.where(fluid, u_pinn, np.nan)

        # OpenFOAM u interpolated onto the same grid
        try:
            x_of, y_of, n_cells = _of_cell_centers(of_dir)
            u_path = os.path.join(of_dir, OF_TIME, "U")
            uvw = _parse_of_vector_field(u_path, n_cells)
            q = np.column_stack([Xg.ravel(), Yg.ravel()])
            u_of_flat = griddata(
                np.column_stack([x_of, y_of]),
                uvw[:, 0],
                q,
                method="linear",
            )
            u_of = u_of_flat.reshape(Ny, Nx)
            u_of = np.where(fluid, u_of, np.nan)
        except Exception as exc:
            print(f"    WARNING: OF interpolation failed for Re={Re}: {exc}")
            continue

        plots_dir = f"plots/backward_step_{label}/Re{int(Re)}"
        os.makedirs(plots_dir, exist_ok=True)
        plot_spatial_error_map(
            Xg,
            Yg,
            u_pinn,
            u_of,
            Re,
            h_step=H_STEP,
            x_min=X_MIN,
            plots_dir=plots_dir,
            filename="spatial_error_map.png",
        )
        # Also save to the shared figures dir
        plot_spatial_error_map(
            Xg,
            Yg,
            u_pinn,
            u_of,
            Re,
            h_step=H_STEP,
            x_min=X_MIN,
            plots_dir=out_dir,
            filename=f"spatial_error_map_{label}_Re{int(Re)}.png",
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n_cols = len(RE_VALUES)
    fig, axes = plt.subplots(
        1,
        n_cols,
        figsize=(3.8 * n_cols, 3.6),
        squeeze=False,
    )

    for col, Re in enumerate(RE_VALUES):
        ax = axes[0][col]

        x_max_re = RE_X_MAX.get(Re, X_MAX)
        pinn_cross = None
        of_cross = None

        # ── PINN ──────────────────────────────────────────────────────────
        pinn_path = PINN_RESULTS.format(re=int(Re))
        if os.path.exists(pinn_path):
            print(f"  Re={Re}: loading PINN from {pinn_path}")
            model = load_pinn_model(pinn_path)

            n_pts = 400
            x_probe_max = min(x_max_re - 0.5, 19.95)
            xx = np.linspace(0.05, x_probe_max, n_pts)

            u_p_x, _, _ = eval_pinn(model, xx, np.full(n_pts, Y_PROBE), x_max=x_max_re)
            ax.plot(xx, u_p_x, color="steelblue", lw=1.8, label="PINN", zorder=3)
            pinn_cross = first_negative_to_positive_crossing(xx, u_p_x)
        else:
            print(f"  Re={Re}: PINN file not found — {pinn_path}")

        # ── OpenFOAM ──────────────────────────────────────────────────────
        of_dir = OF_CASES.get(Re)
        of_x_max = None
        if of_dir is not None and os.path.isdir(of_dir):
            print(f"  Re={Re}: loading OpenFOAM from {of_dir}")
            try:
                coord_x, of_ux_x, _ = of_probe_line(of_dir, y_fixed=Y_PROBE)
                of_x_max = coord_x[-1]
                ax.plot(
                    coord_x, of_ux_x,
                    color="darkorange", ls="--", lw=1.6,
                    label="OpenFOAM", zorder=2,
                )
                of_cross = first_negative_to_positive_crossing(coord_x, of_ux_x)
            except Exception as exc:
                print(f"    WARNING: OF read failed for Re={Re}: {exc}")
        elif of_dir:
            print(f"  Re={Re}: OF directory not found — {of_dir}")

        # ── Geometry & cosmetics ──────────────────────────────────────────
        draw_step_geometry(ax, "horizontal")
        ax.axhline(0, color="0.7", lw=0.7, ls=":")
        ax.set_xlim(0, of_x_max + 0.5 if of_x_max is not None else x_max_re)
        ax.set_title(f"Re = {int(Re)}", fontsize=11, pad=4)
        ax.set_xlabel(f"$x/h$ at $y={Y_PROBE:.1f}h$", fontsize=10)

        text_lines = []
        if pinn_cross is not None:
            text_lines.append(rf"PINN $u=0$: {pinn_cross:.2f}$h$")
        if of_cross is not None:
            text_lines.append(rf"OF $u=0$: {of_cross:.2f}$h$")
        if text_lines:
            ax.text(
                0.03, 0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                va="top", ha="left", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.75),
            )

        if col == 0:
            ax.set_ylabel(r"$u/U_{\mathrm{mean}}$", fontsize=10)
        else:
            ax.tick_params(labelleft=False)

        ax.legend(fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout(pad=1.2, w_pad=1.0)

    for ext in ("pdf", "png"):
        out_path = os.path.join(OUT_DIR, f"backward_step_validation_corrected.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {out_path}")

    plt.close(fig)

    # ── Field comparison (OF | PINN | error) ─────────────────────────────
    print("\nGenerating field comparison plots...")
    make_field_comparison_plots(OUT_DIR)

    # ── Spatial error maps ────────────────────────────────────────────────
    print("\nGenerating spatial error maps...")
    make_spatial_error_plots(OUT_DIR)



if __name__ == "__main__":
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.linewidth": 0.8,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
        }
    )
    main()
