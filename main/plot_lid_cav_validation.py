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
from src.pinn import BackwardStepPINN, load_model_state, eval_uvp_batch_lc
from src.plotting import plot_lid_cavity_comparison


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

RE_VALUES = [100, 500, 800]

WIDTHS = [128, 128, 128, 128]
ACTIVATION = jax.nn.tanh

PINN_RESULTS = "results/lid_cavity/Re{re}/params.npz"

# OpenFOAM case directories
OF_CASES = {
    100: "openfoam/run/lid_cav_re100",
    500: "openfoam/run/lid_cav_re500",
    800: "openfoam/run/lid_cav_re800",
}
OF_TIME = "100"  # last time directory

OUT_DIR = "figures"


# ═══════════════════════════════════════════════════════════════════════════
# PINN evaluation
# ═══════════════════════════════════════════════════════════════════════════


def load_pinn_model(path):
    model = BackwardStepPINN(
        widths=WIDTHS, key=jax.random.key(0), activation=ACTIVATION, n_inputs=2,
    )
    data = np.load(path)
    if "_n" in data:
        load_model_state(model, path)
    else:
        n = sum(1 for k in data if k.startswith("W"))
        for i in range(n):
            model.ws[i].value = jnp.array(data[f"W{i}"])
            model.bs[i].value = jnp.array(data[f"b{i}"])
    return model


def eval_pinn(model, x_arr, y_arr):
    u, v, p = eval_uvp_batch_lc(
        model,
        jnp.array(x_arr, dtype=float),
        jnp.array(y_arr, dtype=float),
    )
    return np.array(u), np.array(v), np.array(p)


def vortex_centre(model, N=200):
    x1d = np.linspace(-1.0, 1.0, N)
    y1d = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x1d, y1d)
    u, _, _ = eval_pinn(model, X.ravel(), Y.ravel())
    u = u.reshape(N, N)
    dy = y1d[1] - y1d[0]
    psi = np.zeros_like(u)
    for j in range(1, N):
        psi[j, :] = psi[j - 1, :] + 0.5 * (u[j - 1, :] + u[j, :]) * dy
    idx = np.argmin(psi)
    iy, ix = np.unravel_index(idx, psi.shape)
    return float(x1d[ix]), float(y1d[iy])


# ═══════════════════════════════════════════════════════════════════════════
# OpenFOAM reader
# ═══════════════════════════════════════════════════════════════════════════


def _read_of_u(case_dir, time_dir=OF_TIME):
    u_path = os.path.join(case_dir, time_dir, "U")
    values = []
    reading = False
    n_cells = None
    with open(u_path) as fh:
        for line in fh:
            s = line.strip()
            if not reading and s.isdigit() and int(s) > 10:
                n_cells = int(s)
                reading = True
                continue
            if reading and s == "(":
                continue
            if reading and s.startswith(")"):
                break
            if reading and s:
                values.append([float(v) for v in s.strip("()").split()])

    uvw = np.array(values)
    N = int(round(n_cells ** 0.5))

    x1d = -1.0 + (np.arange(N) + 0.5) * 2.0 / N
    y1d = -1.0 + (np.arange(N) + 0.5) * 2.0 / N
    xi = np.tile(x1d, N)
    yi = np.repeat(y1d, N)

    return xi, yi, uvw[:, 0], uvw[:, 1]


def _read_of_p(case_dir, time_dir=OF_TIME):
    """Read the pressure field from an OpenFOAM lid-cavity case.

    Returns
    -------
    xi, yi : 1-D arrays of cell-centre coordinates  (length n_cells)
    p_of   : 1-D array of pressure values
    """
    p_path = os.path.join(case_dir, time_dir, "p")
    values = []
    reading = False
    n_cells = None
    with open(p_path) as fh:
        for line in fh:
            s = line.strip()
            if not reading and s.isdigit() and int(s) > 10:
                n_cells = int(s)
                reading = True
                continue
            if reading and s == "(":
                continue
            if reading and s.startswith(")"):
                break
            if reading and s:
                values.append(float(s))

    p_arr = np.array(values)
    N = int(round(n_cells ** 0.5))

    x1d = -1.0 + (np.arange(N) + 0.5) * 2.0 / N
    y1d = -1.0 + (np.arange(N) + 0.5) * 2.0 / N
    xi = np.tile(x1d, N)
    yi = np.repeat(y1d, N)

    return xi, yi, p_arr


def _of_to_grid(case_dir, N=200, time_dir=OF_TIME):
    """Interpolate OpenFOAM u, v, p onto a regular N×N meshgrid over [-1,1]².

    Returns
    -------
    X, Y     : 2-D meshgrids (shape N×N)
    u_g, v_g, p_g : 2-D arrays on that grid
    """
    xi, yi, u_of, v_of = _read_of_u(case_dir, time_dir)
    _, _, p_of = _read_of_p(case_dir, time_dir)

    pts = np.column_stack([xi, yi])

    x1d = np.linspace(-1.0, 1.0, N)
    y1d = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x1d, y1d)
    query = np.column_stack([X.ravel(), Y.ravel()])

    u_g = griddata(pts, u_of, query, method="linear").reshape(N, N)
    v_g = griddata(pts, v_of, query, method="linear").reshape(N, N)
    p_g = griddata(pts, p_of, query, method="linear").reshape(N, N)

    # Remove mean pressure so PINN and OF are on the same datum
    p_g -= np.nanmean(p_g)

    return X, Y, u_g, v_g, p_g


def of_probe_line(case_dir, axis, n_pts=300, time_dir=OF_TIME):
    xi, yi, u_of, v_of = _read_of_u(case_dir, time_dir)
    pts = np.column_stack([xi, yi])

    interior = np.linspace(-0.99, 0.99, n_pts)

    if axis == "x":
        query = np.column_stack([np.zeros(n_pts), interior])
        u_line = griddata(pts, u_of, query, method="linear")
        v_line = griddata(pts, v_of, query, method="linear")
        return interior, u_line, v_line

    if axis == "y":
        query = np.column_stack([interior, np.zeros(n_pts)])
        u_line = griddata(pts, u_of, query, method="linear")
        v_line = griddata(pts, v_of, query, method="linear")
        return interior, u_line, v_line

    raise ValueError("axis must be 'x' or 'y'")


# ═══════════════════════════════════════════════════════════════════════════
# Field comparison plots  (3×3: u / v / p  ×  OF / PINN / error)
# ═══════════════════════════════════════════════════════════════════════════


def make_field_comparison_plots(out_dir=OUT_DIR, N_grid=200):
    """Generate a 3×3 contourf comparison for each Re value."""
    os.makedirs(out_dir, exist_ok=True)

    for Re in RE_VALUES:
        pinn_path = PINN_RESULTS.format(re=int(Re))
        of_dir = OF_CASES.get(Re)

        if not os.path.exists(pinn_path):
            print(f"  Re={Re}: PINN file not found — {pinn_path}, skipping field plot")
            continue

        if of_dir is None or not os.path.isdir(of_dir):
            print(f"  Re={Re}: OF directory not found — {of_dir}, skipping field plot")
            continue

        print(f"  Re={Re}: building field comparison …")

        # ── Load PINN ──
        model = load_pinn_model(pinn_path)

        x1d = np.linspace(-1.0, 1.0, N_grid)
        y1d = np.linspace(-1.0, 1.0, N_grid)
        X, Y = np.meshgrid(x1d, y1d)
        u_p, v_p, p_p = eval_pinn(model, X.ravel(), Y.ravel())
        u_p = u_p.reshape(N_grid, N_grid)
        v_p = v_p.reshape(N_grid, N_grid)
        p_p = p_p.reshape(N_grid, N_grid)
        # Shift PINN pressure to zero mean so it is on the same datum as OF
        p_p -= np.nanmean(p_p)

        # ── Load + interpolate OpenFOAM ──
        try:
            X_of, Y_of, u_of, v_of, p_of = _of_to_grid(of_dir, N=N_grid)
        except Exception as exc:
            print(f"    WARNING: OF grid read failed for Re={Re}: {exc}, skipping")
            continue

        # ── Plot ──
        fname = f"lid_cavity_fields_Re{int(Re)}.png"
        plot_lid_cavity_comparison(
            X, Y,
            u_of, v_of, p_of,
            u_p, v_p, p_p,
            Re=Re,
            plots_dir=out_dir,
            filename=fname,
        )
        print(f"    Saved → {os.path.join(out_dir, fname)}")


# ═══════════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    n_cols = len(RE_VALUES)
    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(3.6 * n_cols, 7.2),
        squeeze=False,
    )

    for col, Re in enumerate(RE_VALUES):
        ax_u = axes[0][col]
        ax_v = axes[1][col]

        # ── PINN ──
        pinn_path = PINN_RESULTS.format(re=int(Re))
        if os.path.exists(pinn_path):
            print(f"  Re={Re}: loading PINN from {pinn_path}")
            model = load_pinn_model(pinn_path)

            n_pts = 400

            yy = np.linspace(-1.0, 1.0, n_pts)
            u_p, _, _ = eval_pinn(model, np.zeros(n_pts), yy)
            ax_u.plot(u_p, yy, color="steelblue", lw=1.8, label="PINN", zorder=3)

            xx = np.linspace(-1.0, 1.0, n_pts)
            _, v_p, _ = eval_pinn(model, xx, np.zeros(n_pts))
            ax_v.plot(xx, v_p, color="steelblue", lw=1.8, label="PINN", zorder=3)

            xv, yv = vortex_centre(model)
            print(f"    vortex centre: ({xv:.3f}, {yv:.3f})")
            ax_u.axhline(yv, color="steelblue", lw=0.7, ls=":", alpha=0.6,
                         label=f"vortex y = {yv:.3f}")
            ax_v.axvline(xv, color="steelblue", lw=0.7, ls=":", alpha=0.6,
                         label=f"vortex x = {xv:.3f}")

        else:
            print(f"  Re={Re}: PINN file not found — {pinn_path}")

        # ── OpenFOAM ──
        of_dir = OF_CASES.get(Re)
        if of_dir is not None and os.path.isdir(of_dir):
            print(f"  Re={Re}: loading OpenFOAM from {of_dir}")
            try:
                coord_u, of_u, _ = of_probe_line(of_dir, axis="x")
                ax_u.plot(of_u, coord_u, color="darkorange", ls="--",
                          lw=1.6, label="OpenFOAM", zorder=2)

                coord_v, _, of_v = of_probe_line(of_dir, axis="y")
                ax_v.plot(coord_v, of_v, color="darkorange", ls="--",
                          lw=1.6, label="OpenFOAM", zorder=2)
            except Exception as exc:
                print(f"    WARNING: OF read failed for Re={Re}: {exc}")
        elif of_dir is not None:
            print(f"  Re={Re}: OF directory not found — {of_dir}")

        # ── Axes cosmetics ──
        ax_u.axvline(0, color="0.7", lw=0.7, ls=":")
        ax_u.axhline(0, color="0.7", lw=0.7, ls=":")
        ax_u.set_xlim(-0.6, 1.05)
        ax_u.set_ylim(-1.0, 1.0)
        ax_u.set_title(f"Re = {int(Re)}", fontsize=11, pad=4)

        ax_v.axhline(0, color="0.7", lw=0.7, ls=":")
        ax_v.axvline(0, color="0.7", lw=0.7, ls=":")
        ax_v.set_xlim(-1.0, 1.0)
        ax_v.set_ylim(-0.5, 0.5)

        if col == 0:
            ax_u.set_ylabel(r"$y$", fontsize=10)
            ax_v.set_ylabel(r"$v / U_\mathrm{lid}$", fontsize=10)
        else:
            ax_u.tick_params(labelleft=False)
            ax_v.tick_params(labelleft=False)

        ax_u.legend(fontsize=8, frameon=False, loc="lower right")
        ax_v.legend(fontsize=8, frameon=False, loc="lower right")

    for col in range(n_cols):
        axes[0][col].set_xlabel(r"$u / U_\mathrm{lid}$  at  $x = 0$", fontsize=10)
        axes[1][col].set_xlabel(r"$x$  (at  $y = 0$)", fontsize=10)

    fig.tight_layout(pad=1.2, h_pad=2.0, w_pad=1.0)

    for ext in ("pdf", "png"):
        out_path = os.path.join(OUT_DIR, f"lid_cavity_validation.{ext}")
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved → {out_path}")

    plt.close(fig)

    # ── Field comparison plots (3×3 contourf) ──
    print("\nGenerating field comparison plots …")
    make_field_comparison_plots(out_dir=OUT_DIR)


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
