import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from src.analytical import taylor_green_u, taylor_green_v, taylor_green_p
from src.pinn import TaylorGreenPINN, load_model_state

# ── paths ────────────────────────────────────────────────────────────────────
FOAM_DIR = "openfoam/run/taylor_green_re200"
PINN_RESULTS_DIR = "results/taylor_green/Re200"
PLOTS_DIR = "plots/taylor_green/openfoam"
os.makedirs(PLOTS_DIR, exist_ok=True)

# X index in filename → actual simulation time
FOAM_FILES = {
    0.25: os.path.join(FOAM_DIR, "taylor_green_re200_0.csv"),
    0.5: os.path.join(FOAM_DIR, "taylor_green_re200_1.csv"),
    1.0: os.path.join(FOAM_DIR, "taylor_green_re200_3.csv"),
    2.0: os.path.join(FOAM_DIR, "taylor_green_re200_7.csv"),
}

# ── PINN model ──────────────────────────────────────────────────────────────
_params_path = os.path.join(PINN_RESULTS_DIR, "params.npz")
_data = np.load(_params_path)
_n_inputs = int(_data["W0"].shape[0]) if "W0" in _data else 5
pinn_model = TaylorGreenPINN(widths=[64, 64, 64], key=jax.random.key(0),
                              activation=jax.nn.tanh, n_inputs=_n_inputs)
if "_n" in _data:
    load_model_state(pinn_model, _params_path)
else:
    n = sum(1 for k in _data if k.startswith("W"))
    for i in range(n):
        pinn_model.ws[i].value = jnp.array(_data[f"W{i}"])
        pinn_model.bs[i].value = jnp.array(_data[f"b{i}"])

def _eval_uvp_batch_compat(model, x, y, t, t_max=2.0, n_inputs=5):
    """Evaluate TG model, handling both legacy 3-input and current 5-input encoding."""
    from src.taylor_green import _net_tg_physical

    def _net_legacy(m, z, tm):
        x_, y_, t_ = z[0], z[1], z[2]
        t_n = 2.0 * t_ / tm - 1.0
        inp = jnp.stack([x_, y_, t_n])
        raw = m(inp[None])[0]
        u_ic = jnp.cos(x_) * jnp.sin(y_)
        v_ic = -jnp.sin(x_) * jnp.cos(y_)
        p_ic = -0.25 * (jnp.cos(2.0 * x_) + jnp.cos(2.0 * y_))
        return jnp.array([u_ic + t_ * raw[0], v_ic + t_ * raw[1], p_ic + t_ * raw[2]])

    net_fn = _net_tg_physical if n_inputs == 5 else _net_legacy

    def eval_single(xi, yi, ti):
        out = net_fn(model, jnp.stack([xi, yi, ti]), t_max)
        return out[0], out[1], out[2]

    return jax.vmap(eval_single)(x, y, t)


# ── settings ─────────────────────────────────────────────────────────────────
Re = 200.0
nu = 1.0 / Re
N = 100
t_eval_list = [0.25, 0.5, 1.0, 2.0]

x = jnp.linspace(0, 2 * jnp.pi, N)
y = jnp.linspace(0, 2 * jnp.pi, N)
X, Y = jnp.meshgrid(x, y)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_foam_csv(filepath):
    data = np.genfromtxt(filepath, delimiter=",", names=True)
    pts_x = data["Points0"]
    pts_y = data["Points1"]
    u = data["U0"]
    v = data["U1"]
    p = data["p"]
    return pts_x, pts_y, u, v, p


def interpolate_to_grid(pts_x, pts_y, values, X, Y):
    points = np.column_stack([pts_x, pts_y])
    target = np.column_stack([np.array(X).ravel(), np.array(Y).ravel()])

    interp = griddata(points, values, target, method="linear")

    nan_mask = np.isnan(interp)
    if nan_mask.any():
        interp[nan_mask] = griddata(points, values, target[nan_mask], method="nearest")

    return interp.reshape(X.shape)


def relative_l2_error(pred, exact):
    return float(jnp.linalg.norm(pred - exact) / jnp.linalg.norm(exact))


def plot_comparison_foam(
    X, Y, u_exact, v_exact, p_exact, u_foam, v_foam, p_foam, t, plots_dir
):
    fields = [
        ("u", u_exact, u_foam),
        ("v", v_exact, v_foam),
        ("p", p_exact, p_foam),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    for row, (name, exact, foam) in enumerate(fields):
        exact_arr = np.array(exact)
        norm = np.linalg.norm(exact_arr)
        rel_err = np.abs(np.array(foam) - exact_arr) / norm

        im0 = axes[row, 0].contourf(X, Y, exact, levels=50, cmap="RdBu_r")
        axes[row, 0].set_title(f"{name} exact, t={t}")
        plt.colorbar(im0, ax=axes[row, 0])

        im1 = axes[row, 1].contourf(X, Y, foam, levels=50, cmap="RdBu_r")
        axes[row, 1].set_title(f"{name} OpenFOAM, t={t}")
        plt.colorbar(im1, ax=axes[row, 1])

        im2 = axes[row, 2].contourf(X, Y, rel_err, levels=50, cmap="hot_r")
        axes[row, 2].set_title(f"{name} relative L2 error, t={t}")
        plt.colorbar(im2, ax=axes[row, 2])

    plt.tight_layout()
    fname = os.path.join(plots_dir, f"foam_comparison_t{t}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


def plot_error_vs_time(t_list, foam_errors, pinn_errors, plots_dir):
    fields = ["u", "v", "p"]
    colors = {"OpenFOAM": "steelblue", "PINN": "darkorange"}
    markers = {"OpenFOAM": "o", "PINN": "s"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, field, idx in zip(axes, fields, range(3)):
        foam_vals = [foam_errors[t][idx] for t in t_list]
        pinn_vals = [pinn_errors[t][idx] for t in t_list]
        for label, vals in [("OpenFOAM", foam_vals), ("PINN", pinn_vals)]:
            ax.semilogy(
                t_list, vals,
                marker=markers[label], color=colors[label],
                label=label, linewidth=1.5, markersize=6,
            )
        ax.set_title(f"Relative L2 error — {field}")
        ax.set_xlabel("t")
        ax.set_ylabel("relative L2 error")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("OpenFOAM vs PINN accuracy (Re=200)", fontsize=13)
    fig.tight_layout()
    fname = os.path.join(plots_dir, "error_vs_time.png")
    fig.savefig(fname, dpi=150)
    plt.close()
    print(f"  Saved {fname}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    errors = {}
    pinn_errors = {}

    for t_eval in t_eval_list:
        csv_path = FOAM_FILES[t_eval]
        assert os.path.exists(csv_path), f"Missing: {csv_path}"

        print(f"\nProcessing t={t_eval} from {csv_path}")

        pts_x, pts_y, u_foam_raw, v_foam_raw, p_foam_raw = load_foam_csv(csv_path)

        print(
            f"  Raw data NaNs: u={np.isnan(u_foam_raw).sum()}, "
            f"v={np.isnan(v_foam_raw).sum()}, p={np.isnan(p_foam_raw).sum()}"
        )
        print(
            f"  x range: [{pts_x.min():.4f}, {pts_x.max():.4f}], "
            f"y range: [{pts_y.min():.4f}, {pts_y.max():.4f}]"
        )

        u_foam = interpolate_to_grid(pts_x, pts_y, u_foam_raw, X, Y)
        v_foam = interpolate_to_grid(pts_x, pts_y, v_foam_raw, X, Y)
        p_foam = interpolate_to_grid(pts_x, pts_y, p_foam_raw, X, Y)

        p_foam -= p_foam.mean()
        p_exact_arr = np.array(taylor_green_p(X, Y, jnp.full_like(X, t_eval), nu))
        p_exact_arr -= p_exact_arr.mean()

        print(
            f"  Interpolated NaNs: u={np.isnan(u_foam).sum()}, "
            f"v={np.isnan(v_foam).sum()}, p={np.isnan(p_foam).sum()}"
        )

        T = jnp.full_like(X, t_eval)
        u_exact = taylor_green_u(X, Y, T, nu)
        v_exact = taylor_green_v(X, Y, T, nu)
        p_exact = taylor_green_p(X, Y, T, nu)

        eu = relative_l2_error(jnp.array(u_foam), u_exact)
        ev = relative_l2_error(jnp.array(v_foam), v_exact)
        ep = relative_l2_error(jnp.array(p_foam), p_exact)
        errors[t_eval] = (eu, ev, ep)

        print(f"  OpenFOAM L2 errors: u={eu:.4e}, v={ev:.4e}, p={ep:.4e}")

        x_flat, y_flat = X.ravel(), Y.ravel()
        t_flat = jnp.full_like(x_flat, t_eval)
        u_pinn, v_pinn, p_pinn = _eval_uvp_batch_compat(pinn_model, x_flat, y_flat, t_flat,
                                                         n_inputs=_n_inputs)
        eu_p = relative_l2_error(u_pinn.reshape(X.shape), u_exact)
        ev_p = relative_l2_error(v_pinn.reshape(X.shape), v_exact)
        ep_p = relative_l2_error(p_pinn.reshape(X.shape), p_exact)
        pinn_errors[t_eval] = (eu_p, ev_p, ep_p)

        print(f"  PINN     L2 errors: u={eu_p:.4e}, v={ev_p:.4e}, p={ep_p:.4e}")

        plot_comparison_foam(
            X, Y, u_exact, v_exact, p_exact,
            jnp.array(u_foam), jnp.array(v_foam), jnp.array(p_foam),
            t=t_eval, plots_dir=PLOTS_DIR,
        )

    plot_error_vs_time(t_eval_list, errors, pinn_errors, PLOTS_DIR)

    print("\n" + "=" * 63)
    print("Relative L2 errors — OpenFOAM and PINN vs analytical:")
    print(f"{'t':>6}  {'OF u':>10}  {'OF v':>10}  {'OF p':>10}  {'PN u':>10}  {'PN v':>10}  {'PN p':>10}")
    print("-" * 63)
    for t in t_eval_list:
        eu, ev, ep = errors[t]
        eu_p, ev_p, ep_p = pinn_errors[t]
        print(f"{t:>6.2f}  {eu:>10.4e}  {ev:>10.4e}  {ep:>10.4e}  {eu_p:>10.4e}  {ev_p:>10.4e}  {ep_p:>10.4e}")
