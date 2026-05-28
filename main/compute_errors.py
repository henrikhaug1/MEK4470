"""Compute L2 relative errors across all PINN cases."""
import os
import sys
import numpy as np

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pinn import (
    BackwardStepPINN,
    TaylorGreenPINN,
    FourierBackwardStepPINN,
    load_model_state,
    load_fourier_model,
    eval_uvp_batch_lc,
    eval_uvp_batch_bs_raw,
)
from src.analytical import taylor_green_u, taylor_green_v, taylor_green_p


def _net_tg_physical_legacy(model, z, t_max):
    """Old 3-input TG evaluation: raw (x, y, t_n) without Fourier features."""
    x_, y_, t_ = z[0], z[1], z[2]
    t_n = 2.0 * t_ / t_max - 1.0
    inp = jnp.stack([x_, y_, t_n])
    raw = model(inp[None])[0]
    u_ic = jnp.cos(x_) * jnp.sin(y_)
    v_ic = -jnp.sin(x_) * jnp.cos(y_)
    p_ic = -0.25 * (jnp.cos(2.0 * x_) + jnp.cos(2.0 * y_))
    return jnp.array([u_ic + t_ * raw[0], v_ic + t_ * raw[1], p_ic + t_ * raw[2]])


def eval_uvp_batch_tg(model, x, y, t, t_max=2.0, n_inputs=5):
    """Evaluate TG model, auto-detecting legacy (3-input) vs current (5-input)."""
    from src.taylor_green import _net_tg_physical
    net_fn = _net_tg_physical if n_inputs == 5 else _net_tg_physical_legacy

    def eval_single(xi, yi, ti):
        out = net_fn(model, jnp.stack([xi, yi, ti]), t_max)
        return out[0], out[1], out[2]

    return jax.vmap(eval_single)(x, y, t)


def l2_rel(pred, ref):
    num = np.sqrt(np.nanmean((pred - ref) ** 2))
    den = np.sqrt(np.nanmean(ref ** 2))
    if den < 1e-14:
        return float("inf")
    return num / den


# ================================================================
# Taylor-Green vortex
# ================================================================
print("=" * 70)
print("TAYLOR-GREEN VORTEX (PINN vs analytical)")
print("=" * 70)

tg_results = []
for Re in [200, 300, 1000]:
    params_path = f"results/taylor_green/Re{Re}/params.npz"
    if not os.path.exists(params_path):
        print(f"  Re={Re}: params not found, skipping")
        continue

    data = np.load(params_path)
    if "_n" in data:
        n_in = int(data.get("_n_inputs", 5))
    else:
        n_in = int(data["W0"].shape[0])

    model = TaylorGreenPINN(widths=[64, 64, 64], key=jax.random.key(0),
                             activation=jax.nn.tanh, n_inputs=n_in)
    if "_n" in data:
        load_model_state(model, params_path)
    else:
        n = sum(1 for k in data if k.startswith("W"))
        for i in range(n):
            model.ws[i].value = jnp.array(data[f"W{i}"])
            model.bs[i].value = jnp.array(data[f"b{i}"])
    nu = 1.0 / Re

    N = 100
    x = jnp.linspace(0, 2 * jnp.pi, N)
    y = jnp.linspace(0, 2 * jnp.pi, N)
    X, Y = jnp.meshgrid(x, y)

    for t_eval in [0.25, 0.5, 1.0, 2.0]:
        T = jnp.full_like(X, t_eval)
        u_exact = np.array(taylor_green_u(X, Y, T, nu))
        v_exact = np.array(taylor_green_v(X, Y, T, nu))
        p_exact = np.array(taylor_green_p(X, Y, T, nu))

        u_pred, v_pred, p_pred = eval_uvp_batch_tg(model, X.ravel(), Y.ravel(), T.ravel(),
                                                     t_max=2.0, n_inputs=n_in)
        u_pred = np.array(u_pred.reshape(N, N))
        v_pred = np.array(v_pred.reshape(N, N))
        p_pred = np.array(p_pred.reshape(N, N))

        eu = l2_rel(u_pred, u_exact)
        ev = l2_rel(v_pred, v_exact)
        ep = l2_rel(p_pred, p_exact)

        tg_results.append((Re, t_eval, eu, ev, ep))
        print(f"  Re={Re:4d}  t={t_eval:.2f}  L2_rel: u={eu:.4e}  v={ev:.4e}  p={ep:.4e}")

print()

# ================================================================
# Lid-driven cavity
# ================================================================
print("=" * 70)
print("LID-DRIVEN CAVITY (PINN vs OpenFOAM)")
print("=" * 70)

# OpenFOAM reader for lid cavity
OF_LC_CASES = {
    100: "openfoam/run/lid_cav_re100",
    500: "openfoam/run/lid_cav_re500",
    800: "openfoam/run/lid_cav_re800",
}
OF_LC_TIME = "100"


def read_of_lc(case_dir, time_dir=OF_LC_TIME):
    from scipy.interpolate import griddata

    # Read U
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
    Nc = int(round(n_cells ** 0.5))
    x1d_of = -1.0 + (np.arange(Nc) + 0.5) * 2.0 / Nc
    y1d_of = -1.0 + (np.arange(Nc) + 0.5) * 2.0 / Nc
    xi = np.tile(x1d_of, Nc)
    yi = np.repeat(y1d_of, Nc)

    # Read p
    p_path = os.path.join(case_dir, time_dir, "p")
    p_vals = []
    reading = False
    with open(p_path) as fh:
        for line in fh:
            s = line.strip()
            if not reading and s.isdigit() and int(s) > 10:
                reading = True
                continue
            if reading and s == "(":
                continue
            if reading and s.startswith(")"):
                break
            if reading and s:
                p_vals.append(float(s))
    p_of = np.array(p_vals)

    N = 150
    x1d = np.linspace(-1.0, 1.0, N)
    y1d = np.linspace(-1.0, 1.0, N)
    X, Y = np.meshgrid(x1d, y1d)
    query = np.column_stack([X.ravel(), Y.ravel()])
    pts = np.column_stack([xi, yi])

    u_g = griddata(pts, uvw[:, 0], query, method="linear").reshape(N, N)
    v_g = griddata(pts, uvw[:, 1], query, method="linear").reshape(N, N)
    p_g = griddata(pts, p_of, query, method="linear").reshape(N, N)
    p_g -= np.nanmean(p_g)

    return X, Y, u_g, v_g, p_g


lc_results = []
for Re in [100, 500, 800]:
    params_path = f"results/lid_cavity/Re{Re}/params.npz"
    of_dir = OF_LC_CASES.get(Re)

    if not os.path.exists(params_path):
        print(f"  Re={Re}: params not found, skipping")
        continue
    if of_dir is None or not os.path.isdir(of_dir):
        print(f"  Re={Re}: OpenFOAM dir not found, skipping")
        continue

    model = BackwardStepPINN(
        widths=[128, 128, 128, 128], key=jax.random.key(0),
        activation=jax.nn.tanh, n_inputs=2,
    )
    load_model_state(model, params_path)

    X, Y, u_of, v_of, p_of = read_of_lc(of_dir)
    N = X.shape[0]

    u_p, v_p, p_p = eval_uvp_batch_lc(model, jnp.array(X.ravel()), jnp.array(Y.ravel()))
    u_p = np.array(u_p.reshape(N, N))
    v_p = np.array(v_p.reshape(N, N))
    p_p = np.array(p_p.reshape(N, N))
    p_p -= np.nanmean(p_p)

    eu = l2_rel(u_p, u_of)
    ev = l2_rel(v_p, v_of)
    ep = l2_rel(p_p, p_of)

    lc_results.append((Re, eu, ev, ep))
    print(f"  Re={Re:4d}  L2_rel: u={eu:.4e}  v={ev:.4e}  p={ep:.4e}")

print()

# ================================================================
# Backward-facing step
# ================================================================
print("=" * 70)
print("BACKWARD-FACING STEP (PINN vs OpenFOAM)")
print("=" * 70)

X_MIN = -2.0
H_STEP = 1.0
H_CHAN = 2.0

OF_BS_CASES = {
    100: "openfoam/run/backwardStep_parabolic_re100",
    200: "openfoam/run/backwardStep_parabolic_re200",
    400: "openfoam/run/backwardStep_parabolic_re400",
}
OF_BS_TIME = "20"
RE_X_MAX = {100: 20.0, 200: 20.0, 400: 30.0}

_OF_H_M = 0.1
_OF_LIN_M = 0.2
_OF_H2_M = 0.2
_OF_NY = 20
_OF_NX1 = 20


def of_bs_cell_centers(case_dir):
    pts_file = os.path.join(case_dir, "constant", "polyMesh", "points")
    x_max_pts = 0.0
    reading = False
    with open(pts_file) as fh:
        for line in fh:
            s = line.strip()
            if not reading and s.isdigit() and int(s) > 100:
                reading = True
                continue
            if reading and s == "(":
                continue
            if reading and s.startswith(")"):
                break
            if reading and s:
                vals = s.strip("()").split()
                if vals:
                    x_max_pts = max(x_max_pts, float(vals[0]))
    Ltot_m = x_max_pts

    u_file = os.path.join(case_dir, OF_BS_TIME, "U")
    n_cells = None
    with open(u_file) as fh:
        for line in fh:
            s = line.strip()
            if s.isdigit() and int(s) > 100:
                n_cells = int(s)
                break

    H, Lin, H2 = _OF_H_M, _OF_LIN_M, _OF_H2_M
    nx_exp = (n_cells - _OF_NX1 * _OF_NY) // (2 * _OF_NY)

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


def parse_of_vector(filepath, n_cells):
    values = []
    reading = False
    with open(filepath) as fh:
        for line in fh:
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
    return np.array(values)


def parse_of_scalar(filepath, n_cells):
    values = []
    reading = False
    with open(filepath) as fh:
        for line in fh:
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
    return np.array(values)


bs_results = []
for Re in [100, 200, 400]:
    params_path = f"results/backward_step_v2/Re{Re}/params.npz"
    of_dir = OF_BS_CASES.get(Re)
    x_max = RE_X_MAX[Re]

    if not os.path.exists(params_path):
        print(f"  Re={Re}: params not found, skipping")
        continue
    if of_dir is None or not os.path.isdir(of_dir):
        print(f"  Re={Re}: OpenFOAM dir not found, skipping")
        continue

    data = np.load(params_path)
    if "_n_fourier" in data:
        model = load_fourier_model(params_path, key=jax.random.key(0), activation=jax.nn.tanh)
    else:
        widths = [int(w) for w in data["_widths"]]
        n_inputs = int(data["_n_inputs"]) if "_n_inputs" in data else 2
        model = BackwardStepPINN(
            widths=widths, key=jax.random.key(0),
            activation=jax.nn.tanh, n_inputs=n_inputs,
        )
        load_model_state(model, params_path)

    from scipy.interpolate import griddata

    Nx, Ny = 400, 80
    x1d = np.linspace(X_MIN, x_max, Nx)
    y1d = np.linspace(0.0, H_CHAN, Ny)
    Xg, Yg = np.meshgrid(x1d, y1d)
    fluid = ~((Xg < 0.0) & (Yg < H_STEP))

    u_p, v_p, p_p = eval_uvp_batch_bs_raw(
        model, jnp.array(Xg.ravel()), jnp.array(Yg.ravel()),
        X_MIN, x_max, H_CHAN,
    )
    u_p = np.where(fluid, np.array(u_p).reshape(Ny, Nx), np.nan)
    v_p = np.where(fluid, np.array(v_p).reshape(Ny, Nx), np.nan)
    p_p = np.where(fluid, np.array(p_p).reshape(Ny, Nx), np.nan)
    p_p = p_p - np.nanmean(p_p)

    x_of, y_of, n_cells = of_bs_cell_centers(of_dir)
    uvw = parse_of_vector(os.path.join(of_dir, OF_BS_TIME, "U"), n_cells)
    p_raw = parse_of_scalar(os.path.join(of_dir, OF_BS_TIME, "p"), n_cells)

    of_pts = np.column_stack([x_of, y_of])
    query = np.column_stack([Xg.ravel(), Yg.ravel()])

    u_of = griddata(of_pts, uvw[:, 0], query, method="linear").reshape(Ny, Nx)
    v_of = griddata(of_pts, uvw[:, 1], query, method="linear").reshape(Ny, Nx)
    p_of = griddata(of_pts, p_raw, query, method="linear").reshape(Ny, Nx)

    u_of = np.where(fluid, u_of, np.nan)
    v_of = np.where(fluid, v_of, np.nan)
    p_of = np.where(fluid, p_of, np.nan)
    p_of = p_of - np.nanmean(p_of)

    eu = l2_rel(u_p, u_of)
    ev = l2_rel(v_p, v_of)
    ep = l2_rel(p_p, p_of)

    bs_results.append((Re, eu, ev, ep))
    print(f"  Re={Re:4d}  L2_rel: u={eu:.4e}  v={ev:.4e}  p={ep:.4e}")

print()

# ================================================================
# Summary table
# ================================================================
print("=" * 70)
print("SUMMARY TABLE — L2 relative errors")
print("=" * 70)
print()
print(f"{'Case':<35s} {'ε_u':>10s} {'ε_v':>10s} {'ε_p':>10s}")
print("-" * 70)

for Re, t, eu, ev, ep in tg_results:
    print(f"  Taylor-Green  Re={Re:4d}  t={t:.2f}    {eu:10.2e} {ev:10.2e} {ep:10.2e}")

print("-" * 70)
for Re, eu, ev, ep in lc_results:
    print(f"  Lid cavity    Re={Re:4d}            {eu:10.2e} {ev:10.2e} {ep:10.2e}")

print("-" * 70)
for Re, eu, ev, ep in bs_results:
    print(f"  Backward step Re={Re:4d}            {eu:10.2e} {ev:10.2e} {ep:10.2e}")

print("-" * 70)
