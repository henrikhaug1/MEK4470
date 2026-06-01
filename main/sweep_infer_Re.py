"""
sweep_infer_Re.py
-----------------
Sweep Re_true × Re_init combinations to characterise inference accuracy.
Observations come from OpenFOAM (independent reference solver).

For each (Re_true, Re_init) pair:
  - load the OpenFOAM solution as the observation source
  - run the joint physics+data optimisation to infer Re
  - record Re_final and the relative error

Produces:
  figures/infer_Re_sweep.pdf/png   — % error vs Re_init/Re_true, one curve per Re_true
  results/infer_Re_sweep/sweep_results.csv  — full table
"""

import os
import sys
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pinn import (
    BackwardStepPINN,
    eval_uvp_batch_bs_raw,
    residuals_batch_bs_raw,
    inlet_profile_bs,
    sample_interior_bs,
    sample_walls_bs,
    sample_inlet_bs,
)

# ═══════════════════════════════════════════════════════════════
# Sweep configuration
# ═══════════════════════════════════════════════════════════════

RE_TRUE_VALUES   = [100, 200, 400]
RE_INIT_FACTORS  = [1/8, 1/4, 1/2, 2, 4, 8]   # Re_init = Re_true × factor

EPOCHS    = 5000
N_OBS     = 500
NET_LR    = 1e-3
RE_LR     = 1e-2

N_COL   = 12000
N_INLET = 500
N_WALL  = 2000

# per-Re domain length (Re=400 needs longer domain for full recirculation)
RE_X_MAX = {100: 20.0, 200: 20.0, 400: 30.0}

RESAMPLE_EVERY   = 1000
PATIENCE_WINDOW  = 500
RE_STABLE_WINDOW = 200
RE_STABLE_TOL    = 0.005
WARMUP_EPOCHS    = 1000

# geometry (must match training)
X_MIN  = -2.0
X_MAX  = 20.0
H_STEP = 1.0
H_CHAN = 2.0
U_MEAN = 1.0

WIDTHS = [64, 64, 64]

W_PHYS  = 5.0
W_DATA  = 500.0
W_INLET = 2.0
W_WALLS = 5.0

OUT_DIR     = "figures"
RESULTS_DIR = "results/infer_Re_sweep"

# ═══════════════════════════════════════════════════════════════
# OpenFOAM case paths + reader helpers
# ═══════════════════════════════════════════════════════════════

OF_CASES = {
    100: "openfoam/run/backwardStep_parabolic_re100",
    200: "openfoam/run/backwardStep_parabolic_re200",
    400: "openfoam/run/backwardStep_parabolic_re400",
}
OF_TIME = "20"

_OF_H_M   = 0.1
_OF_LIN_M = 0.2
_OF_H2_M  = 0.2
_OF_NY    = 20
_OF_NX1   = 20


def _of_detect_mesh(case_dir):
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
    u_file = os.path.join(case_dir, OF_TIME, "U")
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


def _of_cell_centers(case_dir):
    n_cells, nx_exp, Ltot_m = _of_detect_mesh(case_dir)
    H, Lin, H2 = _OF_H_M, _OF_LIN_M, _OF_H2_M

    def _block_centres(x0, x1, nx, y0, y1, ny):
        xc = x0 + (x1 - x0) * (np.arange(nx) + 0.5) / nx
        yc = y0 + (y1 - y0) * (np.arange(ny) + 0.5) / ny
        Xi, Yi = np.meshgrid(xc, yc, indexing="xy")
        return Xi.ravel(), Yi.ravel()

    xb1, yb1 = _block_centres(0, Lin,     _OF_NX1, H, H2, _OF_NY)
    xb2, yb2 = _block_centres(Lin, Ltot_m, nx_exp, H, H2, _OF_NY)
    xb3, yb3 = _block_centres(Lin, Ltot_m, nx_exp, 0, H,  _OF_NY)

    x_nd = np.concatenate([xb1, xb2, xb3]) / H - Lin / H
    y_nd = np.concatenate([yb1, yb2, yb3]) / H
    return x_nd, y_nd, n_cells


def _parse_of_vector_field(filepath, n_cells):
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
    arr = np.array(values)
    if len(arr) != n_cells:
        raise RuntimeError(f"Expected {n_cells} vectors, got {len(arr)}")
    return arr


def load_of_observations(Re_true, x_obs, y_obs):
    """Interpolate OpenFOAM u- and v-velocity onto (x_obs, y_obs) observation points."""
    of_dir = OF_CASES[int(Re_true)]
    x_nd, y_nd, n_cells = _of_cell_centers(of_dir)
    uvw = _parse_of_vector_field(os.path.join(of_dir, OF_TIME, "U"), n_cells)
    u_of = uvw[:, 0]
    v_of = uvw[:, 1]

    pts   = np.column_stack([x_nd, y_nd])
    query = np.column_stack([np.array(x_obs), np.array(y_obs)])

    def _interp(field):
        vals = griddata(pts, field, query, method="linear")
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            vals[nan_mask] = griddata(pts, field, query[nan_mask], method="nearest")
        return jnp.array(vals)

    return _interp(u_of), _interp(v_of)


# ═══════════════════════════════════════════════════════════════
# Loss
# ═══════════════════════════════════════════════════════════════

def make_warmup_loss_fn(x_in, y_in, x_wall, y_wall, x_obs, y_obs, u_obs, v_obs):
    """Data + BCs only — used during warm-up (Re frozen)."""
    u_inlet_ref = inlet_profile_bs(y_in, H_STEP, H_CHAN, U_MEAN)

    def loss_fn(model):
        u_pred, v_pred, _ = eval_uvp_batch_bs_raw(model, x_obs, y_obs, X_MIN, X_MAX, H_CHAN)
        data = jnp.mean((u_pred - u_obs) ** 2) + jnp.mean((v_pred - v_obs) ** 2)

        u_i, v_i, _ = eval_uvp_batch_bs_raw(model, x_in, y_in, X_MIN, X_MAX, H_CHAN)
        bc_inlet = jnp.mean((u_i - u_inlet_ref) ** 2) + jnp.mean(v_i**2)

        u_w, v_w, _ = eval_uvp_batch_bs_raw(model, x_wall, y_wall, X_MIN, X_MAX, H_CHAN)
        bc_walls = jnp.mean(u_w**2 + v_w**2)

        return W_DATA * data + W_INLET * bc_inlet + W_WALLS * bc_walls

    return loss_fn


def make_loss_fn(x_col, y_col, x_in, y_in, x_wall, y_wall, x_obs, y_obs, u_obs, v_obs):
    """Full physics + data + BCs — used in joint phase."""
    u_inlet_ref = inlet_profile_bs(y_in, H_STEP, H_CHAN, U_MEAN)

    def loss_fn(model, log_Re):
        Re = jnp.exp(log_Re)
        cont, mom_x, mom_y = residuals_batch_bs_raw(
            model, Re, x_col, y_col, X_MIN, X_MAX, H_CHAN
        )
        phys = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        u_pred, v_pred, _ = eval_uvp_batch_bs_raw(model, x_obs, y_obs, X_MIN, X_MAX, H_CHAN)
        data = jnp.mean((u_pred - u_obs) ** 2) + jnp.mean((v_pred - v_obs) ** 2)

        u_i, v_i, _ = eval_uvp_batch_bs_raw(model, x_in, y_in, X_MIN, X_MAX, H_CHAN)
        bc_inlet = jnp.mean((u_i - u_inlet_ref) ** 2) + jnp.mean(v_i**2)

        u_w, v_w, _ = eval_uvp_batch_bs_raw(model, x_wall, y_wall, X_MIN, X_MAX, H_CHAN)
        bc_walls = jnp.mean(u_w**2 + v_w**2)

        return W_PHYS * phys + W_DATA * data + W_INLET * bc_inlet + W_WALLS * bc_walls

    return loss_fn


# ═══════════════════════════════════════════════════════════════
# Single inference run
# ═══════════════════════════════════════════════════════════════

def run_inference(Re_true, Re_init, x_obs, y_obs, u_obs, v_obs, seed=42, desc="", x_max=X_MAX):
    """Run one (Re_true, Re_init) inference. Returns Re_final and % error."""
    key = jax.random.key(seed)
    model = BackwardStepPINN(
        widths=WIDTHS, key=key, activation=jax.nn.tanh, n_inputs=2
    )
    log_Re = jnp.array(float(np.log(Re_init)))

    k1, k_in, k_wall = jax.random.split(key, 3)
    x_col, y_col = sample_interior_bs(k1, N_COL,    X_MIN, x_max, H_STEP, H_CHAN)
    x_in,  y_in  = sample_inlet_bs  (k_in, N_INLET, X_MIN, H_STEP, H_CHAN)
    x_wall, y_wall = sample_walls_bs(k_wall, N_WALL, X_MIN, x_max, H_STEP, H_CHAN)

    warmup_loss = make_warmup_loss_fn(x_in, y_in, x_wall, y_wall, x_obs, y_obs, u_obs, v_obs)

    # ── Phase 1: warm-up (data + BCs, Re frozen) ─────────────────────────
    wu_sched = optax.cosine_decay_schedule(NET_LR, WARMUP_EPOCHS, alpha=1e-2)
    wu_opt   = nnx.Optimizer(model, optax.adam(wu_sched), wrt=nnx.Param)

    @nnx.jit
    def wu_step(model, opt):
        loss, grads = nnx.value_and_grad(warmup_loss)(model)
        opt.update(model, grads)
        return loss

    with tqdm(total=WARMUP_EPOCHS, desc=f"{desc} [wu]", unit="ep",
              dynamic_ncols=True, leave=False) as pbar:
        for _ in range(WARMUP_EPOCHS):
            wu_step(model, wu_opt)
            pbar.update(1)

    # ── Phase 2: joint optimisation (physics + data + BCs) ───────────────
    loss_fn = make_loss_fn(x_col, y_col, x_in, y_in, x_wall, y_wall,
                           x_obs, y_obs, u_obs, v_obs)

    net_schedule = optax.cosine_decay_schedule(NET_LR, EPOCHS, alpha=1e-2)
    re_schedule  = optax.cosine_decay_schedule(RE_LR,  EPOCHS, alpha=1e-2)
    net_opt  = nnx.Optimizer(model, optax.adam(net_schedule), wrt=nnx.Param)
    re_opt   = optax.adam(re_schedule)
    re_state = re_opt.init(log_Re)

    @nnx.jit
    def step(model, net_opt, log_Re, re_state):
        loss, net_grads = nnx.value_and_grad(lambda m: loss_fn(m, log_Re))(model)
        re_grad = jax.grad(lambda lr: loss_fn(model, lr))(log_Re)
        net_opt.update(model, net_grads)
        re_upd, new_re_state = re_opt.update(re_grad, re_state)
        return optax.apply_updates(log_Re, re_upd), new_re_state, loss

    k_col        = k1
    loss_history = []
    Re_history   = []

    with tqdm(total=EPOCHS, desc=desc, unit="ep", dynamic_ncols=True, leave=False) as pbar:
        for epoch in range(EPOCHS):
            log_Re, re_state, loss = step(model, net_opt, log_Re, re_state)
            Re_curr = float(jnp.exp(log_Re))
            loss_history.append(float(loss))
            Re_history.append(Re_curr)

            if epoch % 20 == 0:
                pbar.set_postfix(Re=f"{Re_curr:.1f}", loss=f"{float(loss):.2e}")
            pbar.update(1)

            # Re-stability early stopping — but not before the collocation
            # set has been resampled at least twice, so runs that settle on a
            # pre-resample plateau get a chance to be jolted off it.
            if epoch >= max(RE_STABLE_WINDOW, 2 * RESAMPLE_EVERY) and epoch % 20 == 0:
                window = Re_history[-RE_STABLE_WINDOW:]
                rel_std = np.std(window) / (np.mean(window) + 1e-12)
                if rel_std < RE_STABLE_TOL:
                    pbar.write(f"    Re converged at epoch {epoch} (Re={Re_curr:.2f})")
                    break

            # resample collocation points
            if epoch > 0 and epoch % RESAMPLE_EVERY == 0:
                k_col = jax.random.fold_in(k_col, epoch)
                x_col, y_col = sample_interior_bs(k_col, N_COL, X_MIN, x_max, H_STEP, H_CHAN)
                loss_fn = make_loss_fn(x_col, y_col, x_in, y_in, x_wall, y_wall,
                                       x_obs, y_obs, u_obs, v_obs)

            # loss-plateau early stopping
            if epoch > EPOCHS // 2 and epoch % 100 == 0 and len(loss_history) >= 2 * PATIENCE_WINDOW:
                recent = np.mean(loss_history[-PATIENCE_WINDOW:])
                older  = np.mean(loss_history[-2 * PATIENCE_WINDOW:-PATIENCE_WINDOW])
                if abs(recent - older) / (abs(older) + 1e-12) < 5e-4:
                    pbar.write(f"    early stop at epoch {epoch}")
                    break

    Re_final = float(jnp.exp(log_Re))
    err_pct  = abs(Re_final - Re_true) / Re_true * 100.0
    return Re_final, err_pct


# ═══════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    for Re_true in RE_TRUE_VALUES:
        of_dir = OF_CASES.get(Re_true)
        if of_dir is None or not os.path.isdir(of_dir):
            print(f"  Re={Re_true}: OpenFOAM case not found at {of_dir}, skipping")
            continue

        print(f"\n{'═'*55}")
        print(f"  Re_true = {Re_true}  —  loading OpenFOAM observations")
        print(f"{'═'*55}")

        x_max = RE_X_MAX.get(Re_true, X_MAX)

        # Fixed observation locations — 70% biased near step, 30% elsewhere
        key_obs = jax.random.key(99)
        k_near, k_far = jax.random.split(key_obs)
        N_near = int(N_OBS * 0.7) * 6
        N_far  = int(N_OBS * 0.3) * 6

        pts_near = jax.random.uniform(k_near, (N_near, 2))
        x_near = pts_near[:, 0] * 8.0          # x in [0, 8] — recirculation region
        y_near = pts_near[:, 1] * H_CHAN
        valid_near = ~((x_near < 0.0) & (y_near < H_STEP))
        x_near = x_near[valid_near][:int(N_OBS * 0.7)]
        y_near = y_near[valid_near][:int(N_OBS * 0.7)]

        pts_far = jax.random.uniform(k_far, (N_far, 2))
        x_far = pts_far[:, 0] * (x_max - 8.0) + 8.0   # x in [8, x_max]
        y_far = pts_far[:, 1] * H_CHAN
        x_obs = jnp.concatenate([x_near, x_far[:int(N_OBS * 0.3)]])
        y_obs = jnp.concatenate([y_near, y_far[:int(N_OBS * 0.3)]])

        u_obs, v_obs = load_of_observations(Re_true, x_obs, y_obs)
        print(f"  {N_OBS} obs points: u range [{float(u_obs.min()):.3f}, {float(u_obs.max()):.3f}]")

        for factor in RE_INIT_FACTORS:
            Re_init = Re_true * factor
            if Re_init < 5:
                continue
            desc = f"Re_true={Re_true:4d}  Re_init={Re_init:7.1f}"
            t0 = time.perf_counter()
            Re_final, err_pct = run_inference(
                Re_true, Re_init, x_obs, y_obs, u_obs, v_obs, desc=desc, x_max=x_max
            )
            elapsed = time.perf_counter() - t0
            print(f"  {desc}  →  Re_final={Re_final:7.2f}   err={err_pct:6.2f}%   ({elapsed:.0f}s)")

            results.append(dict(
                Re_true=Re_true,
                Re_init=Re_init,
                factor=factor,
                Re_final=Re_final,
                err_pct=err_pct,
                log_ratio=float(np.log10(factor)),
                elapsed_s=elapsed,
            ))

    if not results:
        print("No results — check that OpenFOAM case directories exist.")
        return

    # ── save CSV ────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "sweep_results.csv")
    header = "Re_true,Re_init,factor,Re_final,err_pct,log_ratio,elapsed_s"
    rows = [f"{r['Re_true']},{r['Re_init']:.1f},{r['factor']:.4f},"
            f"{r['Re_final']:.4f},{r['err_pct']:.4f},{r['log_ratio']:.4f},{r['elapsed_s']:.1f}"
            for r in results]
    with open(csv_path, "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"\nResults saved → {csv_path}")

    # ── Plot 1: % error vs Re_init/Re_true ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = plt.cm.tab10.colors
    for i, Re_true in enumerate(RE_TRUE_VALUES):
        sub = [r for r in results if r["Re_true"] == Re_true]
        if not sub:
            continue
        factors  = [r["factor"]  for r in sub]
        err_pcts = [r["err_pct"] for r in sub]
        axes[0].plot(factors, err_pcts, "o-", color=colors[i],
                     lw=1.8, ms=6, label=f"Re = {Re_true}")

    axes[0].axhline(10, color="0.6", lw=0.8, ls="--", label="10 % threshold")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\mathrm{Re}_{\mathrm{init}}\,/\,\mathrm{Re}_{\mathrm{true}}$", fontsize=11)
    axes[0].set_ylabel("Relative error  (%)", fontsize=11)
    axes[0].set_title("Inference error vs initialisation ratio", fontsize=11)
    axes[0].axvline(1, color="0.7", lw=0.7, ls=":")
    axes[0].legend(fontsize=7, loc="upper right")

    # ── Plot 2: Re_final vs Re_true scatter ──────────────────────
    all_Re_true  = [r["Re_true"]  for r in results]
    all_Re_final = [r["Re_final"] for r in results]
    all_factors  = [r["factor"]   for r in results]

    sc = axes[1].scatter(
        all_Re_true, all_Re_final,
        c=np.log10(all_factors),
        cmap="RdYlGn_r",
        s=60, zorder=3, edgecolors="0.3", linewidths=0.5,
    )
    cb = fig.colorbar(sc, ax=axes[1], pad=0.02)
    cb.set_label(
        r"$\log_{10}(\mathrm{Re}_{\mathrm{init}}\,/\,\mathrm{Re}_{\mathrm{true}})$",
        fontsize=9, rotation=270, labelpad=14, va="bottom",
    )

    lo = min(all_Re_true) * 0.8
    hi = max(all_Re_true) * 1.2
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    axes[1].set_xlabel(r"$\mathrm{Re}_{\mathrm{true}}$", fontsize=11)
    axes[1].set_ylabel(r"$\mathrm{Re}_{\mathrm{final}}$", fontsize=11)
    axes[1].set_title("Inferred vs true Re  (obs: OpenFOAM)", fontsize=11)
    axes[1].legend(fontsize=7, loc="upper left")

    fig.tight_layout(pad=1.5)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"infer_Re_sweep.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    main()
