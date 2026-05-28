import os
import time
import argparse

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.interpolate import griddata
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")

from src.pinn import (
    BackwardStepPINN,
    eval_uvp_batch_bs_raw,
    residuals_batch_bs_raw,
    inlet_profile_bs,
    sample_interior_bs,
    sample_walls_bs,
    sample_inlet_bs,
)
from src.plotting import (
    plot_Re_convergence,
    plot_observation_points,
    plot_inference_loss,
)

jax.config.update("jax_enable_x64", True)

# ============================================================
# geometry  (must match train_backward_step_v2.py)
# ============================================================

X_MIN  = -2.0
X_MAX  = 20.0
H_STEP = 1.0
H_CHAN = 2.0
U_MEAN = 1.0

WIDTHS = [32, 32, 32]  # inference model (plain tanh, trained from scratch)

# ============================================================
# OpenFOAM case paths
# ============================================================

OF_CASES = {
    100: "openfoam/run/backwardStep_parabolic_re100",
    200: "openfoam/run/backwardStep_parabolic_re200",
    400: "openfoam/run/backwardStep_parabolic_re400",
}
OF_TIME = "20"

# Physical mesh constants (same for all backwardStep_parabolic cases)
_OF_H_M   = 0.1   # step height in metres
_OF_LIN_M = 0.2   # inlet length in metres
_OF_H2_M  = 0.2   # channel half-width above step in metres
_OF_NY    = 20
_OF_NX1   = 20


# ============================================================
# OpenFOAM reader helpers
# ============================================================

def _of_detect_mesh(case_dir):
    """Return (n_cells, nx_exp, Ltot_m) from the polyMesh points file."""
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
    """Return cell-centre coordinates in PINN non-dimensional units (x, y)."""
    n_cells, nx_exp, Ltot_m = _of_detect_mesh(case_dir)
    H, Lin = _OF_H_M, _OF_LIN_M
    H2 = _OF_H2_M

    def _block_centres(x0, x1, nx, y0, y1, ny):
        xc = x0 + (x1 - x0) * (np.arange(nx) + 0.5) / nx
        yc = y0 + (y1 - y0) * (np.arange(ny) + 0.5) / ny
        Xi, Yi = np.meshgrid(xc, yc, indexing="xy")
        return Xi.ravel(), Yi.ravel()

    xb1, yb1 = _block_centres(0, Lin,    _OF_NX1, H,  H2, _OF_NY)
    xb2, yb2 = _block_centres(Lin, Ltot_m, nx_exp, H,  H2, _OF_NY)
    xb3, yb3 = _block_centres(Lin, Ltot_m, nx_exp, 0,  H,  _OF_NY)

    x_phys = np.concatenate([xb1, xb2, xb3])
    y_phys = np.concatenate([yb1, yb2, yb3])

    # Non-dimensionalise: step height H is the length scale,
    # step face is at x=0 in the PINN system.
    x_nd = x_phys / H - Lin / H
    y_nd = y_phys / H
    return x_nd, y_nd, n_cells


def _parse_of_vector_field(filepath, n_cells):
    """Parse an OpenFOAM vector field file (e.g. U)."""
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
        raise RuntimeError(
            f"Expected {n_cells} vectors, got {len(arr)} in {filepath}"
        )
    return arr


def load_of_observations(Re_true, x_obs, y_obs):
    """
    Interpolate the OpenFOAM u- and v-velocity onto the requested observation
    locations (x_obs, y_obs) in PINN non-dimensional coordinates.

    Returns (u_obs, v_obs) as JAX arrays.
    """
    of_dir = OF_CASES.get(int(Re_true))
    if of_dir is None:
        raise ValueError(f"No OpenFOAM case configured for Re={Re_true}")
    if not os.path.isdir(of_dir):
        raise FileNotFoundError(
            f"OpenFOAM case directory not found: {of_dir}\n"
            f"Expected at: {os.path.abspath(of_dir)}"
        )

    x_nd, y_nd, n_cells = _of_cell_centers(of_dir)
    u_path = os.path.join(of_dir, OF_TIME, "U")
    uvw = _parse_of_vector_field(u_path, n_cells)
    u_of = uvw[:, 0]   # streamwise component
    v_of = uvw[:, 1]   # cross-stream component

    # Interpolate OF cell-centred values onto observation points
    pts = np.column_stack([x_nd, y_nd])
    query = np.column_stack([np.array(x_obs), np.array(y_obs)])

    def _interp(field):
        vals = griddata(pts, field, query, method="linear")
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            vals[nan_mask] = griddata(pts, field, query[nan_mask], method="nearest")
        return jnp.array(vals)

    u_interp = _interp(u_of)
    v_interp = _interp(v_of)

    n_nan = int(np.isnan(griddata(pts, u_of, query, method="linear")).sum())
    if n_nan:
        print(f"  Note: {n_nan} obs points near domain edge filled by nearest-neighbour")

    return u_interp, v_interp


# ============================================================
# loss  (soft-BC formulation matching forward training)
# ============================================================

W_PHYS  = 1.0
W_DATA  = 1.0   # scaled by w_data argument (default 100)
W_INLET = 2.0
W_WALLS = 5.0

# Number of warm-up epochs: train network on data+BCs only, Re frozen.
# This encodes the observed flow into the network BEFORE optimising Re,
# so that the physics gradient on Re is informative from the start of
# the joint phase.
WARMUP_EPOCHS = 1000


def make_warmup_loss_fn(x_in, y_in, x_wall, y_wall, x_obs, y_obs, u_obs, v_obs, w_data):
    """Loss for the warm-up phase: data + BCs, no physics, Re not involved."""
    u_inlet_ref = inlet_profile_bs(y_in, H_STEP, H_CHAN, U_MEAN)

    def loss_fn(model):
        u_pred, v_pred, _ = eval_uvp_batch_bs_raw(model, x_obs, y_obs, X_MIN, X_MAX, H_CHAN)
        data = jnp.mean((u_pred - u_obs) ** 2) + jnp.mean((v_pred - v_obs) ** 2)

        u_i, v_i, _ = eval_uvp_batch_bs_raw(model, x_in, y_in, X_MIN, X_MAX, H_CHAN)
        bc_inlet = jnp.mean((u_i - u_inlet_ref) ** 2) + jnp.mean(v_i**2)

        u_w, v_w, _ = eval_uvp_batch_bs_raw(model, x_wall, y_wall, X_MIN, X_MAX, H_CHAN)
        bc_walls = jnp.mean(u_w**2 + v_w**2)

        return w_data * W_DATA * data + W_INLET * bc_inlet + W_WALLS * bc_walls

    return loss_fn


def make_loss_fn(x_col, y_col, x_in, y_in, x_wall, y_wall,
                 x_obs, y_obs, u_obs, v_obs, w_data):
    """Full loss for the joint phase: physics + data + BCs."""
    u_inlet_ref = inlet_profile_bs(y_in, H_STEP, H_CHAN, U_MEAN)

    def loss_fn(model, log_Re):
        Re = jnp.exp(log_Re)

        cont, mom_x, mom_y = residuals_batch_bs_raw(
            model, Re, x_col, y_col, X_MIN, X_MAX, H_CHAN
        )
        phys = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        # Both u and v observations: doubles the Re-sensitive constraints
        u_pred, v_pred, _ = eval_uvp_batch_bs_raw(model, x_obs, y_obs, X_MIN, X_MAX, H_CHAN)
        data = jnp.mean((u_pred - u_obs) ** 2) + jnp.mean((v_pred - v_obs) ** 2)

        u_i, v_i, _ = eval_uvp_batch_bs_raw(model, x_in, y_in, X_MIN, X_MAX, H_CHAN)
        bc_inlet = jnp.mean((u_i - u_inlet_ref) ** 2) + jnp.mean(v_i**2)

        u_w, v_w, _ = eval_uvp_batch_bs_raw(model, x_wall, y_wall, X_MIN, X_MAX, H_CHAN)
        bc_walls = jnp.mean(u_w**2 + v_w**2)

        return (
            W_PHYS * phys
            + w_data * W_DATA * data
            + W_INLET * bc_inlet
            + W_WALLS * bc_walls
        )

    return loss_fn


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Re_true",
        type=float,
        default=200.0,
        help="True Re whose OpenFOAM solution is used as observations",
    )
    parser.add_argument(
        "--Re_init",
        type=float,
        default=800.0,
        help="Initial guess for Re (start far from truth)",
    )
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8000,
                        help="Joint-phase epochs (after warm-up)")
    parser.add_argument("--net_lr", type=float, default=1e-3)
    parser.add_argument("--re_lr", type=float, default=1e-2)
    parser.add_argument("--w_data", type=float, default=1000.0)
    args = parser.parse_args()

    RESULTS_DIR = "results/infer_Re"
    PLOTS_DIR = "plots/infer_Re"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- sample observation locations within the inference domain [X_MIN, X_MAX] ----
    key_obs = jax.random.key(99)
    pts = jax.random.uniform(key_obs, (args.n_obs * 6, 2))
    x_cand = pts[:, 0] * (X_MAX - X_MIN) + X_MIN
    y_cand = pts[:, 1] * H_CHAN
    # Exclude the step block (x < 0 AND y < H_STEP)
    valid = ~((x_cand < 0.0) & (y_cand < H_STEP))
    x_obs = x_cand[valid][: args.n_obs]
    y_obs = y_cand[valid][: args.n_obs]

    # ---- load OpenFOAM observations ----
    print(f"\nLoading OpenFOAM Re={args.Re_true} observations from {OF_CASES[int(args.Re_true)]}")
    u_obs, v_obs = load_of_observations(args.Re_true, x_obs, y_obs)
    print(f"  Generated {len(x_obs)} observation points from OpenFOAM Re={args.Re_true}")
    print(f"  u_obs: min={float(u_obs.min()):.3f}  max={float(u_obs.max()):.3f}  "
          f"mean={float(u_obs.mean()):.3f}")
    print(f"  v_obs: min={float(v_obs.min()):.3f}  max={float(v_obs.max()):.3f}  "
          f"mean={float(v_obs.mean()):.3f}")

    plot_observation_points(
        x_obs, y_obs, PLOTS_DIR, x_min=X_MIN, x_max=X_MAX, h_step=H_STEP, h_chan=H_CHAN
    )

    # ---- create inference model (random init) ----
    key = jax.random.key(42)
    model = BackwardStepPINN(
        widths=WIDTHS,
        key=key,
        activation=jax.nn.tanh,
        n_inputs=2,
    )

    log_Re = jnp.array(float(np.log(args.Re_init)))

    print(f"\nRe_true = {args.Re_true:.1f}   Re_init = {args.Re_init:.1f}")
    print(f"  Network: random initialization, widths={WIDTHS}")
    print(
        f"  log_Re init = {float(log_Re):.4f}   (truth = {np.log(args.Re_true):.4f})\n"
    )

    # ---- collocation + BC points ----
    k1, k_in, k_wall = jax.random.split(key, 3)
    x_col, y_col = sample_interior_bs(k1, 12000, X_MIN, X_MAX, H_STEP, H_CHAN)
    x_in, y_in = sample_inlet_bs(k_in, 500, X_MIN, H_STEP, H_CHAN)
    x_wall, y_wall = sample_walls_bs(k_wall, 2000, X_MIN, X_MAX, H_STEP, H_CHAN)
    print(f"Collocation pts: {len(x_col)}   Inlet pts: 500   Wall pts: {len(x_wall)}\n")

    warmup_loss_fn = make_warmup_loss_fn(
        x_in, y_in, x_wall, y_wall,
        x_obs, y_obs, u_obs, v_obs,
        w_data=args.w_data,
    )

    # ---- Phase 1: warm-up ----
    # Train network with data + BCs only (no physics, Re frozen).
    # This encodes the observed flow into the network before Re is optimised.
    # After warm-up, the physics gradient on Re is informative (it sees a
    # realistic flow field) instead of chasing random network noise.
    wu_schedule = optax.cosine_decay_schedule(
        init_value=args.net_lr, decay_steps=WARMUP_EPOCHS, alpha=1e-2
    )
    wu_opt = nnx.Optimizer(model, optax.adam(learning_rate=wu_schedule), wrt=nnx.Param)

    @nnx.jit
    def warmup_step(model, opt):
        loss, grads = nnx.value_and_grad(warmup_loss_fn)(model)
        opt.update(model, grads)
        return loss

    print(f"  net_lr = {args.net_lr:.1e}   re_lr = {args.re_lr:.1e}")
    print(f"  warm-up epochs = {WARMUP_EPOCHS}   joint epochs = {args.epochs}\n")
    t0 = time.perf_counter()

    with tqdm(
        total=WARMUP_EPOCHS, desc="Warm-up (data+BC)", unit="epoch", dynamic_ncols=True
    ) as pbar:
        for epoch in range(WARMUP_EPOCHS):
            wu_loss = warmup_step(model, wu_opt)
            if epoch % 20 == 0:
                pbar.set_postfix(loss=f"{float(wu_loss):.3e}")
            pbar.update(1)

    print(f"  Warm-up done. Data loss ≈ {float(wu_loss):.3e}")
    print(f"  Re still at init = {float(jnp.exp(log_Re)):.1f}\n")

    # ---- Phase 2: joint optimisation ----
    # Now both network weights and Re are updated.
    # Fresh optimiser for the joint phase so the cosine schedule restarts.
    net_schedule = optax.cosine_decay_schedule(
        init_value=args.net_lr, decay_steps=args.epochs, alpha=1e-2
    )
    re_schedule = optax.cosine_decay_schedule(
        init_value=args.re_lr, decay_steps=args.epochs, alpha=1e-2
    )
    net_opt = nnx.Optimizer(model, optax.adam(learning_rate=net_schedule), wrt=nnx.Param)
    re_optimizer = optax.adam(learning_rate=re_schedule)
    re_state = re_optimizer.init(log_Re)

    # Build the full physics+data+BC loss over initial collocation points
    loss_fn = make_loss_fn(
        x_col, y_col, x_in, y_in, x_wall, y_wall,
        x_obs, y_obs, u_obs, v_obs,
        w_data=args.w_data,
    )

    @nnx.jit
    def train_step(model, net_opt, log_Re, re_state):
        # Network gradient: full loss (physics + data + BCs)
        loss, net_grads = nnx.value_and_grad(lambda m: loss_fn(m, log_Re))(model)
        # Re gradient: full loss (physics provides the Re-sensitive signal now
        # that the network already encodes a realistic flow from warm-up)
        re_grad = jax.grad(lambda lr: loss_fn(model, lr))(log_Re)
        net_opt.update(model, net_grads)
        re_updates, new_re_state = re_optimizer.update(re_grad, re_state)
        new_log_Re = optax.apply_updates(log_Re, re_updates)
        return new_log_Re, new_re_state, loss

    Re_history = []
    loss_history = []
    PATIENCE_WINDOW = 500
    RESAMPLE_EVERY  = 1000
    RE_STABLE_WINDOW = 200
    RE_STABLE_TOL    = 0.005

    print("Starting joint inference training...\n")

    k_col = k1

    with tqdm(
        total=args.epochs, desc="Infer Re", unit="epoch", dynamic_ncols=True
    ) as pbar:
        for epoch in range(args.epochs):
            log_Re, re_state, loss = train_step(model, net_opt, log_Re, re_state)
            Re_current = float(jnp.exp(log_Re))
            Re_history.append(Re_current)
            loss_history.append(float(loss))

            if epoch % 20 == 0:
                pbar.set_postfix(loss=f"{float(loss):.3e}", Re=f"{Re_current:.1f}")
            pbar.update(1)

            # Re-stability early stopping: if Re has settled, stop before overshoot
            if epoch >= RE_STABLE_WINDOW and epoch % 20 == 0:
                window = Re_history[-RE_STABLE_WINDOW:]
                rel_std = np.std(window) / (np.mean(window) + 1e-12)
                if rel_std < RE_STABLE_TOL:
                    pbar.write(
                        f"Re converged at epoch {epoch}  "
                        f"(Re={Re_current:.2f}, rel_std={rel_std:.4f})"
                    )
                    break

            # resample interior collocation points periodically
            if epoch > 0 and epoch % RESAMPLE_EVERY == 0:
                k_col = jax.random.fold_in(k_col, epoch)
                x_col, y_col = sample_interior_bs(
                    k_col, 12000, X_MIN, X_MAX, H_STEP, H_CHAN
                )
                loss_fn = make_loss_fn(
                    x_col, y_col, x_in, y_in, x_wall, y_wall,
                    x_obs, y_obs, u_obs, v_obs,
                    w_data=args.w_data,
                )
                pbar.write(f"  [resample {epoch}] new collocation points")

            # loss-plateau early stopping (only after half the budget)
            if epoch > args.epochs // 2 and epoch % 100 == 0:
                if len(loss_history) >= 2 * PATIENCE_WINDOW:
                    recent = np.mean(loss_history[-PATIENCE_WINDOW:])
                    older = np.mean(
                        loss_history[-2 * PATIENCE_WINDOW : -PATIENCE_WINDOW]
                    )
                    if abs(recent - older) / (abs(older) + 1e-12) < 5e-4:
                        pbar.write(f"Early stopping at epoch {epoch} (loss plateau)")
                        break

    elapsed = time.perf_counter() - t0
    Re_final = Re_history[-1]
    print(f"\nDone in {elapsed:.1f} s")
    print(
        f"Re_true = {args.Re_true:.1f}   Re_init = {args.Re_init:.1f}   Re_final = {Re_final:.2f}"
    )
    print(f"Error = {abs(Re_final - args.Re_true) / args.Re_true * 100:.1f}%\n")

    # ---- save ----
    np.savetxt(os.path.join(RESULTS_DIR, "Re_history.txt"), np.array(Re_history))
    np.savetxt(os.path.join(RESULTS_DIR, "loss_history.txt"), np.array(loss_history))
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w") as f:
        f.write(f"Re_true  = {args.Re_true}\n")
        f.write(f"Re_init  = {args.Re_init}\n")
        f.write(f"Re_final = {Re_final:.4f}\n")
        f.write(f"error_%  = {abs(Re_final - args.Re_true) / args.Re_true * 100:.2f}\n")
        f.write(f"n_obs    = {len(x_obs)}\n")
        f.write(f"w_data   = {args.w_data}\n")
        f.write(f"epochs   = {args.epochs}\n")
        f.write(f"elapsed_s= {elapsed:.1f}\n")
        f.write(f"obs_src  = OpenFOAM\n")

    # ---- plots ----
    plot_Re_convergence(Re_history, args.Re_true, PLOTS_DIR)
    plot_inference_loss(loss_history, PLOTS_DIR)

    print(f"\nPlots saved to {PLOTS_DIR}/")
    print(f"Results saved to {RESULTS_DIR}/")
