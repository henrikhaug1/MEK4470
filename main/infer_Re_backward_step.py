import os
import time
import argparse

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib

matplotlib.use("Agg")

from src.pinn import (
    BackwardStepPINN,
    load_fourier_model,
    eval_uvp_batch_bs_raw,
    residuals_batch_bs_raw,
    inlet_profile_bs,
    sample_interior_bs,
    sample_walls_bs,
    sample_inlet_bs,
)
from src.plotting import plot_Re_convergence, plot_observation_points, plot_inference_loss

jax.config.update("jax_enable_x64", True)

# ============================================================
# geometry  (must match train_backward_step_v2.py)
# ============================================================

X_MIN = -2.0
X_MAX = 20.0
H_STEP = 1.0
H_CHAN = 2.0
U_MEAN = 1.0

WIDTHS = [128, 128, 128, 128]  # inference model (plain tanh, trained from scratch)

# ============================================================
# loss  (soft-BC formulation matching forward training)
# ============================================================

W_PHYS = 1.0
W_DATA = 1.0  # scaled by w_data argument
W_INLET = 2.0
W_WALLS = 5.0


def make_loss_fn(x_col, y_col, x_in, y_in, x_wall, y_wall, x_obs, y_obs, u_obs, w_data):
    u_inlet_ref = inlet_profile_bs(y_in, H_STEP, H_CHAN, U_MEAN)

    def loss_fn(model, log_Re):
        Re = jnp.exp(log_Re)

        cont, mom_x, mom_y = residuals_batch_bs_raw(
            model, Re, x_col, y_col, X_MIN, X_MAX, H_CHAN
        )
        phys = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        u_pred, _, _ = eval_uvp_batch_bs_raw(model, x_obs, y_obs, X_MIN, X_MAX, H_CHAN)
        data = jnp.mean((u_pred - u_obs) ** 2)

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
        help="True Re whose PINN solution is used as observations",
    )
    parser.add_argument(
        "--Re_init",
        type=float,
        default=800.0,
        help="Initial guess for Re (start far from truth)",
    )
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--net_lr", type=float, default=1e-3)
    parser.add_argument("--re_lr", type=float, default=1e-4)
    parser.add_argument("--w_data", type=float, default=100.0)
    args = parser.parse_args()

    RESULTS_DIR = "results/infer_Re"
    PLOTS_DIR = "plots/infer_Re"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- load trained PINN to generate observations ----
    obs_params_path = f"results/backward_step_v2/Re{int(args.Re_true)}/params.npz"
    if not os.path.exists(obs_params_path):
        raise FileNotFoundError(
            f"Trained Re={args.Re_true} params not found at {obs_params_path}.\n"
            f"Run:  python main/train_backward_step_v2.py --Re {int(args.Re_true)}"
        )
    print(f"\nLoading Re={args.Re_true} PINN from {obs_params_path}")
    # v2 models are FourierBackwardStepPINN — use load_fourier_model to restore _B
    obs_model = load_fourier_model(obs_params_path, key=jax.random.key(0))

    # sample observation locations from the fluid interior
    key_obs = jax.random.key(99)
    pts = jax.random.uniform(key_obs, (args.n_obs * 6, 2))
    x_cand = pts[:, 0] * (X_MAX - X_MIN) + X_MIN
    y_cand = pts[:, 1] * H_CHAN
    valid = ~((x_cand < 0.0) & (y_cand < H_STEP))
    x_obs = x_cand[valid][: args.n_obs]
    y_obs = y_cand[valid][: args.n_obs]

    u_obs, _, _ = eval_uvp_batch_bs_raw(
        obs_model,
        x_obs,
        y_obs,
        X_MIN,
        X_MAX,
        H_CHAN,
    )
    print(f"  Generated {len(x_obs)} observation points from Re={args.Re_true} PINN")
    plot_observation_points(x_obs, y_obs, PLOTS_DIR, x_min=X_MIN, x_max=X_MAX, h_step=H_STEP, h_chan=H_CHAN)

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
    print(f"  Network: random initialization")
    print(
        f"  log_Re init = {float(log_Re):.4f}   (truth = {np.log(args.Re_true):.4f})\n"
    )

    # ---- collocation + BC points ----
    k1, k_in, k_wall = jax.random.split(key, 3)
    x_col, y_col = sample_interior_bs(k1, 12000, X_MIN, X_MAX, H_STEP, H_CHAN)
    x_in, y_in = sample_inlet_bs(k_in, 500, X_MIN, H_STEP, H_CHAN)
    x_wall, y_wall = sample_walls_bs(k_wall, 2000, X_MIN, X_MAX, H_STEP, H_CHAN)
    print(f"Collocation pts: {len(x_col)}   Inlet pts: 500   Wall pts: {len(x_wall)}\n")

    loss_fn = make_loss_fn(
        x_col, y_col, x_in, y_in, x_wall, y_wall, x_obs, y_obs, u_obs, w_data=args.w_data
    )

    # ---- separate optimizers: higher LR for network, lower for Re ----
    net_tx = optax.adam(learning_rate=args.net_lr)
    re_optimizer = optax.adam(learning_rate=args.re_lr)
    net_opt = nnx.Optimizer(model, net_tx, wrt=nnx.Param)
    re_state = re_optimizer.init(log_Re)

    @nnx.jit
    def train_step(model, net_opt, log_Re, re_state):
        # Compute both gradients from the same model state, before any update.
        # (net_opt.update mutates the model in-place, so re_grad must come first.)
        loss, net_grads = nnx.value_and_grad(lambda m: loss_fn(m, log_Re))(model)
        re_grad = jax.grad(lambda lr: loss_fn(model, lr))(log_Re)
        # Apply updates
        net_opt.update(model, net_grads)
        re_updates, new_re_state = re_optimizer.update(re_grad, re_state)
        new_log_Re = optax.apply_updates(log_Re, re_updates)
        return new_log_Re, new_re_state, loss

    # ---- training loop with loss-plateau early stopping ----
    Re_history = []
    loss_history = []
    PATIENCE_WINDOW = 500

    print(f"  net_lr = {args.net_lr:.1e}   re_lr = {args.re_lr:.1e}\n")
    print("Starting inference training...\n")
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        log_Re, re_state, loss = train_step(model, net_opt, log_Re, re_state)
        Re_current = float(jnp.exp(log_Re))
        Re_history.append(Re_current)
        loss_history.append(float(loss))

        if epoch % 200 == 0:
            print(f"Epoch {epoch:5d} | loss = {loss:.4e} | Re = {Re_current:.2f}")

        # loss-plateau early stopping (only after half the budget)
        if epoch > args.epochs // 2 and epoch % 100 == 0:
            window = PATIENCE_WINDOW
            if len(loss_history) >= 2 * window:
                recent = np.mean(loss_history[-window:])
                older = np.mean(loss_history[-2 * window : -window])
                if abs(recent - older) / (abs(older) + 1e-12) < 5e-4:
                    print(f"\nEarly stopping at epoch {epoch} (loss plateau)")
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

    # ---- plots ----
    plot_Re_convergence(Re_history, args.Re_true, PLOTS_DIR)
    plot_inference_loss(loss_history, PLOTS_DIR)

    print(f"\nPlots saved to {PLOTS_DIR}/")
    print(f"Results saved to {RESULTS_DIR}/")
