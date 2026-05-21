"""
infer_Re_backward_step.py
-------------------------
Reynolds number inference (inverse PINN).

Setup
-----
"Observations" are N_obs velocity values sampled from the converged Re=200
PINN solution.  We then start a fresh network with Re initialised far from
the truth (Re_init, default 800) and optimise *both* network weights and Re
jointly.  The loss contains:

    physics residuals (with the current learnable Re)
  + data term:  mean((u_pinn(x_obs, y_obs) - u_obs)^2)

Because the NS residuals depend on Re through the viscosity (nu = 1/Re),
gradients flow back to Re and pull it toward the physically consistent value.

Usage
-----
    python main/infer_Re_backward_step.py --Re_init 800 --n_obs 100
"""

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
import matplotlib.pyplot as plt

from src.pinn import (
    BackwardStepPINN,
    load_model_state,
    eval_uvp_batch_bs,
    residuals_batch_bs,
)

jax.config.update("jax_enable_x64", True)

# ============================================================
# geometry  (must match train_backward_step_v2.py)
# ============================================================

X_MIN = -2.0
X_MAX = 20.0
H_STEP = 1.0
H_CHAN = 2.0
U_MEAN = 1.0
RE_TRUE = 200.0

WIDTHS = [128, 128, 128, 128]


# ============================================================
# sampler
# ============================================================


def sample_interior(key, n):
    k1, k2 = jax.random.split(key)
    pts1 = jax.random.uniform(k1, (n // 2, 2))
    x1 = pts1[:, 0] * 8.0
    y1 = pts1[:, 1] * H_CHAN
    pts2 = jax.random.uniform(k2, (n, 2))
    x2 = pts2[:, 0] * (X_MAX - X_MIN) + X_MIN
    y2 = pts2[:, 1] * H_CHAN
    valid = ~((x2 < 0.0) & (y2 < H_STEP))
    x2, y2 = x2[valid][: n - n // 2], y2[valid][: n - n // 2]
    return jnp.concatenate([x1, x2]), jnp.concatenate([y1, y2])


# ============================================================
# loss
# ============================================================


def make_loss_fn(x_col, y_col, x_obs, y_obs, u_obs, w_data=100.0):
    """
    Loss that takes (model, log_Re) and differentiates w.r.t. both.
    Re = exp(log_Re) keeps it strictly positive.
    All velocity BCs are hard-enforced via the ansatz.
    """

    def loss_fn(model, log_Re):
        Re = jnp.exp(log_Re)

        cont, mom_x, mom_y = residuals_batch_bs(
            model, Re, x_col, y_col, X_MIN, X_MAX, H_CHAN, H_STEP, U_MEAN,
        )
        phys = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        u_pred, _, _ = eval_uvp_batch_bs(
            model, x_obs, y_obs, X_MIN, X_MAX, H_CHAN, H_STEP, U_MEAN,
        )
        data = jnp.mean((u_pred - u_obs) ** 2)

        return 1.0 * phys + w_data * data

    return loss_fn


# ============================================================
# plotting
# ============================================================


def plot_Re_convergence(Re_history, Re_true, plots_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(Re_history, color="steelblue", lw=1.5, label="Inferred Re")
    ax.axhline(
        Re_true, color="crimson", ls="--", lw=1.5, label=f"True Re = {Re_true:.0f}"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Re")
    ax.set_title("Reynolds number inference — PINN inverse problem")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(plots_dir, "Re_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def plot_observation_points(x_obs, y_obs, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(np.array(x_obs), np.array(y_obs), s=8, color="steelblue", alpha=0.7)
    step = plt.Polygon(
        [[-2, 0], [0, 0], [0, 1], [-2, 1]], closed=True, fc="lightgray", ec="gray"
    )
    ax.add_patch(step)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(0, H_CHAN)
    ax.set_aspect("equal")
    ax.set_title(f"Observation points (N={len(x_obs)})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    path = os.path.join(plots_dir, "observation_points.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--Re_true", type=float, default=200.0,
        help="True Re whose PINN solution is used as observations",
    )
    parser.add_argument(
        "--Re_init", type=float, default=800.0,
        help="Initial guess for Re (start far from truth)",
    )
    parser.add_argument("--n_obs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w_data", type=float, default=100.0)
    args = parser.parse_args()

    RESULTS_DIR = "results/infer_Re"
    PLOTS_DIR = "plots/infer_Re"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- load trained Re=200 PINN to generate observations ----
    obs_params_path = f"results/backward_step_v2/Re{int(args.Re_true)}/params.npz"
    if not os.path.exists(obs_params_path):
        raise FileNotFoundError(
            f"Trained Re={args.Re_true} params not found at {obs_params_path}.\n"
            f"Run:  python main/train_backward_step_v2.py --Re {int(args.Re_true)}"
        )
    print(f"\nLoading Re={args.Re_true} PINN from {obs_params_path}")
    obs_model = BackwardStepPINN(
        widths=WIDTHS, key=jax.random.key(0), activation=jax.nn.tanh, n_inputs=2,
    )
    load_model_state(obs_model, obs_params_path)

    # sample observation locations from the fluid interior
    key_obs = jax.random.key(99)
    pts = jax.random.uniform(key_obs, (args.n_obs * 6, 2))
    x_cand = pts[:, 0] * (X_MAX - X_MIN) + X_MIN
    y_cand = pts[:, 1] * H_CHAN
    valid = ~((x_cand < 0.0) & (y_cand < H_STEP))
    x_obs = x_cand[valid][: args.n_obs]
    y_obs = y_cand[valid][: args.n_obs]

    u_obs, _, _ = eval_uvp_batch_bs(
        obs_model, x_obs, y_obs, X_MIN, X_MAX, H_CHAN, H_STEP, U_MEAN,
    )
    print(f"  Generated {len(x_obs)} observation points from Re={args.Re_true} PINN")
    plot_observation_points(x_obs, y_obs, PLOTS_DIR)

    # ---- create inference model, warm-started from Re=200 solution ----
    key = jax.random.key(42)
    model = BackwardStepPINN(
        widths=WIDTHS, key=key, activation=jax.nn.tanh, n_inputs=2,
    )
    nnx.update(model, nnx.state(obs_model))

    log_Re = jnp.array(float(np.log(args.Re_init)))

    print(f"\nRe_true = {args.Re_true:.1f}   Re_init = {args.Re_init:.1f}")
    print(f"  Network: warm-started from Re={args.Re_true:.0f} PINN")
    print(f"  log_Re init = {float(log_Re):.4f}   (truth = {np.log(args.Re_true):.4f})\n")

    # ---- collocation points ----
    (k1,) = jax.random.split(key, 1)
    x_col, y_col = sample_interior(k1, 12000)
    print(f"Collocation pts: {len(x_col)}\n")

    loss_fn = make_loss_fn(x_col, y_col, x_obs, y_obs, u_obs, w_data=args.w_data)

    # ---- separate optimizers for network and Re ----
    net_lr = args.lr * 1e-2
    re_lr = args.lr * 10.0

    net_tx = optax.adam(learning_rate=net_lr)
    re_optimizer = optax.adam(learning_rate=re_lr)
    net_opt = nnx.Optimizer(model, net_tx, wrt=nnx.Param)
    re_state = re_optimizer.init(log_Re)

    @nnx.jit
    def train_step(model, net_opt, log_Re, re_state):
        def joint_loss(m, lr):
            return loss_fn(m, lr)

        (loss, ), grads = nnx.value_and_grad(
            lambda m: (joint_loss(m, log_Re),), has_aux=False
        )(model)
        net_opt.update(model, grads)

        re_grad = jax.grad(lambda lr: loss_fn(model, lr))(log_Re)
        re_updates, new_re_state = re_optimizer.update(re_grad, re_state)
        new_log_Re = optax.apply_updates(log_Re, re_updates)
        return new_log_Re, new_re_state, loss

    # ---- training loop ----
    Re_history = []
    loss_history = []
    Re_min = float("inf")
    epochs_since_min = 0
    PATIENCE = 300

    print(f"  net_lr = {net_lr:.1e}   re_lr = {re_lr:.1e}\n")
    print("Starting inference training...\n")
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        log_Re, re_state, loss = train_step(model, net_opt, log_Re, re_state)
        Re_current = float(jnp.exp(log_Re))
        Re_history.append(Re_current)
        loss_history.append(float(loss))

        if epoch % 200 == 0:
            print(f"Epoch {epoch:5d} | loss = {loss:.4e} | Re = {Re_current:.2f}")

        if Re_current < Re_min:
            Re_min = Re_current
            epochs_since_min = 0
        else:
            epochs_since_min += 1
        if epochs_since_min >= PATIENCE and epoch > 500:
            print(f"\nEarly stopping at epoch {epoch}  (Re minimum = {Re_min:.2f})")
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

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history, color="steelblue", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Inference training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "loss.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {PLOTS_DIR}/")
    print(f"Results saved to {RESULTS_DIR}/")
