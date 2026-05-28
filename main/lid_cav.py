import os
import time
import json
import argparse

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from src.pinn import (
    BackwardStepPINN,
    lbfgs_step,
    save_model,
    load_model_state,
    residuals_batch_lc,
    eval_uvp_batch_lc,
    sample_interior_lc,
)

from src.plotting import plot_loss, plot_lid_cavity

jax.config.update("jax_enable_x64", True)


# ============================================================
# Configuration  (matches train_backward_step_v2.py)
# ============================================================

WIDTHS = [128, 128, 128, 128]
N_COL = 12000
ADAM_EPOCHS = 8000
LBFGS_STEPS = 10000
LR_START = 1e-3
LR_END = 1e-5

RESAMPLE_EVERY = 100
LOG_EVERY = 20
DIAG_EVERY = 2000


# ============================================================
# Loss
# ============================================================


def make_loss_fn(Re, x_col, y_col):
    def loss_fn(m):
        cont, mom_x, mom_y = residuals_batch_lc(m, Re, x_col, y_col)
        phys = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        _, _, p_col_vals = eval_uvp_batch_lc(m, x_col, y_col)
        p_anchor = jnp.mean(p_col_vals) ** 2

        return phys + 0.1 * p_anchor

    return loss_fn


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Re", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Re = args.Re
    seed = args.seed

    print(f"\nLid-driven cavity PINN  |  Re = {Re}  seed = {seed}")
    print(f"  Domain      : [-1, 1]^2")
    print(f"  Network     : {WIDTHS}")
    print(f"  Adam epochs : {ADAM_EPOCHS}   L-BFGS steps: {LBFGS_STEPS}")
    print(f"  Collocation : {N_COL}\n")

    RESULTS_DIR = f"results/lid_cavity/Re{int(Re)}"
    PLOTS_DIR = f"plots/lid_cavity/Re{int(Re)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    PARAMS_FILE = os.path.join(RESULTS_DIR, "params.npz")
    ADAM_LOSSES_FILE = os.path.join(RESULTS_DIR, "adam_losses.txt")
    LBFGS_LOSSES_FILE = os.path.join(RESULTS_DIR, "lbfgs_losses.txt")

    key = jax.random.key(seed)
    model = BackwardStepPINN(
        widths=WIDTHS,
        key=key,
        activation=jax.nn.tanh,
        n_inputs=2,
    )

    if all(
        os.path.exists(p) for p in [PARAMS_FILE, ADAM_LOSSES_FILE, LBFGS_LOSSES_FILE]
    ):
        print(f"Loading saved params from {RESULTS_DIR}/\n")
        load_model_state(model, PARAMS_FILE)
        adam_losses = np.loadtxt(ADAM_LOSSES_FILE).tolist()
        lbfgs_losses = np.loadtxt(LBFGS_LOSSES_FILE).tolist()

    else:
        # --------------------------------------------------------
        # Sample collocation points (boundary BCs via hard ansatz)
        # --------------------------------------------------------
        k_col = key
        print("Sampling collocation points...")
        x_col, y_col = sample_interior_lc(k_col, N_COL)
        print(f"  interior: {len(x_col)}\n")

        loss_fn = make_loss_fn(Re, x_col, y_col)

        # --------------------------------------------------------
        # Adam with cosine LR decay
        # --------------------------------------------------------
        lr_schedule = optax.cosine_decay_schedule(
            init_value=LR_START,
            decay_steps=ADAM_EPOCHS,
            alpha=LR_END / LR_START,
        )
        tx = optax.adam(learning_rate=lr_schedule)
        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

        @nnx.jit
        def adam_step(model, opt):
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            opt.update(model, grads)
            return loss

        adam_losses = []
        t0 = time.perf_counter()

        with tqdm(
            total=ADAM_EPOCHS, desc="Adam", unit="epoch", dynamic_ncols=True
        ) as pbar:
            for epoch in range(ADAM_EPOCHS):
                loss = adam_step(model, opt)

                if epoch % LOG_EVERY == 0:
                    last_loss = float(loss)
                    adam_losses.extend([last_loss] * LOG_EVERY)
                    pbar.set_postfix(loss=f"{last_loss:.3e}")
                pbar.update(1)

                # diagnostic breakdown
                if epoch > 0 and epoch % DIAG_EVERY == 0:
                    cont, mom_x, mom_y = residuals_batch_lc(model, Re, x_col, y_col)
                    d_cont = float(jnp.mean(cont**2))
                    d_mom = float(jnp.mean(mom_x**2) + jnp.mean(mom_y**2))
                    pbar.write(f"  [diag {epoch}] cont={d_cont:.2e}  mom={d_mom:.2e}")

                # resample interior points
                if epoch > 0 and epoch % RESAMPLE_EVERY == 0:
                    k_col = jax.random.fold_in(k_col, epoch)
                    x_col, y_col = sample_interior_lc(k_col, N_COL)
                    loss_fn = make_loss_fn(Re, x_col, y_col)

                    @nnx.jit
                    def adam_step(model, opt):
                        loss, grads = nnx.value_and_grad(loss_fn)(model)
                        opt.update(model, grads)
                        return loss

                    pbar.write(f"  [resample {epoch}] new collocation points")

                # early stopping
                if epoch > ADAM_EPOCHS // 2 and epoch % LOG_EVERY == 0:
                    window = 1000
                    if len(adam_losses) >= 2 * window:
                        recent = np.mean(adam_losses[-window:])
                        older = np.mean(adam_losses[-2 * window : -window])
                        if abs(recent - older) / (abs(older) + 1e-12) < 5e-4:
                            pbar.write(f"Early stopping at epoch {epoch}")
                            break

        adam_losses = adam_losses[: epoch + 1]
        adam_time = time.perf_counter() - t0
        print(f"Adam complete in {adam_time:.1f} s\n")

        # --------------------------------------------------------
        # L-BFGS
        # --------------------------------------------------------
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss_of_params(p):
            m = nnx.merge(graphdef, p, rest)
            return loss_fn(m)

        lbfgs_opt = optax.lbfgs()
        lbfgs_state = lbfgs_opt.init(params)

        @jax.jit
        def lbfgs_train_step(params, opt_state):
            return lbfgs_step(
                graphdef, params, rest, opt_state, lbfgs_opt, loss_of_params
            )

        lbfgs_losses = []
        t1 = time.perf_counter()

        with tqdm(
            total=LBFGS_STEPS, desc="L-BFGS", unit="step", dynamic_ncols=True
        ) as pbar:
            for step in range(LBFGS_STEPS):
                params, lbfgs_state, loss = lbfgs_train_step(params, lbfgs_state)

                if step % LOG_EVERY == 0:
                    last_loss = float(loss)
                    lbfgs_losses.extend([last_loss] * LOG_EVERY)
                    pbar.set_postfix(loss=f"{last_loss:.3e}")
                pbar.update(1)

                # early stopping
                if step >= 200 and step % LOG_EVERY == 0:
                    rel = abs(lbfgs_losses[step] - lbfgs_losses[step - 200]) / (
                        abs(lbfgs_losses[step - 200]) + 1e-12
                    )
                    if rel < 1e-4:
                        pbar.write(f"Early stopping at step {step}")
                        break

        lbfgs_losses = lbfgs_losses[: step + 1]
        nnx.update(model, params)

        lbfgs_time = time.perf_counter() - t1
        print(f"L-BFGS complete in {lbfgs_time:.1f} s\n")

        # --------------------------------------------------------
        # Save
        # --------------------------------------------------------
        save_model(model, PARAMS_FILE)
        np.savetxt(ADAM_LOSSES_FILE, np.array(adam_losses))
        np.savetxt(LBFGS_LOSSES_FILE, np.array(lbfgs_losses))
        with open(os.path.join(RESULTS_DIR, "timing.json"), "w") as f:
            json.dump(
                {
                    "adam_s": adam_time,
                    "lbfgs_s": lbfgs_time,
                    "total_s": adam_time + lbfgs_time,
                    "adam_epochs": len(adam_losses),
                    "lbfgs_steps": len(lbfgs_losses),
                },
                f,
                indent=2,
            )

    # ============================================================
    # Evaluation + plotting
    # ============================================================
    plot_loss(adam_losses, lbfgs_losses, plots_dir=PLOTS_DIR)

    print("Evaluating final field...")
    N = 150
    x1d = jnp.linspace(-1.0, 1.0, N)
    y1d = jnp.linspace(-1.0, 1.0, N)
    X, Y = jnp.meshgrid(x1d, y1d)

    u, v, p = eval_uvp_batch_lc(model, X.ravel(), Y.ravel())
    u = u.reshape(N, N)
    v = v.reshape(N, N)
    p = p.reshape(N, N)
    p = p - jnp.mean(p)

    # streamfunction (integrate u over y)
    u_np = np.array(u)
    dy = float(y1d[1] - y1d[0])
    psi = np.zeros_like(u_np)
    for j in range(1, N):
        psi[j, :] = psi[j - 1, :] + 0.5 * (u_np[j - 1, :] + u_np[j, :]) * dy

    idx = np.argmin(psi)
    iy, ix = np.unravel_index(idx, psi.shape)
    x_vortex = float(x1d[ix])
    y_vortex = float(y1d[iy])
    print(f"\nMain vortex centre: x = {x_vortex:.4f},  y = {y_vortex:.4f}")

    plot_lid_cavity(
        np.array(X),
        np.array(Y),
        u_np,
        np.array(v),
        np.array(p),
        psi,
        x_vortex,
        y_vortex,
        Re,
        plots_dir=PLOTS_DIR,
    )

    print(f"\nFinished.\nPlots saved to:\n{PLOTS_DIR}\n")
