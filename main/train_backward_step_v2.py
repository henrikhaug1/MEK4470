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
    eval_uvp_batch_bs,
    residuals_batch_bs,
)

from src.plotting import plot_backward_step, plot_loss, plot_recirculation

jax.config.update("jax_enable_x64", True)


# ============================================================
# Geometry
# ============================================================

X_MIN = -2.0
H_STEP = 1.0
H_CHAN = 2.0
U_MEAN = 1.0


# ============================================================
# Re-adaptive configuration
# ============================================================


def get_config(Re):
    x_r_est = 0.025 * Re
    x_max = max(20.0, x_r_est * 2.5)
    return dict(
        x_max=x_max,
        widths=[32, 32, 32, 32],
        n_col=5000,
        adam_epochs=5000,
        lr_start=1e-3,
        lr_end=1e-5,
        lbfgs_steps=5000,
        x_recirc=min(x_r_est * 1.5, x_max),
    )


# ============================================================
# Re-adaptive samplers
# ============================================================


def sample_interior(key, n_col, x_max, x_recirc):
    SHEAR_HALF = 0.2
    N_corner = n_col // 10
    N_shear = (n_col * 15) // 100
    N_recirc = (n_col * 30) // 100
    N_inlet = (n_col * 10) // 100
    N_rest = n_col - N_corner - N_shear - N_recirc - N_inlet
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    pts_c = jax.random.uniform(k1, (N_corner * 4, 2))
    xc = pts_c[:, 0] * 2.0 - 0.5
    yc = pts_c[:, 1] * 1.0 + 0.5
    valid_c = ~((xc < 0.0) & (yc < H_STEP))
    xc, yc = xc[valid_c][:N_corner], yc[valid_c][:N_corner]

    pts_s = jax.random.uniform(k2, (N_shear, 2))
    xs = pts_s[:, 0] * x_recirc
    ys = pts_s[:, 1] * (2.0 * SHEAR_HALF) + (H_STEP - SHEAR_HALF)

    pts1 = jax.random.uniform(k3, (N_recirc, 2))
    x1 = pts1[:, 0] * x_recirc
    y1 = pts1[:, 1] * H_CHAN

    pts_i = jax.random.uniform(k4, (N_inlet, 2))
    xi = pts_i[:, 0] * abs(X_MIN) + X_MIN
    yi = pts_i[:, 1] * (H_CHAN - H_STEP) + H_STEP

    pts2 = jax.random.uniform(k5, (N_rest * 2, 2))
    x2 = pts2[:, 0] * (x_max - X_MIN) + X_MIN
    y2 = pts2[:, 1] * H_CHAN
    valid2 = ~((x2 < 0.0) & (y2 < H_STEP))
    x2, y2 = x2[valid2][:N_rest], y2[valid2][:N_rest]

    return (
        jnp.concatenate([xc, xs, x1, xi, x2]),
        jnp.concatenate([yc, ys, y1, yi, y2]),
    )


# ============================================================
# Loss
# ============================================================


def make_loss_fn(model, Re, x_col, y_col, x_max):
    W_P = 1.0
    N_OUT = 200
    x_outlet = jnp.full(N_OUT, x_max)
    y_outlet = jnp.linspace(0.0, H_CHAN, N_OUT)

    def loss_fn(m):
        cont, mom_x, mom_y = residuals_batch_bs(
            m,
            Re,
            x_col,
            y_col,
            X_MIN,
            x_max,
            H_CHAN,
            H_STEP,
            U_MEAN,
        )
        physics_loss = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        _, _, p_out = eval_uvp_batch_bs(
            m,
            x_outlet,
            y_outlet,
            X_MIN,
            x_max,
            H_CHAN,
            H_STEP,
            U_MEAN,
        )
        outlet_loss = jnp.mean(p_out**2)

        return physics_loss + W_P * outlet_loss

    return loss_fn


# ============================================================
# BC sanity check
# ============================================================


def check_bcs(model, x_max, n=200):
    y_in = jnp.linspace(H_STEP, H_CHAN, n)
    x_in = jnp.full(n, X_MIN)
    u_in, v_in, _ = eval_uvp_batch_bs(
        model, x_in, y_in, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    u_ref = U_MEAN * 6.0 * (y_in - H_STEP) * (H_CHAN - y_in) / (H_CHAN - H_STEP) ** 2
    inlet_u_err = float(jnp.sqrt(jnp.mean((u_in - u_ref) ** 2)))
    inlet_v_err = float(jnp.sqrt(jnp.mean(v_in**2)))

    y_out = jnp.linspace(0.0, H_CHAN, n)
    x_out = jnp.full(n, x_max)
    _, _, p_out = eval_uvp_batch_bs(
        model, x_out, y_out, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    outlet_p_err = float(jnp.sqrt(jnp.mean(p_out**2)))

    x_bot = jnp.linspace(0.0, x_max, n)
    y_bot = jnp.zeros(n)
    u_bot, v_bot, _ = eval_uvp_batch_bs(
        model, x_bot, y_bot, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    wall_u_err = float(jnp.sqrt(jnp.mean(u_bot**2)))
    wall_v_err = float(jnp.sqrt(jnp.mean(v_bot**2)))

    y_step = jnp.linspace(0.0, H_STEP, n)
    x_step = jnp.zeros(n)
    u_step, v_step, _ = eval_uvp_batch_bs(
        model, x_step, y_step, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    step_u_err = float(jnp.sqrt(jnp.mean(u_step**2)))
    step_v_err = float(jnp.sqrt(jnp.mean(v_step**2)))

    print("\n── BC sanity check (RMS errors) ──────────────────────────────")
    print(
        f"  Inlet   u error : {inlet_u_err:.2e}   v error  : {inlet_v_err:.2e}  [hard]"
    )
    print(
        f"  Outlet  p error : {outlet_p_err:.2e}                          [soft — nonzero OK]"
    )
    print(f"  Bottom wall u   : {wall_u_err:.2e}   v        : {wall_v_err:.2e}  [hard]")
    print(f"  Step face   u   : {step_u_err:.2e}   v        : {step_v_err:.2e}  [hard]")
    print("──────────────────────────────────────────────────────────────\n")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Re", type=float, default=200.0)
    args = parser.parse_args()
    Re = args.Re

    cfg = get_config(Re)
    X_MAX = cfg["x_max"]

    print(f"\nBackward step PINN v2  |  Re = {Re}")
    print(f"  Domain      : x ∈ [{X_MIN}, {X_MAX:.1f}]  y ∈ [0, {H_CHAN}]")
    print(f"  Network     : {cfg['widths']}")
    print(f"  Adam epochs : {cfg['adam_epochs']}   L-BFGS steps: {cfg['lbfgs_steps']}")
    print(f"  Collocation : {cfg['n_col']}   x_recirc ≈ {cfg['x_recirc']:.1f} H\n")

    RESULTS_DIR = f"results/backward_step_v2/Re{int(Re)}"
    PLOTS_DIR = f"plots/backward_step_v2/Re{int(Re)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    PARAMS_FILE = os.path.join(RESULTS_DIR, "params.npz")
    ADAM_LOSSES_FILE = os.path.join(RESULTS_DIR, "adam_losses.txt")
    LBFGS_LOSSES_FILE = os.path.join(RESULTS_DIR, "lbfgs_losses.txt")

    # Always create the model first
    key = jax.random.key(0)
    model = BackwardStepPINN(
        widths=cfg["widths"],
        key=key,
        activation=jax.nn.tanh,
        n_inputs=2,
    )

    # --------------------------------------------------------
    # sample collocation points
    # --------------------------------------------------------
    _, k1 = jax.random.split(key, 2)
    print("Sampling collocation points...")
    x_col, y_col = sample_interior(k1, cfg["n_col"], X_MAX, cfg["x_recirc"])
    print(f"  interior: {len(x_col)}\n")

    # --------------------------------------------------------
    # loss
    # --------------------------------------------------------
    loss_fn = make_loss_fn(model, Re, x_col, y_col, X_MAX)

    # --------------------------------------------------------
    # Adam with cosine LR decay (nnx.Optimizer)
    # --------------------------------------------------------
    lr_schedule = optax.cosine_decay_schedule(
        init_value=cfg["lr_start"],
        decay_steps=cfg["adam_epochs"],
        alpha=cfg["lr_end"] / cfg["lr_start"],
    )
    tx = optax.adam(learning_rate=lr_schedule)
    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    @nnx.jit
    def adam_step(model, opt):
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    adam_losses = []
    LOG_EVERY = 20
    t0 = time.perf_counter()

    with tqdm(
        total=cfg["adam_epochs"], desc="Adam", unit="epoch", dynamic_ncols=True
    ) as pbar:
        last_loss = 0.0
        for epoch in range(cfg["adam_epochs"]):
            loss = adam_step(model, opt)
            if epoch % LOG_EVERY == 0:
                last_loss = float(loss)
                adam_losses.extend([last_loss] * LOG_EVERY)
                pbar.set_postfix(loss=f"{last_loss:.3e}")
            pbar.update(1)

            if epoch > cfg["adam_epochs"] // 2 and epoch % LOG_EVERY == 0:
                window = 1000
                if len(adam_losses) >= 2 * window:
                    recent = np.mean(adam_losses[-window:])
                    older = np.mean(adam_losses[-2 * window : -window])
                    if abs(recent - older) / abs(older) < 5e-4:
                        pbar.write(f"Early stopping at epoch {epoch}")
                        break

    adam_losses = adam_losses[: epoch + 1]

    adam_time = time.perf_counter() - t0
    print(f"Adam complete in {adam_time:.1f} s\n")
    check_bcs(model, X_MAX)

    # --------------------------------------------------------
    # L-BFGS (nnx.split/merge pattern)
    # --------------------------------------------------------
    graphdef, params, rest = nnx.split(model, nnx.Param, ...)

    def loss_of_params(p):
        m = nnx.merge(graphdef, p, rest)
        return loss_fn(m)

    lbfgs_opt = optax.lbfgs()
    lbfgs_state = lbfgs_opt.init(params)

    @nnx.jit
    def lbfgs_train_step(params, opt_state):
        return lbfgs_step(graphdef, params, rest, opt_state, lbfgs_opt, loss_of_params)

    lbfgs_losses = []
    t1 = time.perf_counter()

    with tqdm(
        total=cfg["lbfgs_steps"], desc="L-BFGS", unit="step", dynamic_ncols=True
    ) as pbar:
        last_loss = 0.0
        for step in range(cfg["lbfgs_steps"]):
            params, lbfgs_state, loss = lbfgs_train_step(params, lbfgs_state)
            if step % LOG_EVERY == 0:
                last_loss = float(loss)
                lbfgs_losses.extend([last_loss] * LOG_EVERY)
                pbar.set_postfix(loss=f"{last_loss:.3e}")
            pbar.update(1)

            if step >= 200 and step % LOG_EVERY == 0:
                rel = abs(lbfgs_losses[step] - lbfgs_losses[step - 200]) / abs(
                    lbfgs_losses[step - 200]
                )
                if rel < 1e-3:
                    pbar.write(f"Early stopping at step {step}")
                    break

    lbfgs_losses = lbfgs_losses[: step + 1]
    nnx.update(model, params)

    lbfgs_time = time.perf_counter() - t1
    print(f"L-BFGS complete in {lbfgs_time:.1f} s\n")
    check_bcs(model, X_MAX)

    # --------------------------------------------------------
    # save
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
    # evaluation + plotting
    # ============================================================
    plot_loss(adam_losses, lbfgs_losses, plots_dir=PLOTS_DIR)

    print("Evaluating final field...")
    Nx, Ny = 400, 80
    x1d = jnp.linspace(X_MIN, X_MAX, Nx)
    y1d = jnp.linspace(0.0, H_CHAN, Ny)
    X, Y = jnp.meshgrid(x1d, y1d)
    mask = ~((X < 0.0) & (Y < H_STEP))

    u, v, p = eval_uvp_batch_bs(
        model,
        X.ravel(),
        Y.ravel(),
        X_MIN,
        X_MAX,
        H_CHAN,
        H_STEP,
        U_MEAN,
    )
    u = jnp.where(mask, u.reshape(Ny, Nx), jnp.nan)
    v = jnp.where(mask, v.reshape(Ny, Nx), jnp.nan)
    p = jnp.where(mask, p.reshape(Ny, Nx), jnp.nan)

    plot_backward_step(X, Y, u, v, p, plots_dir=PLOTS_DIR)

    # ---- reattachment ----
    print("Computing reattachment length...")
    N_PROBE = 6000
    x_probe = jnp.linspace(0.1, X_MAX, N_PROBE)
    y_probe = jnp.full(N_PROBE, 0.15)

    u_wall, _, _ = eval_uvp_batch_bs(
        model,
        x_probe,
        y_probe,
        X_MIN,
        X_MAX,
        H_CHAN,
        H_STEP,
        U_MEAN,
    )

    u_wall_min = float(jnp.min(u_wall))
    print(
        f"  min u at y=0.15 : {u_wall_min:.4f}  "
        f"({'backflow detected' if u_wall_min < 0 else 'NO backflow — check ansatz'})"
    )

    transitions = jnp.diff(jnp.sign(u_wall))
    neg_to_pos = jnp.where(transitions > 0)[0]

    if len(neg_to_pos) == 0:
        x_r = None
        print("  No reattachment found in domain.")
    else:
        i = int(neg_to_pos[0])
        x0, x1_probe = float(x_probe[i]), float(x_probe[i + 1])
        u0, u1 = float(u_wall[i]), float(u_wall[i + 1])
        x_r = x0 - u0 * (x1_probe - x0) / (u1 - u0)
        print(f"  x_r = {x_r:.4f}  ({x_r / H_STEP:.2f} step heights)")

    plot_recirculation(X, Y, u, x_r, H_STEP, plots_dir=PLOTS_DIR)

    print(f"\nFinished.\nPlots saved to:\n{PLOTS_DIR}\n")
