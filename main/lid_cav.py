import os
import time
import json
import argparse

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

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
# Re-adaptive configuration
# ============================================================


def get_config(Re):
    if Re <= 100:
        n_col, adam_epochs, lbfgs_steps = 8000, 5000, 1000
    elif Re <= 400:
        n_col, adam_epochs, lbfgs_steps = 10000, 8000, 2000
    else:
        n_col, adam_epochs, lbfgs_steps = 12000, 10000, 3000
    return dict(
        widths=[128, 128, 128, 128],
        n_col=n_col,
        adam_epochs=adam_epochs,
        lr_start=1e-3,
        lr_end=1e-5,
        lbfgs_steps=lbfgs_steps,
    )


# ============================================================
# loss
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
# main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Re", type=float, default=100.0)
    parser.add_argument(
        "--warm_start",
        type=float,
        default=None,
        metavar="RE",
        help="Re value of a previously trained run to warm-start from.",
    )
    args = parser.parse_args()
    Re = args.Re

    cfg = get_config(Re)

    print(f"\nLid-driven cavity PINN  |  Re = {Re}")
    print(f"  Domain      : [-1, 1]^2")
    print(f"  Network     : {cfg['widths']}")
    print(f"  Adam epochs : {cfg['adam_epochs']}   L-BFGS steps: {cfg['lbfgs_steps']}")
    print(f"  Collocation : {cfg['n_col']}\n")

    RESULTS_DIR = f"results/lid_cavity/Re{int(Re)}"
    PLOTS_DIR = f"plots/lid_cavity/Re{int(Re)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    PRED_FILES = {
        "params": os.path.join(RESULTS_DIR, "params.npz"),
        "adam_losses": os.path.join(RESULTS_DIR, "adam_losses.txt"),
        "lbfgs_losses": os.path.join(RESULTS_DIR, "lbfgs_losses.txt"),
    }

    key = jax.random.key(0)
    model = BackwardStepPINN(
        widths=cfg["widths"],
        key=key,
        activation=jax.nn.tanh,
        n_inputs=2,
    )

    if all(os.path.exists(PRED_FILES[k]) for k in PRED_FILES):
        print(f"Loading saved params from {RESULTS_DIR}/\n")
        load_model_state(model, PRED_FILES["params"])
        adam_losses = np.loadtxt(PRED_FILES["adam_losses"]).tolist()
        lbfgs_losses = np.loadtxt(PRED_FILES["lbfgs_losses"]).tolist()

    else:
        if args.warm_start is not None:
            warm_path = f"results/lid_cavity/Re{int(args.warm_start)}/params.npz"
            print(f"Warm-starting from {warm_path}\n")
            load_model_state(model, warm_path)

        # --------------------------------------------------------
        # sample points
        # --------------------------------------------------------
        (k1,) = jax.random.split(key, 1)
        print("Sampling collocation points...")
        x_col, y_col = sample_interior_lc(k1, cfg["n_col"])
        print(f"  interior: {len(x_col)}\n")

        # --------------------------------------------------------
        # loss
        # --------------------------------------------------------
        loss_fn = make_loss_fn(Re, x_col, y_col)

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
        print("Starting Adam training...\n")
        t0 = time.perf_counter()

        for epoch in range(cfg["adam_epochs"]):
            loss = adam_step(model, opt)
            adam_losses.append(float(loss))

            if epoch % 500 == 0:
                print(f"Epoch {epoch:6d} | loss = {loss:.6e}")

            if args.warm_start is None and epoch > cfg["adam_epochs"] // 2:
                window = 1000
                if len(adam_losses) >= 2 * window:
                    recent = np.mean(adam_losses[-window:])
                    older = np.mean(adam_losses[-2 * window : -window])
                    if abs(recent - older) / abs(older) < 5e-4:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

        adam_time = time.perf_counter() - t0
        print(f"\nAdam complete in {adam_time:.1f} s\n")

        # --------------------------------------------------------
        # L-BFGS (nnx.split/merge)
        # --------------------------------------------------------
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss_of_params(p):
            m = nnx.merge(graphdef, p, rest)
            return loss_fn(m)

        lbfgs_opt = optax.lbfgs()
        lbfgs_state = lbfgs_opt.init(params)

        @jax.jit
        def lbfgs_train_step(params, opt_state):
            return lbfgs_step(graphdef, params, rest, opt_state, lbfgs_opt, loss_of_params)

        lbfgs_losses = []
        print("Starting L-BFGS training...\n")
        t1 = time.perf_counter()

        for step in range(cfg["lbfgs_steps"]):
            params, lbfgs_state, loss = lbfgs_train_step(params, lbfgs_state)
            lbfgs_losses.append(float(loss))

            if step % 200 == 0:
                print(f"Step {step:5d} | loss = {loss:.6e}")

            if step >= 200:
                rel = abs(lbfgs_losses[step] - lbfgs_losses[step - 200]) / abs(
                    lbfgs_losses[step - 200]
                )
                if rel < 1e-3:
                    print(f"\nEarly stopping at step {step}")
                    break

        nnx.update(model, params)

        lbfgs_time = time.perf_counter() - t1
        print(f"\nL-BFGS complete in {lbfgs_time:.1f} s\n")

        # --------------------------------------------------------
        # save
        # --------------------------------------------------------
        save_model(model, PRED_FILES["params"])
        np.savetxt(PRED_FILES["adam_losses"], np.array(adam_losses))
        np.savetxt(PRED_FILES["lbfgs_losses"], np.array(lbfgs_losses))
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
    N = 150
    x1d = jnp.linspace(-1.0, 1.0, N)
    y1d = jnp.linspace(-1.0, 1.0, N)
    X, Y = jnp.meshgrid(x1d, y1d)

    u, v, p = eval_uvp_batch_lc(model, X.ravel(), Y.ravel())
    u = u.reshape(N, N)
    v = v.reshape(N, N)
    p = p.reshape(N, N)

    p = p - jnp.mean(p)

    # ---- streamfunction ----
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
