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
    eval_uvp_batch_bs,
    sample_interior_bs,
    sample_walls_bs,
    sample_inlet_bs,
    sample_outlet_bs,
    residuals_batch_bs,
)

from src.plotting import (
    plot_backward_step,
    plot_loss,
    plot_recirculation,
)

jax.config.update("jax_enable_x64", True)


# =========================================================
# geometry
# =========================================================

X_MIN = -2.0
X_MAX = 22.0
H_STEP = 1.0
H_CHAN = 2.0
U_MAX = 1.0


# =========================================================
# loss
# =========================================================


def make_loss_fn(Re, x_col, y_col, x_wall, y_wall, x_inlet, y_inlet, x_outlet, y_outlet):
    def inlet_profile(y):
        return U_MAX * 4.0 * (y - H_STEP) * (H_CHAN - y) / (H_CHAN - H_STEP) ** 2

    def loss_fn(m):
        cont, mom_x, mom_y = residuals_batch_bs(
            m, Re, x_col, y_col, X_MIN, X_MAX, H_CHAN, H_STEP, U_MAX,
        )
        physics_loss = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        u_wall, v_wall, _ = eval_uvp_batch_bs(
            m, x_wall, y_wall, X_MIN, X_MAX, H_CHAN, H_STEP, U_MAX,
        )
        wall_loss = jnp.mean(u_wall**2) + jnp.mean(v_wall**2)

        u_in, v_in, _ = eval_uvp_batch_bs(
            m, x_inlet, y_inlet, X_MIN, X_MAX, H_CHAN, H_STEP, U_MAX,
        )
        u_ref = inlet_profile(y_inlet)
        inlet_loss = jnp.mean((u_in - u_ref) ** 2) + jnp.mean(v_in**2)

        _, _, p_out = eval_uvp_batch_bs(
            m, x_outlet, y_outlet, X_MIN, X_MAX, H_CHAN, H_STEP, U_MAX,
        )
        outlet_loss = jnp.mean(p_out**2)

        total = (
            1.0 * physics_loss
            + 10.0 * wall_loss
            + 10.0 * inlet_loss
            + 1.0 * outlet_loss
        )

        return total

    return loss_fn


# =========================================================
# sample points
# =========================================================


def make_collocation_points(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)

    print("Sampling collocation points...")

    x_col, y_col = sample_interior_bs(k1, N=12000, x_min=X_MIN, x_max=X_MAX, h=H_STEP, H=H_CHAN)
    x_wall, y_wall = sample_walls_bs(k2, N=4000, x_min=X_MIN, x_max=X_MAX, h=H_STEP, H=H_CHAN)
    x_inlet, y_inlet = sample_inlet_bs(k3, N=1500, x_min=X_MIN, h=H_STEP, H=H_CHAN)
    x_outlet, y_outlet = sample_outlet_bs(k4, N=1500, x_max=X_MAX, H=H_CHAN)

    print("Done.\n")

    return x_col, y_col, x_wall, y_wall, x_inlet, y_inlet, x_outlet, y_outlet


# =========================================================
# main
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Re", type=float, default=200.0, help="Reynolds number")
    args = parser.parse_args()
    Re = args.Re

    print(f"\nTraining backward step PINN at Re={Re}\n")

    RESULTS_DIR = f"results/backward_step/Re{int(Re)}"
    PLOTS_DIR = f"plots/backward_step/Re{int(Re)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    PRED_FILES = {
        "params": os.path.join(RESULTS_DIR, "params.npz"),
        "adam_losses": os.path.join(RESULTS_DIR, "adam_losses.txt"),
        "lbfgs_losses": os.path.join(RESULTS_DIR, "lbfgs_losses.txt"),
    }

    key = jax.random.key(0)
    model = BackwardStepPINN(
        widths=[64, 64, 64], key=key, activation=jax.nn.tanh, n_inputs=2,
    )

    x_col, y_col, x_wall, y_wall, x_inlet, y_inlet, x_outlet, y_outlet = make_collocation_points(key)

    if all(os.path.exists(PRED_FILES[k]) for k in PRED_FILES):
        print(f"Loading saved params from {RESULTS_DIR}/\n")
        load_model_state(model, PRED_FILES["params"])
        losses = np.loadtxt(PRED_FILES["adam_losses"]).tolist()
        lbfgs_losses = np.loadtxt(PRED_FILES["lbfgs_losses"]).tolist()
    else:
        # =====================================================
        # loss
        # =====================================================
        loss_fn = make_loss_fn(Re, x_col, y_col, x_wall, y_wall, x_inlet, y_inlet, x_outlet, y_outlet)

        # =====================================================
        # Adam (nnx.Optimizer)
        # =====================================================
        tx = optax.adam(learning_rate=1e-3)
        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

        @nnx.jit
        def adam_step(model, opt):
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            opt.update(model, grads)
            return loss

        losses = []
        print("Starting Adam training...\n")
        t0 = time.perf_counter()

        for epoch in range(3000):
            loss = adam_step(model, opt)
            losses.append(float(loss))

            if epoch % 500 == 0:
                print(f"Epoch {epoch:6d} | loss = {loss:.6e}")

        adam_time = time.perf_counter() - t0
        print(f"\nAdam complete in {adam_time:.1f} s\n")

        # =====================================================
        # L-BFGS (nnx.split/merge)
        # =====================================================
        print("Starting L-BFGS training...\n")

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
        t1 = time.perf_counter()

        for step in range(500):
            params, lbfgs_state, loss = lbfgs_train_step(params, lbfgs_state)
            lbfgs_losses.append(float(loss))

            if step % 200 == 0:
                print(f"Step {step:5d} | loss = {loss:.6e}")

            if step >= 200:
                rel_change = abs(lbfgs_losses[step] - lbfgs_losses[step - 200]) / abs(
                    lbfgs_losses[step - 200]
                )
                if rel_change < 1e-3:
                    print(f"\nEarly stopping at step {step}")
                    break

        nnx.update(model, params)

        lbfgs_time = time.perf_counter() - t1
        print(f"\nL-BFGS complete in {lbfgs_time:.1f} s\n")

        # =====================================================
        # save
        # =====================================================
        save_model(model, PRED_FILES["params"])
        np.savetxt(PRED_FILES["adam_losses"], np.array(losses))
        np.savetxt(PRED_FILES["lbfgs_losses"], np.array(lbfgs_losses))

        with open(os.path.join(RESULTS_DIR, "timing.json"), "w") as f:
            json.dump(
                {
                    "adam_s": adam_time,
                    "lbfgs_s": lbfgs_time,
                    "total_s": adam_time + lbfgs_time,
                    "adam_epochs": len(losses),
                    "lbfgs_steps": len(lbfgs_losses),
                },
                f,
                indent=2,
            )

    # =====================================================
    # plotting
    # =====================================================
    plot_loss(losses, lbfgs_losses, plots_dir=PLOTS_DIR)

    print("Evaluating final field...")
    Nx = 300
    Ny = 80
    x1d = jnp.linspace(X_MIN, X_MAX, Nx)
    y1d = jnp.linspace(0.0, H_CHAN, Ny)
    X, Y = jnp.meshgrid(x1d, y1d)
    mask = ~((X < 0.0) & (Y < H_STEP))

    u, v, p = eval_uvp_batch_bs(
        model, X.ravel(), Y.ravel(), X_MIN, X_MAX, H_CHAN, H_STEP, U_MAX,
    )
    u = jnp.where(mask, u.reshape(Ny, Nx), jnp.nan)
    v = jnp.where(mask, v.reshape(Ny, Nx), jnp.nan)
    p = jnp.where(mask, p.reshape(Ny, Nx), jnp.nan)

    plot_backward_step(X, Y, u, v, p, plots_dir=PLOTS_DIR)

    # =====================================================
    # reattachment length
    # =====================================================
    print("Computing reattachment length...")
    x_probe = jnp.linspace(0.0, X_MAX, 4000)
    y_probe = jnp.full(4000, 0.01)

    u_wall, _, _ = eval_uvp_batch_bs(
        model, x_probe, y_probe, X_MIN, X_MAX, H_CHAN, H_STEP, U_MAX,
    )

    transitions = jnp.diff(jnp.sign(u_wall))
    neg_to_pos = jnp.where(transitions > 0)[0]

    if len(neg_to_pos) == 0:
        x_r = None
        print("  No reattachment point found in domain.")
    else:
        i = int(neg_to_pos[0])
        x0, x1_p = float(x_probe[i]), float(x_probe[i + 1])
        u0, u1 = float(u_wall[i]), float(u_wall[i + 1])
        x_r = x0 - u0 * (x1_p - x0) / (u1 - u0)
        print(f"  x_r = {x_r:.4f}  ({x_r / H_STEP:.2f} step heights)")

    plot_recirculation(X, Y, u, x_r, H_STEP, plots_dir=PLOTS_DIR)

    print(f"\nFinished.\nPlots saved to:\n{PLOTS_DIR}\n")
