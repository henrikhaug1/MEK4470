import argparse
import json
import os
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.analytical import taylor_green_p, taylor_green_u, taylor_green_v
from src.plotting import plot_comparison, plot_loss
from src.pinn import (
    TaylorGreenPINN,
    eval_uvp_batch,
    loss_fn_tg,
    lbfgs_step,
    save_model,
    load_model_state,
    sample_interior,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Re", type=float, default=200.0, help="Reynolds number")
    args = parser.parse_args()
    Re = args.Re

    RESULTS_DIR = f"results/taylor_green/Re{Re:.0f}"
    PLOTS_DIR = f"plots/taylor_green/Re{Re:.0f}"
    PRED_FILES = {
        "params": os.path.join(RESULTS_DIR, "params.npz"),
        "adam_losses": os.path.join(RESULTS_DIR, "adam_losses.txt"),
        "lbfgs_losses": os.path.join(RESULTS_DIR, "lbfgs_losses.txt"),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    nu = 1.0 / Re

    key = jax.random.key(0)
    model = TaylorGreenPINN(widths=[64, 64, 64], key=key, activation=jax.nn.tanh)

    if all(os.path.exists(PRED_FILES[k]) for k in PRED_FILES):
        print(f"Loading saved params from {RESULTS_DIR}/")
        load_model_state(model, PRED_FILES["params"])
        adam_losses = np.loadtxt(PRED_FILES["adam_losses"]).tolist()
        lbfgs_losses = np.loadtxt(PRED_FILES["lbfgs_losses"]).tolist()
    else:
        (k1,) = jax.random.split(key, 1)
        x_col, y_col, t_col = sample_interior(k1, 10000, t_max=2.0)

        # Adam phase (nnx.Optimizer + nnx.value_and_grad)
        tx = optax.adam(1e-3)
        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

        @nnx.jit
        def adam_step(model, opt):
            def lf(m):
                return loss_fn_tg(m, x_col, y_col, t_col, Re)

            loss, grads = nnx.value_and_grad(lf)(model)
            opt.update(model, grads)
            return loss

        adam_losses = []
        t_adam = time.perf_counter()
        for epoch in range(8000):
            loss = adam_step(model, opt)
            adam_losses.append(float(loss))
            if epoch % 500 == 0:
                print(f"Adam  epoch {epoch:5d}  loss {loss:.4e}")
        adam_s = time.perf_counter() - t_adam
        print(f"Adam done in {adam_s:.1f}s")

        # L-BFGS phase (nnx.split/merge)
        graphdef, params, rest = nnx.split(model, nnx.Param, ...)

        def loss_of_params(p):
            m = nnx.merge(graphdef, p, rest)
            return loss_fn_tg(m, x_col, y_col, t_col, Re)

        lbfgs_opt = optax.lbfgs()
        lbfgs_state = lbfgs_opt.init(params)

        @jax.jit
        def lbfgs_train_step(params, opt_state):
            return lbfgs_step(
                graphdef, params, rest, opt_state, lbfgs_opt, loss_of_params
            )

        lbfgs_losses = []
        t_lbfgs = time.perf_counter()
        for step in range(5000):
            params, lbfgs_state, loss = lbfgs_train_step(params, lbfgs_state)
            lbfgs_losses.append(float(loss))
            if step % 200 == 0:
                print(f"L-BFGS step {step:5d}  loss {loss:.4e}")
                if step >= 200:
                    rel_change = abs(
                        lbfgs_losses[step] - lbfgs_losses[step - 200]
                    ) / abs(lbfgs_losses[step - 200])
                    if rel_change < 1e-3:
                        print(f"Early stopping at step {step}")
                        break
        lbfgs_s = time.perf_counter() - t_lbfgs
        print(f"L-BFGS done in {lbfgs_s:.1f}s")

        nnx.update(model, params)

        save_model(model, PRED_FILES["params"])
        np.savetxt(PRED_FILES["adam_losses"], np.array(adam_losses))
        np.savetxt(PRED_FILES["lbfgs_losses"], np.array(lbfgs_losses))
        with open(os.path.join(RESULTS_DIR, "timing.json"), "w") as f:
            json.dump(
                {
                    "adam_s": adam_s,
                    "lbfgs_s": lbfgs_s,
                    "total_s": adam_s + lbfgs_s,
                    "adam_epochs": len(adam_losses),
                    "lbfgs_steps": len(lbfgs_losses),
                },
                f,
            )
        print(f"Params saved to {RESULTS_DIR}/")

    plot_loss(adam_losses, lbfgs_losses, plots_dir=PLOTS_DIR)

    N = 100
    t_eval_list = [0.25, 0.5, 1.0, 2.0]
    x = jnp.linspace(0, 2 * jnp.pi, N)
    y = jnp.linspace(0, 2 * jnp.pi, N)
    X, Y = jnp.meshgrid(x, y)

    for t_eval in t_eval_list:
        T = jnp.full_like(X, t_eval)
        u_exact = taylor_green_u(X, Y, T, nu)
        v_exact = taylor_green_v(X, Y, T, nu)
        p_exact = taylor_green_p(X, Y, T, nu)

        u_pred, v_pred, p_pred = eval_uvp_batch(model, X.ravel(), Y.ravel(), T.ravel())
        plot_comparison(
            X,
            Y,
            u_exact,
            v_exact,
            p_exact,
            u_pred.reshape(N, N),
            v_pred.reshape(N, N),
            p_pred.reshape(N, N),
            t=t_eval,
            plots_dir=PLOTS_DIR,
        )
