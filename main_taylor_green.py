import os

import jax
import jax.numpy as jnp
import numpy as np

from src.analytical import *
from src.plotting import *
from src.pinn import *

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
PRED_FILES = {
    "params": os.path.join(RESULTS_DIR, "params.npz"),
    "adam_losses": os.path.join(RESULTS_DIR, "adam_losses.txt"),
    "lbfgs_losses": os.path.join(RESULTS_DIR, "lbfgs_losses.txt"),
}

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    Re = 200.0
    nu = 1 / Re  # viscosity
    N_coll = 3000  # Collocation points (physics)
    N_ic = 500  # Initial condition points
    N = 50
    t = 0.5
    x = jnp.linspace(0, 2 * jnp.pi, N)
    y = jnp.linspace(0, 2 * jnp.pi, N)
    X, Y = jnp.meshgrid(x, y)

    t_eval_list = [0.25, 0.5, 1.0, 2.0]
    N = 100
    x = jnp.linspace(0, 2 * jnp.pi, N)
    y = jnp.linspace(0, 2 * jnp.pi, N)
    X, Y = jnp.meshgrid(x, y)

    activation = jax.nn.tanh
    pred_keys = ("params", "adam_losses", "lbfgs_losses")
    if all(os.path.exists(PRED_FILES[k]) for k in pred_keys):
        print("Loading saved params from results/")
        data = np.load(PRED_FILES["params"])
        n = sum(1 for k in data if k.startswith("W"))
        ws = tuple(jnp.array(data[f"W{i}"]) for i in range(n))
        bs = tuple(jnp.array(data[f"b{i}"]) for i in range(n))
        params = (ws, bs)
        adam_losses = np.loadtxt(PRED_FILES["adam_losses"]).tolist()
        lbfgs_losses = np.loadtxt(PRED_FILES["lbfgs_losses"]).tolist()
    else:
        # Create PINN instance
        key = jax.random.key(0)
        model = NavierStokesPINN(
            widths=[64, 64, 64], key=key, Re=200.0, activation=jax.nn.tanh
        )
        params = pack_params(model)

        # sample points once
        k1, k2, k3, k4 = jax.random.split(key, 4)
        x_col, y_col, t_col = sample_interior(k1, 10000, t_max=2 * jnp.pi)
        x_ic, y_ic, t_ic = sample_ic(k2, 2000)
        x_data, y_data, t_data = sample_interior(k3, 2000, t_max=2 * jnp.pi)

        # Adam phase
        adam_optimizer = optax.adam(1e-3)
        adam_state = adam_optimizer.init(params)
        adam_losses = []
        for epoch in range(10000):
            params, adam_state, loss = train_step_adam(
                params,
                adam_state,
                adam_optimizer,
                activation,
                x_col,
                y_col,
                t_col,
                x_ic,
                y_ic,
                t_ic,
                x_data,
                y_data,
                t_data,
                Re,
            )
            adam_losses.append(float(loss))
            if epoch % 500 == 0:
                print(f"Adam  epoch {epoch:5d}  loss {loss:.4e}")

        # L-BFGS phase
        lbfgs_optimizer = optax.lbfgs()
        lbfgs_state = lbfgs_optimizer.init(params)
        value_and_grad_fn = jax.value_and_grad(
            make_loss_for_lbfgs(
                activation,
                x_col,
                y_col,
                t_col,
                x_ic,
                y_ic,
                t_ic,
                x_data,
                y_data,
                t_data,
                Re,
            )
        )
        lbfgs_losses = []
        for step in range(2000):
            params, lbfgs_state, loss = train_step_lbfgs(
                params, lbfgs_state, lbfgs_optimizer, value_and_grad_fn
            )
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

        ws, bs = params
        np.savez(
            PRED_FILES["params"],
            **{f"W{i}": np.array(W) for i, W in enumerate(ws)},
            **{f"b{i}": np.array(b) for i, b in enumerate(bs)},
        )
        np.savetxt(PRED_FILES["adam_losses"], np.array(adam_losses))
        np.savetxt(PRED_FILES["lbfgs_losses"], np.array(lbfgs_losses))
        print(f"Params saved to {RESULTS_DIR}/")

    plot_loss(adam_losses, lbfgs_losses, plots_dir=PLOTS_DIR)

    for t_eval in t_eval_list:
        T = jnp.full_like(X, t_eval)
        u_exact = taylor_green_u(X, Y, T, nu)
        v_exact = taylor_green_v(X, Y, T, nu)
        p_exact = taylor_green_p(X, Y, T, nu)

        x_flat = X.ravel()
        y_flat = Y.ravel()
        t_flat = T.ravel()
        u_pred_t, v_pred_t, p_pred_t = eval_uvp_batch(
            params, activation, x_flat, y_flat, t_flat
        )
        u_pred_t = u_pred_t.reshape(N, N)
        v_pred_t = v_pred_t.reshape(N, N)
        p_pred_t = p_pred_t.reshape(N, N)

        plot_comparison(
            X,
            Y,
            u_exact,
            v_exact,
            p_exact,
            u_pred_t,
            v_pred_t,
            p_pred_t,
            t=t_eval,
            plots_dir=PLOTS_DIR,
        )
