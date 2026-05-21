"""
train_backward_step_surgical.py

Backward-facing-step PINN training script.

This is deliberately based on the closest-to-working parameter-pytree version,
with only surgical corrections:
  - corrected hard-BC ansatz imported from pinn_backward_step_surgical,
  - outlet p=0 soft loss retained,
  - soft zero-gradient outlet velocity loss added,
  - fuller BC diagnostics,
  - inlet/outlet mass-flux diagnostic,
  - component-wise loss logging,
  - more physical reattachment estimate from a near-wall shear proxy,
  - tqdm progress bars.
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import trange

from src.pinn_backward_step_surgical import (
    BackwardStepPINN,
    bottom_wall_shear_proxy_bs,
    eval_uvp_batch_bs,
    outlet_velocity_gradients_bs,
    pack_params,
    residuals_batch_bs,
    train_step_lbfgs,
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
    # Kept from the closest-to-working setup.
    x_r_est = 0.025 * Re
    x_max = max(20.0, x_r_est * 2.5)
    return dict(
        x_max=x_max,
        widths=[32, 32, 32, 32],  # kept unchanged on purpose
        n_col=3000,  # kept unchanged on purpose
        adam_epochs=10000,
        lr_start=1e-3,
        lr_end=1e-5,
        lbfgs_steps=1000,
        x_recirc=min(x_r_est * 1.5, x_max),
        w_outlet_p=1.0,
        w_outlet_grad=0.10,  # soft zero-gradient velocity outlet penalty
    )


# ============================================================
# Re-adaptive samplers
# ============================================================


def sample_interior(key, n_col, x_max, x_recirc):
    """
    Point budget:
      10% – corner enrichment
      15% – shear-layer strip
      30% – recirculation zone
      10% – pre-step channel
      35% – global domain
    """
    SHEAR_HALF = 0.2
    N_corner = n_col // 10
    N_shear = (n_col * 15) // 100
    N_recirc = (n_col * 30) // 100
    N_inlet = (n_col * 10) // 100
    N_rest = n_col - N_corner - N_shear - N_recirc - N_inlet
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    # corner enrichment around (0, H_STEP)
    pts_c = jax.random.uniform(k1, (N_corner * 4, 2))
    xc = pts_c[:, 0] * 2.0 - 0.5
    yc = pts_c[:, 1] * 1.0 + 0.5
    valid_c = ~((xc < 0.0) & (yc < H_STEP))
    xc, yc = xc[valid_c][:N_corner], yc[valid_c][:N_corner]

    # shear-layer strip
    pts_s = jax.random.uniform(k2, (N_shear, 2))
    xs = pts_s[:, 0] * x_recirc
    ys = pts_s[:, 1] * (2.0 * SHEAR_HALF) + (H_STEP - SHEAR_HALF)

    # recirculation zone
    pts1 = jax.random.uniform(k3, (N_recirc, 2))
    x1 = pts1[:, 0] * x_recirc
    y1 = pts1[:, 1] * H_CHAN

    # pre-step inlet channel
    pts_i = jax.random.uniform(k4, (N_inlet, 2))
    xi = pts_i[:, 0] * abs(X_MIN) + X_MIN
    yi = pts_i[:, 1] * (H_CHAN - H_STEP) + H_STEP

    # global zone
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


def make_loss_fns(Re, x_col, y_col, x_max, w_outlet_p, w_outlet_grad):
    """
    Total loss =
        PDE residual
        + w_outlet_p    * mean(p_out^2)
        + w_outlet_grad * mean(u_x_out^2 + v_x_out^2)

    The pressure outlet penalty is retained from the closest-to-working version.
    The velocity-gradient term is a conservative soft approximation of a
    zero-normal-gradient outflow condition.
    """
    N_OUT = 200
    x_outlet = jnp.full(N_OUT, x_max)
    y_outlet = jnp.linspace(0.0, H_CHAN, N_OUT)

    def loss_terms(params):
        cont, mom_x, mom_y = residuals_batch_bs(
            params,
            jax.nn.tanh,
            Re,
            x_col,
            y_col,
            X_MIN,
            x_max,
            H_CHAN,
            H_STEP,
            U_MEAN,
        )

        cont_loss = jnp.mean(cont**2)
        mom_x_loss = jnp.mean(mom_x**2)
        mom_y_loss = jnp.mean(mom_y**2)
        physics_loss = cont_loss + mom_x_loss + mom_y_loss

        _, _, p_out = eval_uvp_batch_bs(
            params,
            jax.nn.tanh,
            x_outlet,
            y_outlet,
            X_MIN,
            x_max,
            H_CHAN,
            H_STEP,
            U_MEAN,
        )
        outlet_p_loss = jnp.mean(p_out**2)

        ux_out, vx_out = outlet_velocity_gradients_bs(
            params,
            jax.nn.tanh,
            x_outlet,
            y_outlet,
            X_MIN,
            x_max,
            H_CHAN,
            H_STEP,
            U_MEAN,
        )
        outlet_grad_loss = jnp.mean(ux_out**2 + vx_out**2)

        total = (
            physics_loss + w_outlet_p * outlet_p_loss + w_outlet_grad * outlet_grad_loss
        )

        return (
            total,
            physics_loss,
            cont_loss,
            mom_x_loss,
            mom_y_loss,
            outlet_p_loss,
            outlet_grad_loss,
        )

    def loss_fn(params):
        return loss_terms(params)[0]

    return loss_fn, loss_terms


def _format_loss_terms(terms):
    (
        total,
        physics,
        cont,
        mom_x,
        mom_y,
        outlet_p,
        outlet_grad,
    ) = [float(v) for v in terms]

    return (
        f"total={total:.3e} | phys={physics:.3e} "
        f"(cont={cont:.3e}, mx={mom_x:.3e}, my={mom_y:.3e}) | "
        f"p_out={outlet_p:.3e} | grad_out={outlet_grad:.3e}"
    )


# ============================================================
# BC and conservation sanity checks
# ============================================================


def check_bcs(params, activation, x_max, n=300):
    """
    Verify the hard velocity boundaries and report soft outlet diagnostics.

    Velocity BCs should be near machine zero.
    Outlet p and outlet velocity-gradient errors are soft diagnostics.
    """
    # inlet
    y_in = jnp.linspace(H_STEP, H_CHAN, n)
    x_in = jnp.full(n, X_MIN)
    u_in, v_in, _ = eval_uvp_batch_bs(
        params, activation, x_in, y_in, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    u_ref = U_MEAN * 6.0 * (y_in - H_STEP) * (H_CHAN - y_in) / (H_CHAN - H_STEP) ** 2
    inlet_u_err = float(jnp.sqrt(jnp.mean((u_in - u_ref) ** 2)))
    inlet_v_err = float(jnp.sqrt(jnp.mean(v_in**2)))

    # outlet pressure and velocity gradient
    y_out = jnp.linspace(0.0, H_CHAN, n)
    x_out = jnp.full(n, x_max)
    u_out, _, p_out = eval_uvp_batch_bs(
        params, activation, x_out, y_out, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    ux_out, vx_out = outlet_velocity_gradients_bs(
        params, activation, x_out, y_out, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    outlet_p_err = float(jnp.sqrt(jnp.mean(p_out**2)))
    outlet_ux_err = float(jnp.sqrt(jnp.mean(ux_out**2)))
    outlet_vx_err = float(jnp.sqrt(jnp.mean(vx_out**2)))

    # downstream bottom wall
    x_bot = jnp.linspace(0.0, x_max, n)
    y_bot = jnp.zeros(n)
    u_bot, v_bot, _ = eval_uvp_batch_bs(
        params, activation, x_bot, y_bot, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    bottom_u_err = float(jnp.sqrt(jnp.mean(u_bot**2)))
    bottom_v_err = float(jnp.sqrt(jnp.mean(v_bot**2)))

    # top wall
    x_top = jnp.linspace(X_MIN, x_max, n)
    y_top = jnp.full(n, H_CHAN)
    u_top, v_top, _ = eval_uvp_batch_bs(
        params, activation, x_top, y_top, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    top_u_err = float(jnp.sqrt(jnp.mean(u_top**2)))
    top_v_err = float(jnp.sqrt(jnp.mean(v_top**2)))

    # upstream lower wall
    x_floor = jnp.linspace(X_MIN, 0.0, n)
    y_floor = jnp.full(n, H_STEP)
    u_floor, v_floor, _ = eval_uvp_batch_bs(
        params, activation, x_floor, y_floor, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    floor_u_err = float(jnp.sqrt(jnp.mean(u_floor**2)))
    floor_v_err = float(jnp.sqrt(jnp.mean(v_floor**2)))

    # vertical step face
    y_step = jnp.linspace(0.0, H_STEP, n)
    x_step = jnp.zeros(n)
    u_step, v_step, _ = eval_uvp_batch_bs(
        params, activation, x_step, y_step, X_MIN, x_max, H_CHAN, H_STEP, U_MEAN
    )
    step_u_err = float(jnp.sqrt(jnp.mean(u_step**2)))
    step_v_err = float(jnp.sqrt(jnp.mean(v_step**2)))

    # mass flux diagnostic: Q_in should be close to Q_out
    q_in = float(jnp.trapezoid(u_in, y_in))
    q_out = float(jnp.trapezoid(u_out, y_out))
    q_rel_err = abs(q_out - q_in) / max(abs(q_in), 1e-14)

    print("\n── BC / outlet / conservation sanity check ───────────────────")
    print(
        f"  Inlet profile RMS u : {inlet_u_err:.2e}   inlet RMS v : {inlet_v_err:.2e}  [hard]"
    )
    print(
        f"  Top wall RMS u      : {top_u_err:.2e}   top RMS v    : {top_v_err:.2e}  [hard]"
    )
    print(
        f"  Upstream floor RMS u: {floor_u_err:.2e}   RMS v        : {floor_v_err:.2e}  [hard]"
    )
    print(
        f"  Step face RMS u     : {step_u_err:.2e}   RMS v        : {step_v_err:.2e}  [hard]"
    )
    print(
        f"  Bottom wall RMS u   : {bottom_u_err:.2e}   RMS v       : {bottom_v_err:.2e}  [hard]"
    )
    print(f"  Outlet RMS p        : {outlet_p_err:.2e}                         [soft]")
    print(
        f"  Outlet RMS u_x      : {outlet_ux_err:.2e}   RMS v_x     : {outlet_vx_err:.2e}  [soft]"
    )
    print(f"  Flux Q_in           : {q_in:.6f}")
    print(f"  Flux Q_out          : {q_out:.6f}")
    print(f"  Relative flux error : {q_rel_err:.2e}")
    print("──────────────────────────────────────────────────────────────\n")


# ============================================================
# Reattachment diagnostic
# ============================================================


def estimate_reattachment_from_shear(params, activation, x_max, n_probe=6000):
    """
    Estimate primary reattachment from a near-wall shear-sign crossing.

    The shear proxy is:
        u(x, y_eps) / y_eps
    on the downstream bottom wall.
    """
    x_probe = jnp.linspace(1e-4, x_max, n_probe)
    shear_proxy = bottom_wall_shear_proxy_bs(
        params,
        activation,
        x_probe,
        X_MIN,
        x_max,
        H_CHAN,
        H_STEP,
        U_MEAN,
        y_eps=1e-3,
    )

    shear_np = np.asarray(shear_proxy)
    x_np = np.asarray(x_probe)

    negative = shear_np < 0.0
    min_shear = float(np.min(shear_np))

    print(
        f"  min near-wall shear proxy : {min_shear:.4e}  "
        f"({'negative region detected' if min_shear < 0 else 'no negative region detected'})"
    )

    if not np.any(negative):
        return None

    # First physically meaningful negative-to-positive crossing after the
    # solution has entered a negative-shear region.
    crossings = np.where(negative[:-1] & (~negative[1:]))[0]
    if len(crossings) == 0:
        print(
            "  Negative near-wall shear persists to the outlet; no reattachment found."
        )
        return None

    i = int(crossings[0])
    x0, x1 = x_np[i], x_np[i + 1]
    s0, s1 = shear_np[i], shear_np[i + 1]
    if abs(s1 - s0) < 1e-14:
        return float(x0)

    return float(x0 - s0 * (x1 - x0) / (s1 - s0))


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

    print(f"\nBackward step PINN — surgical correction  |  Re = {Re}")
    print(f"  Domain      : x ∈ [{X_MIN}, {X_MAX:.1f}]  y ∈ [0, {H_CHAN}]")
    print(f"  Network     : {cfg['widths']}")
    print(f"  Adam epochs : {cfg['adam_epochs']}   L-BFGS steps: {cfg['lbfgs_steps']}")
    print(f"  Collocation : {cfg['n_col']}   x_recirc ≈ {cfg['x_recirc']:.1f} H")
    print(
        f"  Outlet loss : W_p={cfg['w_outlet_p']:.2f}, W_grad={cfg['w_outlet_grad']:.2f}\n"
    )

    RESULTS_DIR = f"results/backward_step_surgical/Re{int(Re)}"
    PLOTS_DIR = f"plots/backward_step_surgical/Re{int(Re)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    activation = jax.nn.tanh

    PRED_FILES = {
        "params": os.path.join(RESULTS_DIR, "params.npz"),
        "adam_losses": os.path.join(RESULTS_DIR, "adam_losses.txt"),
        "lbfgs_losses": os.path.join(RESULTS_DIR, "lbfgs_losses.txt"),
    }

    if all(os.path.exists(PRED_FILES[k]) for k in PRED_FILES):
        print(f"Loading saved params from {RESULTS_DIR}/\n")
        data = np.load(PRED_FILES["params"])
        n = sum(1 for k in data if k.startswith("W"))
        ws = tuple(jnp.array(data[f"W{i}"]) for i in range(n))
        bs = tuple(jnp.array(data[f"b{i}"]) for i in range(n))
        params = (ws, bs)
        adam_losses = np.loadtxt(PRED_FILES["adam_losses"]).tolist()
        lbfgs_losses = np.loadtxt(PRED_FILES["lbfgs_losses"]).tolist()
        check_bcs(params, activation, X_MAX)

    else:
        key = jax.random.key(0)
        model = BackwardStepPINN(
            widths=cfg["widths"],
            key=key,
            activation=activation,
            n_inputs=2,
        )
        params = pack_params(model)

        _, k1 = jax.random.split(key, 2)
        print("Sampling collocation points...")
        x_col, y_col = sample_interior(k1, cfg["n_col"], X_MAX, cfg["x_recirc"])
        print(f"  interior: {len(x_col)}\n")

        loss_fn, loss_terms_fn = make_loss_fns(
            Re,
            x_col,
            y_col,
            X_MAX,
            cfg["w_outlet_p"],
            cfg["w_outlet_grad"],
        )
        loss_terms_eval = jax.jit(loss_terms_fn)

        # --------------------------------------------------------
        # Adam
        # --------------------------------------------------------
        lr_schedule = optax.cosine_decay_schedule(
            init_value=cfg["lr_start"],
            decay_steps=cfg["adam_epochs"],
            alpha=cfg["lr_end"] / cfg["lr_start"],
        )
        optimizer = optax.adam(learning_rate=lr_schedule)
        opt_state = optimizer.init(params)

        @jax.jit
        def adam_step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_state = optimizer.update(grads, opt_state)
            return optax.apply_updates(params, updates), new_state, loss

        adam_losses = []
        print("Starting Adam training...\n")
        t0 = time.perf_counter()

        adam_bar = trange(cfg["adam_epochs"], desc="Adam", unit="epoch")
        for epoch in adam_bar:
            params, opt_state, loss = adam_step(params, opt_state)
            loss_float = float(loss)
            adam_losses.append(loss_float)
            adam_bar.set_postfix(loss=f"{loss_float:.3e}")

            if epoch % 500 == 0:
                terms = loss_terms_eval(params)
                print(f"Epoch {epoch:6d} | {_format_loss_terms(terms)}")

            if epoch > cfg["adam_epochs"] // 2:
                window = 1000
                if len(adam_losses) >= 2 * window:
                    recent = np.mean(adam_losses[-window:])
                    older = np.mean(adam_losses[-2 * window : -window])
                    if abs(recent - older) / max(abs(older), 1e-14) < 5e-4:
                        print(f"\nEarly stopping at epoch {epoch}")
                        break

        adam_time = time.perf_counter() - t0
        print(f"\nAdam complete in {adam_time:.1f} s\n")
        check_bcs(params, activation, X_MAX)

        # --------------------------------------------------------
        # L-BFGS
        # --------------------------------------------------------
        lbfgs_optimizer = optax.lbfgs()
        lbfgs_state = lbfgs_optimizer.init(params)
        lbfgs_losses = []

        @jax.jit
        def compiled_lbfgs_step(params, opt_state):
            return train_step_lbfgs(
                params,
                opt_state,
                lbfgs_optimizer,
                loss_fn,
            )

        print("Starting L-BFGS training...\n")
        t1 = time.perf_counter()

        lbfgs_bar = trange(cfg["lbfgs_steps"], desc="L-BFGS", unit="step")
        for step in lbfgs_bar:
            params, lbfgs_state, loss = train_step_lbfgs(
                params,
                lbfgs_state,
                lbfgs_optimizer,
                value_and_grad_fn,
            )
            loss_float = float(loss)
            lbfgs_losses.append(loss_float)
            lbfgs_bar.set_postfix(loss=f"{loss_float:.3e}")

            if step % 200 == 0:
                terms = loss_terms_eval(params)
                print(f"Step {step:5d} | {_format_loss_terms(terms)}")

            if step >= 200:
                rel = abs(lbfgs_losses[step] - lbfgs_losses[step - 200]) / max(
                    abs(lbfgs_losses[step - 200]),
                    1e-14,
                )
                if rel < 1e-3:
                    print(f"\nEarly stopping at step {step}")
                    break

        lbfgs_time = time.perf_counter() - t1
        print(f"\nL-BFGS complete in {lbfgs_time:.1f} s\n")
        check_bcs(params, activation, X_MAX)

        # --------------------------------------------------------
        # Save
        # --------------------------------------------------------
        ws, bs = params
        np.savez(
            PRED_FILES["params"],
            **{f"W{i}": np.array(W) for i, W in enumerate(ws)},
            **{f"b{i}": np.array(b) for i, b in enumerate(bs)},
        )
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
    # Evaluation + plotting
    # ============================================================
    plot_loss(adam_losses, lbfgs_losses, plots_dir=PLOTS_DIR)

    print("Evaluating final field...")
    Nx, Ny = 400, 80
    x1d = jnp.linspace(X_MIN, X_MAX, Nx)
    y1d = jnp.linspace(0.0, H_CHAN, Ny)
    X, Y = jnp.meshgrid(x1d, y1d)
    mask = ~((X < 0.0) & (Y < H_STEP))

    u, v, p = eval_uvp_batch_bs(
        params,
        activation,
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

    print("Computing reattachment length from near-wall shear proxy...")
    x_r = estimate_reattachment_from_shear(params, activation, X_MAX)
    if x_r is None:
        print("  No reattachment point found.")
    else:
        print(f"  x_r = {x_r:.4f}  ({x_r / H_STEP:.2f} step heights)")

    plot_recirculation(X, Y, u, x_r, H_STEP, plots_dir=PLOTS_DIR)

    print(f"\nFinished.\nPlots saved to:\n{PLOTS_DIR}\n")
