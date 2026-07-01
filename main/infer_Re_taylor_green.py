import os
import time
import argparse

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")

from src.pinn import (
    TaylorGreenPINN,
    load_model_state,
    eval_uvp_batch,
    residuals_batch,
    residual_parts,
    sample_interior,
)
from src.analytical import taylor_green_u, taylor_green_v
from src.plotting import (
    plot_Re_convergence,
    plot_inference_loss,
)

jax.config.update("jax_enable_x64", True)

# ============================================================
# setup  (Taylor-Green vortex, hard IC ansatz)
#   Domain [0, 2π]^2,  t in [0, T_MAX],  nu = 1/Re  (so Re = 1/nu)
# ============================================================

T_MAX = 2.0
FWD_WIDTHS = [64, 64, 64]   # forward (synthetic-source) net, matches train_taylor_green.py
WIDTHS = [32, 32, 32]       # inference net (trained from scratch)

# Forward-PINN checkpoints trained at each Re (used for synthetic observations).
FORWARD_CKPTS = {
    200: "results/taylor_green/Re200/params.npz",
    300: "results/taylor_green/Re300/params.npz",
    1000: "results/taylor_green/Re1000/params.npz",
}


# ============================================================
# observation sources
# ============================================================


def load_analytic_observations(Re_true, x_obs, y_obs, t_obs):
    """Exact Taylor-Green velocity at (x, y, t) for the given Re (nu = 1/Re)."""
    nu = 1.0 / Re_true
    u = taylor_green_u(x_obs, y_obs, t_obs, nu)
    v = taylor_green_v(x_obs, y_obs, t_obs, nu)
    return jnp.asarray(u), jnp.asarray(v)


def load_synthetic_observations(Re_true, x_obs, y_obs, t_obs):
    """Evaluate the trained forward Taylor-Green PINN at Re_true on the obs points."""
    ckpt = FORWARD_CKPTS.get(int(Re_true))
    if ckpt is None:
        raise ValueError(f"No forward checkpoint configured for Re={Re_true}")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"Forward checkpoint not found: {ckpt}\n"
            f"Train it first with main/train_taylor_green.py --Re {Re_true}"
        )
    fwd = TaylorGreenPINN(
        widths=FWD_WIDTHS, key=jax.random.key(0), activation=jax.nn.tanh
    )
    load_model_state(fwd, ckpt)
    u, v, _ = eval_uvp_batch(fwd, x_obs, y_obs, t_obs, t_max=T_MAX)
    return jnp.asarray(u), jnp.asarray(v)


# ============================================================
# loss weights
# ============================================================

W_PHYS = 1.0
W_DATA = 1.0   # scaled by --w_data
WARMUP_EPOCHS = 1000


def make_warmup_loss_fn(x_obs, y_obs, t_obs, u_obs, v_obs, w_data):
    """Warm-up loss: data only (the IC is baked into the ansatz)."""
    def loss_fn(model):
        u_pred, v_pred, _ = eval_uvp_batch(model, x_obs, y_obs, t_obs, t_max=T_MAX)
        data = jnp.mean((u_pred - u_obs) ** 2) + jnp.mean((v_pred - v_obs) ** 2)
        return w_data * W_DATA * data

    return loss_fn


def make_loss_fn(x_col, y_col, t_col, x_obs, y_obs, t_obs, u_obs, v_obs, w_data):
    """Full joint loss: physics + data."""
    def loss_fn(model, log_Re):
        Re = jnp.exp(log_Re)
        cont, mom_x, mom_y = residuals_batch(model, Re, x_col, y_col, t_col, T_MAX)
        phys = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)
        u_pred, v_pred, _ = eval_uvp_batch(model, x_obs, y_obs, t_obs, t_max=T_MAX)
        data = jnp.mean((u_pred - u_obs) ** 2) + jnp.mean((v_pred - v_obs) ** 2)
        return W_PHYS * phys + w_data * W_DATA * data

    return loss_fn


# ============================================================
# joint-phase drivers
# ============================================================

RE_STABLE_WINDOW = 200
RE_STABLE_TOL = 0.005


def run_free(model, log_Re, args, x_col, y_col, t_col,
             x_obs, y_obs, t_obs, u_obs, v_obs, Re_history, loss_history):
    """Re is a free optimiser variable (biased — drifts high)."""
    net_schedule = optax.cosine_decay_schedule(args.net_lr, args.epochs, alpha=1e-2)
    re_schedule = optax.cosine_decay_schedule(args.re_lr, args.epochs, alpha=1e-2)
    net_opt = nnx.Optimizer(model, optax.adam(learning_rate=net_schedule), wrt=nnx.Param)
    re_optimizer = optax.adam(learning_rate=re_schedule)
    re_state = re_optimizer.init(log_Re)

    loss_fn = make_loss_fn(
        x_col, y_col, t_col, x_obs, y_obs, t_obs, u_obs, v_obs, w_data=args.w_data
    )

    @nnx.jit
    def train_step(model, net_opt, log_Re, re_state):
        loss, net_grads = nnx.value_and_grad(lambda m: loss_fn(m, log_Re))(model)
        re_grad = jax.grad(lambda lr: loss_fn(model, lr))(log_Re)
        net_opt.update(model, net_grads)
        re_updates, new_re_state = re_optimizer.update(re_grad, re_state)
        new_log_Re = optax.apply_updates(log_Re, re_updates)
        return new_log_Re, new_re_state, loss

    print("Starting joint inference (free Re)...\n")
    with tqdm(total=args.epochs, desc="Infer Re", unit="epoch", dynamic_ncols=True) as pbar:
        for epoch in range(args.epochs):
            log_Re, re_state, loss = train_step(model, net_opt, log_Re, re_state)
            Re_current = float(jnp.exp(log_Re))
            Re_history.append(Re_current)
            loss_history.append(float(loss))
            if epoch % 20 == 0:
                pbar.set_postfix(loss=f"{float(loss):.3e}", Re=f"{Re_current:.1f}")
            pbar.update(1)

            if not args.no_early_stop and epoch >= RE_STABLE_WINDOW and epoch % 20 == 0:
                window = Re_history[-RE_STABLE_WINDOW:]
                rel_std = np.std(window) / (np.mean(window) + 1e-12)
                if rel_std < RE_STABLE_TOL:
                    pbar.write(f"Re converged at epoch {epoch} "
                               f"(Re={Re_current:.2f}, rel_std={rel_std:.4f})")
                    break
    return Re_history[-1]


def run_closed_form(model, args, x_col, y_col, t_col,
                    x_obs, y_obs, t_obs, u_obs, v_obs, Re_history, loss_history):
    """Sturdy mode: Re slaved to the field's least-squares viscous balance.

        nu* = Σ(a·b)/Σ(b·b),   a = u_t + u·∇u + ∇p,  b = ∇²u,   Re = 1/nu*.
    """
    net_schedule = optax.cosine_decay_schedule(args.net_lr, args.epochs, alpha=1e-2)
    net_opt = nnx.Optimizer(model, optax.adam(learning_rate=net_schedule), wrt=nnx.Param)

    # nu = 1/Re,  Re in [1, 1e5]
    NU_MIN, NU_MAX = 1.0 / 1e5, 1.0
    beta = args.re_ema

    n_val = max(1, len(x_obs) // 5)
    xv, yv, tv, uv, vv = (x_obs[:n_val], y_obs[:n_val], t_obs[:n_val],
                          u_obs[:n_val], v_obs[:n_val])
    xt, yt, tt, ut, vt = (x_obs[n_val:], y_obs[n_val:], t_obs[n_val:],
                          u_obs[n_val:], v_obs[n_val:])

    def cf_loss(m, nu_use):
        cont, a_x, b_x, a_y, b_y = residual_parts(m, x_col, y_col, t_col, T_MAX)
        mom_x = a_x - nu_use * b_x
        mom_y = a_y - nu_use * b_y
        phys = jnp.mean(cont**2) + jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        num = jnp.sum(a_x * b_x + a_y * b_y)
        den = jnp.sum(b_x**2 + b_y**2) + 1e-12
        nu_star = num / den

        u_pred, v_pred, _ = eval_uvp_batch(m, xt, yt, tt, t_max=T_MAX)
        data = jnp.mean((u_pred - ut) ** 2) + jnp.mean((v_pred - vt) ** 2)

        loss = W_PHYS * phys + args.w_data * W_DATA * data
        return loss, jax.lax.stop_gradient(nu_star)

    @nnx.jit
    def cf_step(model, net_opt, nu_use):
        (loss, nu_star), grads = nnx.value_and_grad(cf_loss, has_aux=True)(model, nu_use)
        net_opt.update(model, grads)
        return loss, nu_star

    @nnx.jit
    def val_mse(m):
        u_p, v_p, _ = eval_uvp_batch(m, xv, yv, tv, t_max=T_MAX)
        return jnp.mean((u_p - uv) ** 2) + jnp.mean((v_p - vv) ** 2)

    _, nu0 = cf_loss(model, jnp.asarray(1.0 / args.Re_init))
    nu_ema = float(np.clip(float(nu0), NU_MIN, NU_MAX))
    print(f"Starting closed-form Re inference  (initial nu*={nu_ema:.4e} -> "
          f"Re={1.0 / nu_ema:.1f})\n")

    best_val = np.inf
    Re_at_best = 1.0 / nu_ema
    best_epoch = 0
    VAL_EVERY = 20
    VAL_PATIENCE = 40

    with tqdm(total=args.epochs, desc="Infer Re (closed-form)", unit="epoch",
              dynamic_ncols=True) as pbar:
        for epoch in range(args.epochs):
            loss, nu_star = cf_step(model, net_opt, jnp.asarray(nu_ema))
            nu_star = float(np.clip(float(nu_star), NU_MIN, NU_MAX))
            nu_ema = beta * nu_ema + (1.0 - beta) * nu_star
            Re_current = 1.0 / nu_ema
            Re_history.append(Re_current)
            loss_history.append(float(loss))

            if epoch % VAL_EVERY == 0:
                vm = float(val_mse(model))
                if vm < best_val:
                    best_val, Re_at_best, best_epoch = vm, Re_current, epoch
                pbar.set_postfix(loss=f"{float(loss):.3e}", Re=f"{Re_current:.1f}",
                                 val=f"{vm:.2e}", Re_best=f"{Re_at_best:.1f}")
                if (not args.no_early_stop
                        and epoch - best_epoch >= VAL_PATIENCE * VAL_EVERY):
                    pbar.write(f"Val-based stop at epoch {epoch}  "
                               f"(best val @ {best_epoch}, Re={Re_at_best:.2f})")
                    break
            pbar.update(1)

    print(f"  Best validation field at epoch {best_epoch}: "
          f"val_mse={best_val:.3e}  ->  Re={Re_at_best:.2f}")
    return Re_at_best


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Re_true", type=float, default=200.0)
    parser.add_argument("--Re_init", type=float, default=1000.0)
    parser.add_argument("--n_obs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--net_lr", type=float, default=1e-3)
    parser.add_argument("--re_lr", type=float, default=1e-2)
    parser.add_argument("--w_data", type=float, default=1000.0)
    parser.add_argument(
        "--obs_source", choices=["analytic", "synthetic"], default="analytic",
        help="analytic (default): exact Taylor-Green velocity at Re_true. "
             "synthetic: evaluate the trained forward PINN at Re_true.",
    )
    parser.add_argument(
        "--re_mode", choices=["free", "closed_form"], default="closed_form",
        help="free: Re is a free optimiser variable (biased). closed_form (default, "
             "sturdy): Re slaved to the field's least-squares viscous balance.",
    )
    parser.add_argument("--re_ema", type=float, default=0.99)
    parser.add_argument("--no_early_stop", action="store_true")
    args = parser.parse_args()

    suffix = ""
    if args.re_mode == "free":
        suffix += "_free"
    if args.obs_source == "synthetic":
        suffix += "_synthetic"
    RESULTS_DIR = f"results/infer_Re_taylor_green{suffix}"
    PLOTS_DIR = f"plots/infer_Re_taylor_green{suffix}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- sample observation locations in [0,2π]^2 x [0,T_MAX] ----
    key_obs = jax.random.key(99)
    pts = jax.random.uniform(key_obs, (args.n_obs, 3))
    x_obs = pts[:, 0] * 2.0 * jnp.pi
    y_obs = pts[:, 1] * 2.0 * jnp.pi
    t_obs = pts[:, 2] * T_MAX

    if args.obs_source == "synthetic":
        print(f"\nGenerating synthetic Re={args.Re_true} observations from forward PINN "
              f"{FORWARD_CKPTS[int(args.Re_true)]}")
        u_obs, v_obs = load_synthetic_observations(args.Re_true, x_obs, y_obs, t_obs)
        print(f"  Generated {len(x_obs)} synthetic observation points")
    else:
        print(f"\nGenerating analytic Re={args.Re_true} Taylor-Green observations")
        u_obs, v_obs = load_analytic_observations(args.Re_true, x_obs, y_obs, t_obs)
        print(f"  Generated {len(x_obs)} exact observation points (no model error)")
    print(f"  u_obs: min={float(u_obs.min()):.3f}  max={float(u_obs.max()):.3f}  "
          f"mean={float(u_obs.mean()):.3f}")
    print(f"  v_obs: min={float(v_obs.min()):.3f}  max={float(v_obs.max()):.3f}  "
          f"mean={float(v_obs.mean()):.3f}")

    # ---- inference model (random init) ----
    key = jax.random.key(42)
    model = TaylorGreenPINN(widths=WIDTHS, key=key, activation=jax.nn.tanh)
    log_Re = jnp.array(float(np.log(args.Re_init)))

    print(f"\nRe_true = {args.Re_true:.1f}   Re_init = {args.Re_init:.1f}")
    print(f"  Network: random initialization, widths={WIDTHS}")
    print(f"  log_Re init = {float(log_Re):.4f}   (truth = {np.log(args.Re_true):.4f})\n")

    # ---- collocation points ----
    k1, _ = jax.random.split(key)
    x_col, y_col, t_col = sample_interior(k1, 12000, t_max=T_MAX)
    print(f"Collocation pts: {len(x_col)}\n")

    warmup_loss_fn = make_warmup_loss_fn(
        x_obs, y_obs, t_obs, u_obs, v_obs, w_data=args.w_data
    )

    wu_schedule = optax.cosine_decay_schedule(
        init_value=args.net_lr, decay_steps=args.warmup_epochs, alpha=1e-2
    )
    wu_opt = nnx.Optimizer(model, optax.adam(learning_rate=wu_schedule), wrt=nnx.Param)

    @nnx.jit
    def warmup_step(model, opt):
        loss, grads = nnx.value_and_grad(warmup_loss_fn)(model)
        opt.update(model, grads)
        return loss

    print(f"  net_lr = {args.net_lr:.1e}   re_lr = {args.re_lr:.1e}")
    print(f"  warm-up epochs = {args.warmup_epochs}   joint epochs = {args.epochs}\n")
    t0 = time.perf_counter()

    with tqdm(total=args.warmup_epochs, desc="Warm-up (data)", unit="epoch",
              dynamic_ncols=True) as pbar:
        for epoch in range(args.warmup_epochs):
            wu_loss = warmup_step(model, wu_opt)
            if epoch % 20 == 0:
                pbar.set_postfix(loss=f"{float(wu_loss):.3e}")
            pbar.update(1)

    print(f"  Warm-up done. Data loss ≈ {float(wu_loss):.3e}\n")

    Re_history = []
    loss_history = []

    if args.re_mode == "closed_form":
        Re_final = run_closed_form(
            model, args, x_col, y_col, t_col, x_obs, y_obs, t_obs, u_obs, v_obs,
            Re_history, loss_history,
        )
    else:
        Re_final = run_free(
            model, log_Re, args, x_col, y_col, t_col, x_obs, y_obs, t_obs, u_obs, v_obs,
            Re_history, loss_history,
        )

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f} s")
    print(f"Re_true = {args.Re_true:.1f}   Re_init = {args.Re_init:.1f}   Re_final = {Re_final:.2f}")
    print(f"Error = {abs(Re_final - args.Re_true) / args.Re_true * 100:.1f}%\n")

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
        f.write(f"obs_src  = {args.obs_source}\n")
        f.write(f"re_mode  = {args.re_mode}\n")

    plot_Re_convergence(Re_history, args.Re_true, PLOTS_DIR)
    plot_inference_loss(loss_history, PLOTS_DIR)

    print(f"\nPlots saved to {PLOTS_DIR}/")
    print(f"Results saved to {RESULTS_DIR}/")
