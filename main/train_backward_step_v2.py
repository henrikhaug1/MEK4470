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
    FourierBackwardStepPINN,
    lbfgs_step,
    save_fourier_model,
    eval_uvp_batch_bs_raw,
    residuals_batch_bs_raw,
    dudx_at_outlet_bs_raw,
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


WIDTHS = [128, 128, 128, 128]
N_FOURIER = 16  # Fourier feature pairs → 2×16 = 32-dim input to MLP
N_COL = 12000
N_BC = 2000
ADAM_EPOCHS = 8000
LBFGS_STEPS = 5000
LR_START = 1e-3
LR_END = 1e-5


def get_config(Re):
    x_r_est = 0.025 * Re
    if Re >= 400:
        x_max = 30.0
    else:
        x_max = 20.0
    return dict(
        x_max=x_max,
        widths=WIDTHS,
        n_col=N_COL,
        n_bc=N_BC,
        adam_epochs=ADAM_EPOCHS,
        lr_start=LR_START,
        lr_end=LR_END,
        lbfgs_steps=LBFGS_STEPS,
        x_recirc=min(x_r_est * 1.5, x_max),
    )


# ============================================================
# Samplers
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


def sample_boundaries(key, n_per_seg, x_min, x_max, h_step, h_chan):
    """Sample points on each of the 6 boundary segments."""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # Inlet: x = x_min, y in [h_step, h_chan]
    x_in = jnp.full(n_per_seg, x_min)
    y_in = jax.random.uniform(k1, (n_per_seg,)) * (h_chan - h_step) + h_step

    # Top wall: y = h_chan, x in [x_min, x_max]
    x_top = jax.random.uniform(k2, (n_per_seg,)) * (x_max - x_min) + x_min
    y_top = jnp.full(n_per_seg, h_chan)

    # Bottom wall: y = 0, x in [0, x_max]
    x_bot = jax.random.uniform(k3, (n_per_seg,)) * x_max
    y_bot = jnp.zeros(n_per_seg)

    # Step top (upstream lower wall): y = h_step, x in [x_min, 0]
    x_st = jax.random.uniform(k4, (n_per_seg,)) * abs(x_min) + x_min
    y_st = jnp.full(n_per_seg, h_step)

    # Step face: x = 0, y in [0, h_step]
    x_sf = jnp.zeros(n_per_seg)
    y_sf = jax.random.uniform(k5, (n_per_seg,)) * h_step

    # Outlet: x = x_max, y in [0, h_chan]
    x_out = jnp.full(n_per_seg, x_max)
    y_out = jax.random.uniform(k6, (n_per_seg,)) * h_chan

    return (
        x_in,
        y_in,
        x_top,
        y_top,
        x_bot,
        y_bot,
        x_st,
        y_st,
        x_sf,
        y_sf,
        x_out,
        y_out,
    )


# ============================================================
# Loss with explicit boundary terms
# ============================================================


def inlet_profile(y, h_step, h_chan, u_mean):
    return u_mean * 6.0 * (y - h_step) * (h_chan - y) / (h_chan - h_step) ** 2


N_FLUX_SECTIONS = 15
N_FLUX_QUAD = 40

# ── Fixed loss weights — [cont, mom, inlet, walls, step, outlet, flux] ───────
LOSS_WEIGHTS = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 2.0])


def make_loss_fn(
    Re, x_col, y_col, bc_pts, x_min, x_max, x_flux, y_flux_quad, weights=LOSS_WEIGHTS
):
    """weights: array of 7 floats — [cont, mom, inlet, walls, step, outlet, flux]."""
    (x_in, y_in, x_top, y_top, x_bot, y_bot, x_st, y_st, x_sf, y_sf, x_out, y_out) = (
        bc_pts
    )

    u_inlet_ref = inlet_profile(y_in, H_STEP, H_CHAN, U_MEAN)
    target_flux = U_MEAN * (H_CHAN - H_STEP)
    w_cont, w_mom, w_inlet, w_walls, w_step, w_outlet, w_flux = (
        float(w) for w in weights
    )

    def loss_fn(m):
        cont, mom_x, mom_y = residuals_batch_bs_raw(
            m, Re, x_col, y_col, x_min, x_max, H_CHAN
        )
        loss_cont = jnp.mean(cont**2)
        loss_mom = jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        u_i, v_i, _ = eval_uvp_batch_bs_raw(m, x_in, y_in, x_min, x_max, H_CHAN)
        loss_inlet = jnp.mean((u_i - u_inlet_ref) ** 2) + jnp.mean(v_i**2)

        u_t, v_t, _ = eval_uvp_batch_bs_raw(m, x_top, y_top, x_min, x_max, H_CHAN)
        u_b, v_b, _ = eval_uvp_batch_bs_raw(m, x_bot, y_bot, x_min, x_max, H_CHAN)
        loss_walls = jnp.mean(u_t**2 + v_t**2) + jnp.mean(u_b**2 + v_b**2)

        u_st, v_st, _ = eval_uvp_batch_bs_raw(m, x_st, y_st, x_min, x_max, H_CHAN)
        u_sf, v_sf, _ = eval_uvp_batch_bs_raw(m, x_sf, y_sf, x_min, x_max, H_CHAN)
        loss_step = jnp.mean(u_st**2 + v_st**2) + jnp.mean(u_sf**2 + v_sf**2)

        _, _, p_o = eval_uvp_batch_bs_raw(m, x_out, y_out, x_min, x_max, H_CHAN)
        dudx_o, dvdx_o = dudx_at_outlet_bs_raw(m, x_out, y_out, x_min, x_max, H_CHAN)
        loss_outlet = jnp.mean(p_o**2) + jnp.mean(dudx_o**2) + jnp.mean(dvdx_o**2)

        def flux_at_section(x0):
            x_pts = jnp.full_like(y_flux_quad, x0)
            u_q, _, _ = eval_uvp_batch_bs_raw(
                m, x_pts, y_flux_quad, x_min, x_max, H_CHAN
            )
            return jnp.trapezoid(u_q, y_flux_quad)

        fluxes = jax.vmap(flux_at_section)(x_flux)
        loss_flux = jnp.mean((fluxes - target_flux) ** 2)

        return (
            w_cont * loss_cont
            + w_mom * loss_mom
            + w_inlet * loss_inlet
            + w_walls * loss_walls
            + w_step * loss_step
            + w_outlet * loss_outlet
            + w_flux * loss_flux
        )

    return loss_fn


def make_diag_fn(Re, x_col, y_col, bc_pts, x_min, x_max, x_flux, y_flux_quad):
    """Return individual loss components for logging."""
    (x_in, y_in, x_top, y_top, x_bot, y_bot, x_st, y_st, x_sf, y_sf, x_out, y_out) = (
        bc_pts
    )
    u_inlet_ref = inlet_profile(y_in, H_STEP, H_CHAN, U_MEAN)
    target_flux = U_MEAN * (H_CHAN - H_STEP)

    @jax.jit
    def diag_fn(m):
        cont, mom_x, mom_y = residuals_batch_bs_raw(
            m,
            Re,
            x_col,
            y_col,
            x_min,
            x_max,
            H_CHAN,
        )
        loss_cont = jnp.mean(cont**2)
        loss_mom = jnp.mean(mom_x**2) + jnp.mean(mom_y**2)

        u_i, v_i, _ = eval_uvp_batch_bs_raw(m, x_in, y_in, x_min, x_max, H_CHAN)
        loss_inlet = jnp.mean((u_i - u_inlet_ref) ** 2) + jnp.mean(v_i**2)

        u_t, v_t, _ = eval_uvp_batch_bs_raw(m, x_top, y_top, x_min, x_max, H_CHAN)
        u_b, v_b, _ = eval_uvp_batch_bs_raw(m, x_bot, y_bot, x_min, x_max, H_CHAN)
        loss_walls = jnp.mean(u_t**2 + v_t**2) + jnp.mean(u_b**2 + v_b**2)

        u_st, v_st, _ = eval_uvp_batch_bs_raw(m, x_st, y_st, x_min, x_max, H_CHAN)
        u_sf, v_sf, _ = eval_uvp_batch_bs_raw(m, x_sf, y_sf, x_min, x_max, H_CHAN)
        loss_step = jnp.mean(u_st**2 + v_st**2) + jnp.mean(u_sf**2 + v_sf**2)

        _, _, p_o = eval_uvp_batch_bs_raw(m, x_out, y_out, x_min, x_max, H_CHAN)
        dudx_o, dvdx_o = dudx_at_outlet_bs_raw(m, x_out, y_out, x_min, x_max, H_CHAN)
        loss_outlet = jnp.mean(p_o**2) + jnp.mean(dudx_o**2) + jnp.mean(dvdx_o**2)

        def flux_at_section(x0):
            x_pts = jnp.full_like(y_flux_quad, x0)
            u_q, _, _ = eval_uvp_batch_bs_raw(
                m, x_pts, y_flux_quad, x_min, x_max, H_CHAN
            )
            return jnp.trapezoid(u_q, y_flux_quad)

        fluxes = jax.vmap(flux_at_section)(x_flux)
        loss_flux = jnp.mean((fluxes - target_flux) ** 2)

        return (
            loss_cont,
            loss_mom,
            loss_inlet,
            loss_walls,
            loss_step,
            loss_outlet,
            loss_flux,
        )

    return diag_fn


# ============================================================
# BC sanity check
# ============================================================


def check_bcs(model, x_min, x_max, n=200):
    def _rms(arr):
        return float(jnp.sqrt(jnp.mean(arr**2)))

    # Inlet
    y_in = jnp.linspace(H_STEP, H_CHAN, n)
    x_in = jnp.full(n, x_min)
    u_in, v_in, _ = eval_uvp_batch_bs_raw(model, x_in, y_in, x_min, x_max, H_CHAN)
    u_ref = inlet_profile(y_in, H_STEP, H_CHAN, U_MEAN)

    # Top wall
    x_top = jnp.linspace(x_min, x_max, n)
    y_top = jnp.full(n, H_CHAN)
    u_top, v_top, _ = eval_uvp_batch_bs_raw(model, x_top, y_top, x_min, x_max, H_CHAN)

    # Bottom wall
    x_bot = jnp.linspace(0.0, x_max, n)
    y_bot = jnp.zeros(n)
    u_bot, v_bot, _ = eval_uvp_batch_bs_raw(model, x_bot, y_bot, x_min, x_max, H_CHAN)

    # Step top
    x_st = jnp.linspace(x_min, 0.0, n)
    y_st = jnp.full(n, H_STEP)
    u_st, v_st, _ = eval_uvp_batch_bs_raw(model, x_st, y_st, x_min, x_max, H_CHAN)

    # Step face
    x_sf = jnp.zeros(n)
    y_sf = jnp.linspace(0.0, H_STEP, n)
    u_sf, v_sf, _ = eval_uvp_batch_bs_raw(model, x_sf, y_sf, x_min, x_max, H_CHAN)

    # Outlet
    x_out = jnp.full(n, x_max)
    y_out = jnp.linspace(0.0, H_CHAN, n)
    _, _, p_out = eval_uvp_batch_bs_raw(model, x_out, y_out, x_min, x_max, H_CHAN)

    print("\n── BC sanity check (RMS errors) ──────────────────────────────")
    print(f"  Inlet      u: {_rms(u_in - u_ref):.2e}   v: {_rms(v_in):.2e}")
    print(f"  Top wall   u: {_rms(u_top):.2e}   v: {_rms(v_top):.2e}")
    print(f"  Bottom wall u: {_rms(u_bot):.2e}   v: {_rms(v_bot):.2e}")
    print(f"  Step top   u: {_rms(u_st):.2e}   v: {_rms(v_st):.2e}")
    print(f"  Step face  u: {_rms(u_sf):.2e}   v: {_rms(v_sf):.2e}")
    print(f"  Outlet     p: {_rms(p_out):.2e}")
    print("──────────────────────────────────────────────────────────────\n")


# ============================================================
# main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--Re", type=float, default=200.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    Re = args.Re
    seed = args.seed

    cfg = get_config(Re)
    X_MAX = cfg["x_max"]

    print(f"\nBackward step PINN v2 (soft BCs)  |  Re = {Re}  seed = {seed}")
    print(f"  Domain      : x in [{X_MIN}, {X_MAX:.1f}]  y in [0, {H_CHAN}]")
    print(f"  Network     : {cfg['widths']}")
    print(f"  Adam epochs : {cfg['adam_epochs']}   L-BFGS steps: {cfg['lbfgs_steps']}")
    print(f"  Collocation : {cfg['n_col']}   BC pts/seg: {cfg['n_bc']}")
    print(f"  x_recirc    : {cfg['x_recirc']:.1f} H\n")

    RESULTS_DIR = f"results/backward_step_v2/Re{int(Re)}"
    PLOTS_DIR = f"plots/backward_step_v2/Re{int(Re)}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    PARAMS_FILE = os.path.join(RESULTS_DIR, "params.npz")
    ADAM_LOSSES_FILE = os.path.join(RESULTS_DIR, "adam_losses.txt")
    LBFGS_LOSSES_FILE = os.path.join(RESULTS_DIR, "lbfgs_losses.txt")

    key = jax.random.key(seed)
    model = FourierBackwardStepPINN(
        widths=cfg["widths"],
        key=key,
        activation=jax.nn.tanh,
        n_inputs=2,
        n_fourier=N_FOURIER,
    )

    # --------------------------------------------------------
    # sample points (boundary points are fixed; collocation resampled)
    # --------------------------------------------------------
    RESAMPLE_EVERY = 2000

    k_col, k_bc = jax.random.split(key, 2)
    print("Sampling boundary points...")
    bc_pts = sample_boundaries(k_bc, cfg["n_bc"], X_MIN, X_MAX, H_STEP, H_CHAN)
    print(f"  boundary : {cfg['n_bc']} x 6 segments = {cfg['n_bc'] * 6}")

    print("Sampling initial collocation points...")
    x_col, y_col = sample_interior(k_col, cfg["n_col"], X_MAX, cfg["x_recirc"])
    print(f"  interior : {len(x_col)}\n")

    # --------------------------------------------------------
    # mass flux quadrature (fixed throughout training)
    # --------------------------------------------------------
    x_flux = jnp.linspace(1.0, X_MAX - 1.0, N_FLUX_SECTIONS)
    y_flux_quad = jnp.linspace(0.0, H_CHAN, N_FLUX_QUAD)

    # --------------------------------------------------------
    # Loss + diagnostics (fixed weights)
    # --------------------------------------------------------
    loss_fn = make_loss_fn(Re, x_col, y_col, bc_pts, X_MIN, X_MAX, x_flux, y_flux_quad)
    diag_fn = make_diag_fn(Re, x_col, y_col, bc_pts, X_MIN, X_MAX, x_flux, y_flux_quad)

    # --------------------------------------------------------
    # Adam with cosine LR decay
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
    DIAG_EVERY = 2000
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

            if epoch > 0 and epoch % DIAG_EVERY == 0:
                d_cont, d_mom, d_inlet, d_walls, d_step, d_outlet, d_flux = diag_fn(
                    model
                )
                pbar.write(
                    f"  [diag {epoch}] cont={float(d_cont):.2e}  mom={float(d_mom):.2e}"
                    f"  inlet={float(d_inlet):.2e}  walls={float(d_walls):.2e}"
                    f"  step={float(d_step):.2e}  outlet={float(d_outlet):.2e}"
                    f"  flux={float(d_flux):.2e}"
                )

            if epoch > 0 and epoch % RESAMPLE_EVERY == 0:
                k_col = jax.random.fold_in(k_col, epoch)
                x_col, y_col = sample_interior(
                    k_col, cfg["n_col"], X_MAX, cfg["x_recirc"]
                )
                loss_fn = make_loss_fn(
                    Re,
                    x_col,
                    y_col,
                    bc_pts,
                    X_MIN,
                    X_MAX,
                    x_flux,
                    y_flux_quad,
                )
                diag_fn = make_diag_fn(
                    Re, x_col, y_col, bc_pts, X_MIN, X_MAX, x_flux, y_flux_quad
                )

                @nnx.jit
                def adam_step(model, opt):
                    loss, grads = nnx.value_and_grad(loss_fn)(model)
                    opt.update(model, grads)
                    return loss

                pbar.write(f"  [resample {epoch}] new collocation points")

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
    check_bcs(model, X_MIN, X_MAX)

    d_cont, d_mom, d_inlet, d_walls, d_step, d_outlet, d_flux = diag_fn(model)
    print(f"── Loss breakdown after Adam ──")
    print(
        f"  cont={float(d_cont):.2e}  mom={float(d_mom):.2e}"
        f"  inlet={float(d_inlet):.2e}  walls={float(d_walls):.2e}"
        f"  step={float(d_step):.2e}  outlet={float(d_outlet):.2e}"
        f"  flux={float(d_flux):.2e}\n"
    )

    # --------------------------------------------------------
    # L-BFGS (uses last collocation set)
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
                if rel < 1e-4:
                    pbar.write(f"Early stopping at step {step}")
                    break

    lbfgs_losses = lbfgs_losses[: step + 1]
    nnx.update(model, params)

    lbfgs_time = time.perf_counter() - t1
    print(f"L-BFGS complete in {lbfgs_time:.1f} s\n")
    check_bcs(model, X_MIN, X_MAX)

    diag_fn = make_diag_fn(Re, x_col, y_col, bc_pts, X_MIN, X_MAX, x_flux, y_flux_quad)
    d_cont, d_mom, d_inlet, d_walls, d_step, d_outlet, d_flux = diag_fn(model)
    print(f"── Loss breakdown after L-BFGS ──")
    print(
        f"  cont={float(d_cont):.2e}  mom={float(d_mom):.2e}"
        f"  inlet={float(d_inlet):.2e}  walls={float(d_walls):.2e}"
        f"  step={float(d_step):.2e}  outlet={float(d_outlet):.2e}"
        f"  flux={float(d_flux):.2e}\n"
    )

    # --------------------------------------------------------
    # save
    # --------------------------------------------------------
    save_fourier_model(model, PARAMS_FILE)
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

    u, v, p = eval_uvp_batch_bs_raw(
        model,
        X.ravel(),
        Y.ravel(),
        X_MIN,
        X_MAX,
        H_CHAN,
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

    u_wall, _, _ = eval_uvp_batch_bs_raw(
        model,
        x_probe,
        y_probe,
        X_MIN,
        X_MAX,
        H_CHAN,
    )

    u_wall_min = float(jnp.min(u_wall))
    print(
        f"  min u at y=0.15 : {u_wall_min:.4f}  "
        f"({'backflow detected' if u_wall_min < 0 else 'NO backflow'})"
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
