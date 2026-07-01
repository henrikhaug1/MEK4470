"""
sweep_infer_Re_lid_cavity.py
----------------------------
Sweep Re_true × Re_init combinations for the lid-driven cavity inference.
Reuses the run drivers and observation loaders from main/infer_Re_lid_cavity.py.

Observations come from a trained forward PINN (default) or OpenFOAM
(OBS_SOURCE = "openfoam").

Produces:
  figures/infer_Re_sweep_lid_cavity.pdf/png
  results/infer_Re_sweep_lid_cavity/sweep_results.csv
"""

import os
import sys
import time
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pinn import BackwardStepPINN, sample_interior_lc
from main.infer_Re_lid_cavity import (
    WIDTHS,
    load_synthetic_observations,
    load_of_observations,
    make_warmup_loss_fn,
    run_free,
    run_closed_form,
)

# ═══════════════════════════════════════════════════════════════
# Sweep configuration
# ═══════════════════════════════════════════════════════════════

RE_TRUE_VALUES  = [100, 500, 800]
RE_INIT_FACTORS = [1/8, 1/4, 1/2, 2, 4, 8]   # Re_init = Re_true × factor

OBS_SOURCE = "synthetic"     # "synthetic" (needs forward ckpt) or "openfoam"
RE_MODE    = "closed_form"   # "closed_form" (sturdy) or "free"

EPOCHS        = 5000
WARMUP_EPOCHS = 1000
N_OBS         = 500
N_COL         = 12000
NET_LR        = 1e-3
RE_LR         = 1e-2
W_DATA        = 1000.0
RE_EMA        = 0.99
NO_EARLY_STOP = False

OUT_DIR     = "figures"
RESULTS_DIR = "results/infer_Re_sweep_lid_cavity"


def _make_args(Re_init):
    return SimpleNamespace(
        Re_init=Re_init,
        epochs=EPOCHS,
        net_lr=NET_LR,
        re_lr=RE_LR,
        w_data=W_DATA,
        re_ema=RE_EMA,
        no_early_stop=NO_EARLY_STOP,
    )


def _load_obs(Re_true, x_obs, y_obs):
    if OBS_SOURCE == "openfoam":
        return load_of_observations(Re_true, x_obs, y_obs)
    return load_synthetic_observations(Re_true, x_obs, y_obs)


# ═══════════════════════════════════════════════════════════════
# Single inference run (warm-up + chosen joint driver)
# ═══════════════════════════════════════════════════════════════

def run_inference(Re_true, Re_init, x_obs, y_obs, u_obs, v_obs, seed=42, desc=""):
    key = jax.random.key(seed)
    model = BackwardStepPINN(widths=WIDTHS, key=key, activation=jax.nn.tanh, n_inputs=2)
    log_Re = jnp.array(float(np.log(Re_init)))
    args = _make_args(Re_init)

    k1, _ = jax.random.split(key)
    x_col, y_col = sample_interior_lc(k1, N_COL)

    # ── Phase 1: warm-up (data only) ──────────────────────────────
    warmup_loss = make_warmup_loss_fn(x_obs, y_obs, u_obs, v_obs, w_data=W_DATA)
    wu_sched = optax.cosine_decay_schedule(NET_LR, WARMUP_EPOCHS, alpha=1e-2)
    wu_opt = nnx.Optimizer(model, optax.adam(wu_sched), wrt=nnx.Param)

    @nnx.jit
    def wu_step(model, opt):
        loss, grads = nnx.value_and_grad(warmup_loss)(model)
        opt.update(model, grads)
        return loss

    for _ in range(WARMUP_EPOCHS):
        wu_step(model, wu_opt)

    # ── Phase 2: joint inference ──────────────────────────────────
    Re_history, loss_history = [], []
    if RE_MODE == "closed_form":
        Re_final = run_closed_form(
            model, args, x_col, y_col, x_obs, y_obs, u_obs, v_obs,
            Re_history, loss_history,
        )
    else:
        Re_final = run_free(
            model, log_Re, args, x_col, y_col, x_obs, y_obs, u_obs, v_obs,
            Re_history, loss_history,
        )

    err_pct = abs(Re_final - Re_true) / Re_true * 100.0
    return Re_final, err_pct


# ═══════════════════════════════════════════════════════════════
# Main sweep
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    for Re_true in RE_TRUE_VALUES:
        print(f"\n{'═'*55}")
        print(f"  Re_true = {Re_true}  —  {OBS_SOURCE} observations  ({RE_MODE})")
        print(f"{'═'*55}")

        # Fixed observation locations in the cavity interior [-1,1]²
        key_obs = jax.random.key(99)
        pts = jax.random.uniform(key_obs, (N_OBS, 2)) * 2.0 - 1.0
        x_obs, y_obs = pts[:, 0], pts[:, 1]

        try:
            u_obs, v_obs = _load_obs(Re_true, x_obs, y_obs)
        except (ValueError, FileNotFoundError) as e:
            print(f"  Re={Re_true}: {e}\n  skipping")
            continue
        print(f"  {N_OBS} obs points: u range [{float(u_obs.min()):.3f}, {float(u_obs.max()):.3f}]")

        for factor in RE_INIT_FACTORS:
            Re_init = Re_true * factor
            if Re_init < 5:
                continue
            desc = f"Re_true={Re_true:4d}  Re_init={Re_init:7.1f}"
            t0 = time.perf_counter()
            Re_final, err_pct = run_inference(
                Re_true, Re_init, x_obs, y_obs, u_obs, v_obs, desc=desc
            )
            elapsed = time.perf_counter() - t0
            print(f"  {desc}  →  Re_final={Re_final:8.2f}   err={err_pct:6.2f}%   ({elapsed:.0f}s)")

            results.append(dict(
                Re_true=Re_true, Re_init=Re_init, factor=factor,
                Re_final=Re_final, err_pct=err_pct,
                log_ratio=float(np.log10(factor)), elapsed_s=elapsed,
            ))

    if not results:
        print("No results.")
        return

    csv_path = os.path.join(RESULTS_DIR, "sweep_results.csv")
    header = "Re_true,Re_init,factor,Re_final,err_pct,log_ratio,elapsed_s"
    rows = [f"{r['Re_true']},{r['Re_init']:.1f},{r['factor']:.4f},"
            f"{r['Re_final']:.4f},{r['err_pct']:.4f},{r['log_ratio']:.4f},{r['elapsed_s']:.1f}"
            for r in results]
    with open(csv_path, "w") as f:
        f.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"\nResults saved → {csv_path}")

    _plot(results, "Lid-driven cavity", "lid_cavity")


def _plot(results, title_tag, file_tag):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = plt.cm.tab10.colors

    for i, Re_true in enumerate(RE_TRUE_VALUES):
        sub = [r for r in results if r["Re_true"] == Re_true]
        if not sub:
            continue
        factors = [r["factor"] for r in sub]
        err_pcts = [r["err_pct"] for r in sub]
        axes[0].plot(factors, err_pcts, "o-", color=colors[i],
                     lw=1.8, ms=6, label=f"Re = {Re_true}")

    axes[0].axhline(10, color="0.6", lw=0.8, ls="--", label="10 % threshold")
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\mathrm{Re}_{\mathrm{init}}\,/\,\mathrm{Re}_{\mathrm{true}}$", fontsize=11)
    axes[0].set_ylabel("Relative error  (%)", fontsize=11)
    axes[0].set_title(f"{title_tag}: error vs initialisation ratio", fontsize=11)
    axes[0].axvline(1, color="0.7", lw=0.7, ls=":")
    axes[0].legend(fontsize=7, loc="upper right")

    all_Re_true = [r["Re_true"] for r in results]
    all_Re_final = [r["Re_final"] for r in results]
    all_factors = [r["factor"] for r in results]

    sc = axes[1].scatter(
        all_Re_true, all_Re_final, c=np.log10(all_factors),
        cmap="RdYlGn_r", s=60, zorder=3, edgecolors="0.3", linewidths=0.5,
    )
    cb = fig.colorbar(sc, ax=axes[1], pad=0.02)
    cb.set_label(
        r"$\log_{10}(\mathrm{Re}_{\mathrm{init}}\,/\,\mathrm{Re}_{\mathrm{true}})$",
        fontsize=9, rotation=270, labelpad=14, va="bottom",
    )

    lo = min(all_Re_true) * 0.8
    hi = max(all_Re_true) * 1.2
    axes[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
    axes[1].set_xlabel(r"$\mathrm{Re}_{\mathrm{true}}$", fontsize=11)
    axes[1].set_ylabel(r"$\mathrm{Re}_{\mathrm{final}}$", fontsize=11)
    axes[1].set_title(f"Inferred vs true Re  (obs: {OBS_SOURCE})", fontsize=11)
    axes[1].legend(fontsize=7, loc="upper left")

    fig.tight_layout(pad=1.5)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"infer_Re_sweep_{file_tag}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved → {path}")
    plt.close(fig)


if __name__ == "__main__":
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })
    main()
