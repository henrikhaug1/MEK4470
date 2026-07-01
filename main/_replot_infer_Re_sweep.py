"""Regenerate figures/infer_Re_sweep.* from the cached sweep CSV.

Mirrors the plotting block of sweep_infer_Re_backward_step.py so the figure
can be re-rendered without re-running the (slow) inference sweep.
"""
import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RE_TRUE_VALUES  = [100, 200, 400]
RE_INIT_FACTORS = [1 / 8, 1 / 4, 1 / 2, 2, 4, 8]
OUT_DIR     = "figures"
CSV_PATH    = "results/infer_Re_sweep/sweep_results.csv"

os.makedirs(OUT_DIR, exist_ok=True)

results = []
with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        results.append({
            "Re_true":  int(float(row["Re_true"])),
            "factor":   float(row["factor"]),
            "Re_final": float(row["Re_final"]),
            "err_pct":  float(row["err_pct"]),
        })

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# ── Plot 1: % error vs Re_init/Re_true ───────────────────────
colors = plt.cm.tab10.colors
for i, Re_true in enumerate(RE_TRUE_VALUES):
    sub = [r for r in results if r["Re_true"] == Re_true]
    if not sub:
        continue
    sub.sort(key=lambda r: r["factor"])
    factors  = [r["factor"]  for r in sub]
    err_pcts = [r["err_pct"] for r in sub]
    axes[0].plot(factors, err_pcts, "o-", color=colors[i],
                 lw=1.8, ms=6, label=f"Re = {Re_true}")

axes[0].axhline(10, color="0.6", lw=0.8, ls="--", label="10 % threshold")
axes[0].set_xscale("log")
tick_labels = {
    0.125: "1/8", 0.25: "1/4", 0.5: "1/2",
    1.0: "1", 2.0: "2", 4.0: "4", 8.0: "8",
}
ticks = sorted(set(RE_INIT_FACTORS) | {1.0})
axes[0].set_xticks(ticks)
axes[0].set_xticklabels([tick_labels.get(t, f"{t:g}") for t in ticks], fontsize=9)
axes[0].minorticks_off()
axes[0].set_xlabel(
    r"$\mathrm{Re}_{\mathrm{init}}\,/\,\mathrm{Re}_{\mathrm{true}}$"
    "   (initialisation ratio)",
    fontsize=11,
)
axes[0].set_ylabel("Relative error  (%)", fontsize=11)
axes[0].set_title("Inference error vs initialisation ratio", fontsize=11)
axes[0].axvline(1, color="0.7", lw=0.7, ls=":")
axes[0].legend(fontsize=7, loc="upper right")

# ── Plot 2: Re_final vs Re_true scatter ──────────────────────
import numpy as np
all_Re_true  = [r["Re_true"]  for r in results]
all_Re_final = [r["Re_final"] for r in results]
all_factors  = [r["factor"]   for r in results]

# Spread the 6 init-factor runs horizontally within each Re_true column so
# the (otherwise overlapping) markers are all distinguishable. The jitter is
# proportional to log10(Re_init/Re_true) and ±4 % of Re_true wide.
x_jit = [rt * (1.0 + 0.045 * np.log10(f))
         for rt, f in zip(all_Re_true, all_factors)]

axes[1].scatter(
    x_jit, all_Re_final,
    color="black",
    s=14, zorder=3, edgecolors="0.3", linewidths=0.4,
)

lo = min(all_Re_true) * 0.8
hi = max(all_Re_true) * 1.2
axes[1].plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect")
axes[1].set_xlabel(r"$\mathrm{Re}_{\mathrm{true}}$", fontsize=11)
axes[1].set_ylabel(r"$\mathrm{Re}_{\mathrm{final}}$", fontsize=11)
axes[1].set_title("Inferred vs true Re  (obs: OpenFOAM)", fontsize=11)
axes[1].legend(fontsize=7, loc="upper left")

fig.tight_layout(pad=1.5)
for ext in ("pdf", "png"):
    path = os.path.join(OUT_DIR, f"infer_Re_sweep.{ext}")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved → {path}")
plt.close(fig)
