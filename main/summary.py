"""
Print a wall-clock timing summary for all PINN training runs and OpenFOAM simulations.

PINN times are read from results/{case}/Re{N}/timing.json (written by training scripts).
OpenFOAM times are read from log.icoFoam in each case directory (ClockTime on last line).
If no log file exists the entry falls back to a hardcoded value or N/A.
"""

import csv
import json
import os
import re as re_module

import pandas as pd


def fmt(seconds):
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s:02d}s"
    else:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"


def load_pinn_timing(path):
    timing_file = os.path.join(path, "timing.json")
    if not os.path.exists(timing_file):
        return None
    with open(timing_file) as f:
        return json.load(f)


def load_openfoam_timing(case_dir):
    """Read ClockTime from the last ExecutionTime line in log.icoFoam."""
    log_path = os.path.join(case_dir, "log.icoFoam")
    if not os.path.exists(log_path):
        return None
    clock_time = None
    pattern = re_module.compile(r"ClockTime\s*=\s*([\d.]+)\s*s")
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                clock_time = float(m.group(1))
    return clock_time


# ── PINN result directories to scan ──────────────────────────────────────────
PINN_CASES = [
    ("Taylor-Green",  "results/taylor_green"),
    ("Backward Step", "results/backward_step_v2"),
    ("Lid Cavity",    "results/lid_cavity"),
]

# ── OpenFOAM cases ────────────────────────────────────────────────────────────
OPENFOAM_CASES = [
    # Taylor-Green
    ("Taylor-Green",  200,  "linear",  "openfoam/run/taylor_green_re200",             163),
    ("Taylor-Green",  1000, "linear",  "openfoam/run/taylor_green_re1000",             152),
    # Backward Step (parabolic inlet)
    ("Backward Step", 100,  "linear",  "openfoam/run/backwardStep_parabolic_re100",   None),
    ("Backward Step", 200,  "linear",  "openfoam/run/backwardStep_parabolic_re200",   None),
    ("Backward Step", 400,  "linear",  "openfoam/run/backwardStep_parabolic_re400",   None),
    # Lid Cavity
    ("Lid Cavity",    100,  "upwind",  "openfoam/run/lid_cav_re100",                  None),
    ("Lid Cavity",    500,  "upwind",  "openfoam/run/lid_cav_re500",                  None),
    ("Lid Cavity",    800,  "upwind",  "openfoam/run/lid_cav_re800",                  None),
]

OF_CONFIGS = {
    "Taylor-Green":  "t=0–20, dt=0.001, icoFoam",
    "Backward Step": "steady-state, icoFoam",
    "Lid Cavity":    "steady-state, icoFoam",
}

# ── Collect PINN timings ──────────────────────────────────────────────────────
rows = []

for case_name, results_dir in PINN_CASES:
    if not os.path.isdir(results_dir):
        continue
    for re_dir in sorted(
        os.listdir(results_dir),
        key=lambda d: int(d[2:]) if d.startswith("Re") and d[2:].isdigit() else 0,
    ):
        if not re_dir.startswith("Re"):
            continue
        re_val = int(re_dir[2:])
        timing = load_pinn_timing(os.path.join(results_dir, re_dir))
        if timing:
            ep = timing.get("adam_epochs", "?")
            st = timing.get("lbfgs_steps", "?")
            config = f"{ep} Adam / {st} L-BFGS"
        else:
            config = "—"
        rows.append({
            "Solver": "PINN",
            "Case":   case_name,
            "Re":     re_val,
            "Scheme": "—",
            "Adam":   fmt(timing.get("adam_s"))  if timing else "N/A",
            "L-BFGS": fmt(timing.get("lbfgs_s")) if timing else "N/A",
            "Total":  fmt(timing.get("total_s")) if timing else "N/A",
            "Config": config,
        })

# ── Collect OpenFOAM timings ──────────────────────────────────────────────────
for case_name, re_val, scheme, case_dir, hardcoded in OPENFOAM_CASES:
    total = hardcoded if hardcoded is not None else load_openfoam_timing(case_dir)
    rows.append({
        "Solver": "OpenFOAM",
        "Case":   case_name,
        "Re":     re_val,
        "Scheme": scheme,
        "Adam":   "—",
        "L-BFGS": "—",
        "Total":  fmt(total),
        "Config": OF_CONFIGS.get(case_name, ""),
    })

# ── Collect Re-inference sweep timings ───────────────────────────────────────
SWEEP_CSV = "results/infer_Re_sweep/sweep_results.csv"
infer_rows = []

if os.path.exists(SWEEP_CSV):
    with open(SWEEP_CSV) as f:
        reader = csv.DictReader(f)
        sweep_data = list(reader)

    has_elapsed = "elapsed_s" in (sweep_data[0] if sweep_data else {})
    by_re = {}
    for r in sweep_data:
        re_true = int(float(r["Re_true"]))
        by_re.setdefault(re_true, []).append(r)

    for re_true in sorted(by_re):
        runs = by_re[re_true]
        n = len(runs)
        if has_elapsed:
            total_s = sum(float(r["elapsed_s"]) for r in runs)
            per_run = total_s / n
            timing_str = fmt(total_s)
            per_run_str = f"{per_run:.0f}s/run"
        else:
            timing_str = "N/A"
            per_run_str = "N/A"
        err_pcts = [float(r["err_pct"]) for r in runs]
        infer_rows.append({
            "Re_true":  re_true,
            "N_runs":   n,
            "Per-run":  per_run_str,
            "Total":    timing_str,
            "Err min%": f"{min(err_pcts):.1f}",
            "Err max%": f"{max(err_pcts):.1f}",
        })

# ── Print table ───────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)

print()
print("Wall-clock timing summary")
print("=" * 100)

for case_name, group in df.groupby("Case", sort=False):
    print(f"\n  {case_name}")
    print(
        group.to_string(
            index=False,
            columns=["Solver", "Re", "Scheme", "Adam", "L-BFGS", "Total", "Config"],
            col_space={"Solver": 10, "Re": 6, "Scheme": 10, "Adam": 10, "L-BFGS": 10, "Total": 10},
        )
    )

# ── Re inference sweep ────────────────────────────────────────────────────────
if infer_rows:
    print("\n  Re Inference (sweep, obs: OpenFOAM)")
    df_inf = pd.DataFrame(infer_rows)
    print(
        df_inf.to_string(
            index=False,
            col_space={"Re_true": 8, "N_runs": 7, "Per-run": 10, "Total": 10,
                       "Err min%": 10, "Err max%": 10},
        )
    )

print()
print("=" * 100)
print("Note: PINN times include JAX JIT compilation on the first step.")
print("      OpenFOAM times marked N/A have no log.icoFoam — fill in hardcoded values above.")
print("      Re inference elapsed_s recorded from next sweep run onwards.")
