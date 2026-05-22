"""
compare_optimizers.py

Runs GA and PPO-RL optimizers on both CASCADE and PISTIL chiplet design models,
then produces:
  1. Hypervolume vs. number of function evaluations (CASCADE)
  2. Hypervolume vs. number of function evaluations (PISTIL)
  3. Pareto front visualization (CASCADE)
  4. Pareto front visualization (PISTIL)

Usage:
    python compare_optimizers.py [--cascade-only] [--pistil-only] [--skip-run]

Dependencies:
    pip install pymoo matplotlib numpy scipy torch
"""

import os
import sys
import json
import argparse
import tempfile
import time
import csv
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Hypervolume helper (uses pymoo's built-in)
# ---------------------------------------------------------------------------
try:
    from pymoo.indicators.hv import HV
    def compute_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
        """
        Compute hypervolume indicator given a set of objective points and a
        reference point.  Both are assumed to be minimisation objectives.
        Returns 0.0 if fewer than 1 non-dominated point is available.
        """
        if points is None or len(points) == 0:
            return 0.0
        pts = np.atleast_2d(points).astype(float)
        # Filter points that are dominated by ref_point on all objectives
        dominated_by_ref = np.all(pts < ref_point, axis=1)
        pts = pts[dominated_by_ref]
        if len(pts) == 0:
            return 0.0
        ind = HV(ref_point=ref_point)
        return float(ind(pts))
except ImportError:
    warnings.warn("pymoo HV not found – falling back to a simple 2-D hypervolume.")
    def compute_hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
        """Simple 2-D sweep-line hypervolume (minimisation)."""
        if points is None or len(points) == 0:
            return 0.0
        pts = np.atleast_2d(points).astype(float)
        ref = np.array(ref_point, dtype=float)
        # Keep only points dominated by ref
        mask = np.all(pts < ref, axis=1)
        pts = pts[mask]
        if len(pts) == 0:
            return 0.0
        # Sort by first objective ascending
        order = np.argsort(pts[:, 0])
        pts = pts[order]
        hv = 0.0
        prev_x = ref[0]
        for i in range(len(pts) - 1, -1, -1):
            x, y = pts[i]
            hv += (prev_x - x) * (ref[1] - y)
            prev_x = x
        return float(hv)


# ---------------------------------------------------------------------------
# Pareto-front utilities
# ---------------------------------------------------------------------------

def pareto_filter(objectives: np.ndarray) -> np.ndarray:
    """
    Return the non-dominated (Pareto) subset of a 2-D objectives array
    (minimisation on both axes).
    """
    pts = np.atleast_2d(objectives).astype(float)
    n = len(pts)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= on all and < on at least one
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                is_dominated[i] = True
                break
    return pts[~is_dominated]


def reference_point(all_objectives_list: list, margin: float = 1.1) -> np.ndarray:
    """
    Compute a reference point from a list of objective arrays by taking the
    element-wise maximum across all runs, then scaling by `margin`.
    """
    combined = np.vstack([np.atleast_2d(o) for o in all_objectives_list if len(o) > 0])
    return np.max(combined, axis=0) * margin


def log_transform_objectives(objectives: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """
    Apply log10 transform to objectives for better visualisation when values
    span many orders of magnitude and converge near zero.
    Clips values to epsilon before taking log to avoid log(0).
    Only used for PISTIL — CASCADE is left on the linear scale.
    """
    if len(objectives) == 0:
        return objectives
    
    valid = objectives[objectives < 1e9]
    if valid.size > 0:
        upper_clip = max(float(np.max(valid)), epsilon)
        clipped = np.clip(objectives, epsilon, upper_clip)
    else:
        clipped = np.clip(objectives, epsilon, None)
    return np.log10(clipped)

# ---------------------------------------------------------------------------
# CSV reader – used to reconstruct per-evaluation history from points.csv
# ---------------------------------------------------------------------------

def read_points_csv_cascade(csv_path: str) -> np.ndarray:
    """
    Read a Cascade points.csv (format: exe_time, energy, n_gpu, n_atten,
    n_sparse, n_conv) and return an (N, 2) objectives array in order of
    appearance (proxy for evaluation order).
    """
    objectives = []
    try:
        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                try:
                    objectives.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue  # skip header if any
    except FileNotFoundError:
        pass
    return np.array(objectives) if objectives else np.empty((0, 2))


def read_points_csv_pistil(csv_path: str) -> np.ndarray:
    """
    Read a Pistil points.csv (has a header row, columns include latency_ms
    and energy_mJ) and return an (N, 2) objectives array in order of
    appearance.
    """
    objectives = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row["latency_ms"])
                    eng = float(row["energy_mJ"])
                    objectives.append([lat, eng])
                except (KeyError, ValueError):
                    continue
    except FileNotFoundError:
        pass
    return np.array(objectives) if objectives else np.empty((0, 2))


# ---------------------------------------------------------------------------
# Hypervolume curve builder
# ---------------------------------------------------------------------------

def hv_curve(objectives_in_order: np.ndarray, ref_point: np.ndarray) -> tuple:
    """
    Given objectives in the order they were evaluated, compute the cumulative
    hypervolume after each evaluation.

    Returns
    -------
    evals : list[int]   – x-axis (1-based evaluation index)
    hvs   : list[float] – y-axis (hypervolume after that many evaluations)
    """
    evals, hvs = [], []
    accumulated = []
    for i, pt in enumerate(objectives_in_order):
        accumulated.append(pt)
        pts_arr = np.array(accumulated)
        pf = pareto_filter(pts_arr)
        hv = compute_hypervolume(pf, ref_point)
        evals.append(i + 1)
        hvs.append(hv)
    return evals, hvs


# ---------------------------------------------------------------------------
# ── CASCADE runners ──────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def run_cascade_ga(
    output_dir: str,
    pop_size: int = 10,
    n_gen: int = 5,
    trace: str = "gpt-j-65536-weighted",
) -> np.ndarray:
    """
    Run the Cascade GA and return all evaluated objective points (N, 2) in
    evaluation order, read back from points.csv.
    """
    # Import here so that the rest of the script works even if cascade is not
    # installed in the current environment.
    try:
        from api.Evaluator.gaCascade import runGACascade
    except ImportError:
        # Try relative import path used during development
        sys.path.insert(0, str(Path(__file__).parent))
        from gaCascade import runGACascade

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[CASCADE GA] pop_size={pop_size}, n_gen={n_gen}, trace={trace}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*60}")

    runGACascade(
        pop_size=pop_size,
        n_gen=n_gen,
        trace=trace,
        output_dir=output_dir,
    )

    csv_path = os.path.join(output_dir, "points.csv")
    objectives = read_points_csv_cascade(csv_path)
    print(f"[CASCADE GA] Read {len(objectives)} points from {csv_path}")
    return objectives


def run_cascade_rl(
    output_dir: str,
    num_epochs: int = 20,
    mini_batch_size: int = 5,
    trace: str = "gpt-j-65536-weighted",
) -> np.ndarray:
    """
    Run the Cascade PPO-RL and return all evaluated objective points (N, 2)
    in evaluation order.
    """
    try:
        from api.Evaluator.rlCascade import runPPOCascade
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from rlCascade import runPPOCascade

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[CASCADE RL] num_epochs={num_epochs}, mini_batch_size={mini_batch_size}")
    print(f"{'='*60}")

    all_designs, all_objectives, design_points = runPPOCascade(
        num_epochs=num_epochs,
        mini_batch_size=mini_batch_size,
        trace=trace,
    )

    # Also persist to CSV for reproducibility
    csv_path = os.path.join(output_dir, "points.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["exe_time_ms", "energy_mJ"])
        for obj in all_objectives:
            writer.writerow([obj[0], obj[1]])

    objectives = np.array(all_objectives) if len(all_objectives) > 0 else np.empty((0, 2))
    print(f"[CASCADE RL] Evaluated {len(objectives)} designs total")
    return objectives


# ---------------------------------------------------------------------------
# ── PISTIL runners ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def run_pistil_ga(
    output_dir: str,
    pop_size: int = 10,
    n_gen: int = 5,
    model_name: str = "llama3-8b",
    allowed_num_cus: list = None,
    batch_bounds: tuple = (1, 64),
    kv_cache_bounds: tuple = (1024, 8192),
) -> np.ndarray:
    """
    Run the Pistil GA and return all evaluated objective points (N, 2) in
    evaluation order, read back from points.csv.
    """
    try:
        from api.Evaluator.gaPistil import runGAPistil
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from gaPistil import runGAPistil

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[PISTIL GA] pop_size={pop_size}, n_gen={n_gen}, model={model_name}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*60}")

    runGAPistil(
        pop_size=pop_size,
        n_gen=n_gen,
        model_name=model_name,
        allowed_num_cus=allowed_num_cus,
        batch_bounds=batch_bounds,
        kv_cache_bounds=kv_cache_bounds,
        output_dir=output_dir,
    )

    csv_path = os.path.join(output_dir, "points.csv")
    objectives = read_points_csv_pistil(csv_path)
    print(f"[PISTIL GA] Read {len(objectives)} points from {csv_path}")
    return objectives


def run_pistil_rl(
    output_dir: str,
    num_epochs: int = 20,
    mini_batch_size: int = 5,
    model_name: str = "llama3-8b",
    allowed_num_cus: list = None,
    batch_bounds: tuple = (1, 64),
    kv_cache_bounds: tuple = (1024, 8192),
) -> np.ndarray:
    try:
        from api.Evaluator.rlPistil import runPPOPistil
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from rlPistil import runPPOPistil

    os.makedirs(output_dir, exist_ok=True)

    # runPPOPistil expects the full CSV path as output_dir
    points_csv = os.path.join(output_dir, "points.csv")

    print(f"\n{'='*60}")
    print(f"[PISTIL RL] num_epochs={num_epochs}, mini_batch_size={mini_batch_size}")
    print(f"{'='*60}")

    try:
        all_designs, all_objectives, design_points = runPPOPistil(
            num_epochs=num_epochs,
            mini_batch_size=mini_batch_size,
            model_name=model_name,
            output_dir=points_csv,          # ← full CSV path, as rlPistil expects
            allowed_num_cus=allowed_num_cus,
            batch_bounds=batch_bounds,
            kv_cache_bounds=kv_cache_bounds,
        )
        objectives = np.array(all_objectives) if len(all_objectives) > 0 else np.empty((0, 2))
        print(f"[PISTIL RL] Evaluated {len(objectives)} designs total")
        return objectives

    except Exception as exc:
        # The optimisation itself may have completed fine; the crash often happens
        # in the post-processing block (second PistilEvaluator without output_dir).
        # Fall back to reading whatever was already written to points.csv.
        warnings.warn(
            f"[PISTIL RL] runPPOPistil raised an exception ({exc}); "
            f"attempting to load results from existing {points_csv}"
        )
        if os.path.exists(points_csv):
            objectives = read_points_csv_pistil(points_csv)
            print(f"[PISTIL RL] Loaded {len(objectives)} points from {points_csv}")
            return objectives
        raise   # re-raise only if the CSV is also missing

def run_pistil_random(
    output_dir: str,
    n_samples: int = 100,
    model_name: str = "llama3-8b",
    allowed_num_cus: list = None,
    batch_bounds: tuple = (1, 64),
    kv_cache_bounds: tuple = (1024, 8192),
    seed: int = 42,
) -> np.ndarray:
    """
    Run the Pistil Random Search baseline and return all evaluated objective
    points (N, 2) in evaluation order, read back from points.csv.
    """
    try:
        from api.Evaluator.rsPistil import runRandomPistil
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from rsPistil import runRandomPistil

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"[PISTIL RS] n_samples={n_samples}, model={model_name}")
    print(f"  output_dir: {output_dir}")
    print(f"{'='*60}")

    runRandomPistil(
        n_samples=n_samples,
        model_name=model_name,
        allowed_num_cus=allowed_num_cus,
        batch_bounds=batch_bounds,
        kv_cache_bounds=kv_cache_bounds,
        output_dir=output_dir,
        seed=seed,
    )

    csv_path = os.path.join(output_dir, "points.csv")
    objectives = read_points_csv_pistil(csv_path)
    print(f"[PISTIL RS] Read {len(objectives)} points from {csv_path}")
    return objectives

# ---------------------------------------------------------------------------
# ── Plotting ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Colour palette
COLORS = {
    "GA": "#2196F3",   # blue
    "RL": "#F44336",   # red
    "RS": "#4CAF50",   # green  ← ADD
}

MARKERS = {
    "GA": "o",
    "RL": "s",
    "RS": "^",   # ← ADD
}

LINE_STYLES = {
    "GA": "-",
    "RL": "--",
    "RS": ":",   # ← ADD
}


def plot_hv_comparison(
    ga_objectives: np.ndarray,
    rl_objectives: np.ndarray,
    ref_point: np.ndarray,
    title: str,
    save_path: str,
    rs_objectives: np.ndarray = None,   # ← ADD optional RS
) -> None:
    ga_evals, ga_hvs = hv_curve(ga_objectives, ref_point)
    rl_evals, rl_hvs = hv_curve(rl_objectives, ref_point)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        ga_evals, ga_hvs,
        color=COLORS["GA"], linestyle=LINE_STYLES["GA"],
        marker=MARKERS["GA"], markersize=4, linewidth=2,
        label=f"GA  (final HV={ga_hvs[-1]:.3e})" if ga_hvs else "GA",
    )
    ax.plot(
        rl_evals, rl_hvs,
        color=COLORS["RL"], linestyle=LINE_STYLES["RL"],
        marker=MARKERS["RL"], markersize=4, linewidth=2,
        label=f"RL  (final HV={rl_hvs[-1]:.3e})" if rl_hvs else "RL",
    )

    # ADD: plot RS curve when provided
    if rs_objectives is not None and len(rs_objectives) > 0:
        rs_evals, rs_hvs = hv_curve(rs_objectives, ref_point)
        ax.plot(
            rs_evals, rs_hvs,
            color=COLORS["RS"], linestyle=LINE_STYLES["RS"],
            marker=MARKERS["RS"], markersize=4, linewidth=2,
            label=f"RS  (final HV={rs_hvs[-1]:.3e})" if rs_hvs else "RS",
        )

    ax.set_xlabel("Number of Function Evaluations", fontsize=13)
    ax.set_ylabel("Hypervolume Indicator", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_pareto_fronts(
    ga_objectives: np.ndarray,
    rl_objectives: np.ndarray,
    title: str,
    save_path: str,
    xlabel: str = "Latency / Execution Time (ms)",
    ylabel: str = "Energy (mJ)",
    rs_objectives: np.ndarray = None,   # ← ADD optional RS
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    def _plot_method(objectives, name, color, marker):
        if len(objectives) == 0:
            return
        ax.scatter(
            objectives[:, 0], objectives[:, 1],
            color=color, marker=marker, alpha=0.25,
            s=30, label=f"{name} – all evaluations",
        )
        pf = np.unique(pareto_filter(objectives), axis=0)
        if len(pf) > 0:
            order = np.argsort(pf[:, 0])
            pf = pf[order]
            ax.plot(pf[:, 0], pf[:, 1], color=color, linewidth=2.5, zorder=5)
            ax.scatter(
                pf[:, 0], pf[:, 1],
                color=color, marker=marker, s=80,
                edgecolors="black", linewidths=0.8, zorder=6,
                label=f"{name} – Pareto front ({len(pf)} pts)",
            )

    _plot_method(ga_objectives, "GA", COLORS["GA"], MARKERS["GA"])
    _plot_method(rl_objectives, "RL", COLORS["RL"], MARKERS["RL"])

    # ADD: plot RS when provided
    if rs_objectives is not None and len(rs_objectives) > 0:
        _plot_method(rs_objectives, "RS", COLORS["RS"], MARKERS["RS"])

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")

# ---------------------------------------------------------------------------
# ── Main orchestration ───────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    # parser = argparse.ArgumentParser(
    #     description="Compare GA vs RL on CASCADE and PISTIL chiplet design models."
    # )
    # parser.add_argument("--cascade-only", action="store_true",
    #                     help="Only run CASCADE experiments.")
    # parser.add_argument("--pistil-only", action="store_true",
    #                     help="Only run PISTIL experiments.")
    # parser.add_argument("--skip-run", action="store_true",
    #                     help="Skip running optimizers; load existing points.csv files.")
    # parser.add_argument("--output-dir", type=str, default="comparison_results",
    #                     help="Root directory for all outputs (CSVs + plots).")

    # # ── CASCADE hyperparameters ──────────────────────────────────────────────
    # parser.add_argument("--cascade-trace", type=str,
    #                     default="gpt-j-65536-weighted",
    #                     help="Cascade trace name (no path, no .json).")
    # parser.add_argument("--cascade-ga-pop", type=int, default=10,
    #                     help="Cascade GA population size.")
    # parser.add_argument("--cascade-ga-gen", type=int, default=5,
    #                     help="Cascade GA number of generations.")
    # parser.add_argument("--cascade-rl-epochs", type=int, default=10,
    #                     help="Cascade RL number of epochs.")
    # parser.add_argument("--cascade-rl-batch", type=int, default=5,
    #                     help="Cascade RL mini-batch size.")

    # # ── PISTIL hyperparameters ───────────────────────────────────────────────
    # parser.add_argument("--pistil-model", type=str, default="llama3-8b",
    #                     help="Pistil model name (e.g. llama3-8b).")
    # parser.add_argument("--pistil-ga-pop", type=int, default=10,
    #                     help="Pistil GA population size.")
    # parser.add_argument("--pistil-ga-gen", type=int, default=5,
    #                     help="Pistil GA number of generations.")
    # parser.add_argument("--pistil-rl-epochs", type=int, default=10,
    #                     help="Pistil RL number of epochs.")
    # parser.add_argument("--pistil-rl-batch", type=int, default=5,
    #                     help="Pistil RL mini-batch size.")
    # parser.add_argument("--pistil-batch-min", type=int, default=1)
    # parser.add_argument("--pistil-batch-max", type=int, default=64)
    # parser.add_argument("--pistil-kv-min", type=int, default=1024)
    # parser.add_argument("--pistil-kv-max", type=int, default=8192)

    # args = parser.parse_args()
    param_a = 75
    param_b = 16

    args = {
        "cascade_only": False,
        "pistil_only": True,
        "skip_run": True,
        "output_dir": "comparison_results",
        "cascade_trace": "gpt-j-65536-weighted",
        "cascade_ga_pop": param_a,
        "cascade_ga_gen": param_b,
        "cascade_rl_epochs": param_a,
        "cascade_rl_batch": param_b,
        "pistil_model": "llama3-8b",
        "pistil_ga_pop": param_a,
        "pistil_ga_gen": param_b,
        "pistil_rl_epochs": param_a,
        "pistil_rl_batch": param_b,
        "pistil_batch_min": 1,
        "pistil_batch_max": 64,
        "pistil_kv_min": 1024,
        "pistil_kv_max": 8192,
        "pistil_rs_samples": param_a * param_b,
    }

    run_cascade = not args['pistil_only']
    run_pistil  = not args['cascade_only']

    # ── Directory layout ─────────────────────────────────────────────────────
    root = Path(args['output_dir'])
    cascade_ga_dir  = root / "cascade" / "ga"
    cascade_rl_dir  = root / "cascade" / "rl"
    pistil_ga_dir   = root / "pistil"  / "ga"
    pistil_rl_dir   = root / "pistil"  / "rl"
    pistil_rs_dir   = root / "pistil"  / "rs"
    plots_dir       = root / "plots"

    for d in [cascade_ga_dir, cascade_rl_dir,
              pistil_ga_dir,  pistil_rl_dir,
              pistil_rs_dir,  plots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 1.  CASCADE
    # =========================================================================
    cascade_ga_obj = np.empty((0, 2))
    cascade_rl_obj = np.empty((0, 2))

    if run_cascade:
        print("\n" + "#" * 70)
        print("#  CASCADE")
        print("#" * 70)

        if not args['skip_run']:
            # ---------- GA ----------
            try:
                cascade_ga_obj = run_cascade_ga(
                    output_dir=str(cascade_ga_dir),
                    pop_size=args['cascade_ga_pop'],
                    n_gen=args['cascade_ga_gen'],
                    trace=args['cascade_trace'],
                )
            except Exception as exc:
                warnings.warn(f"[CASCADE GA] Run failed: {exc}")
                import traceback; traceback.print_exc()

            # ---------- RL ----------
            try:
                cascade_rl_obj = run_cascade_rl(
                    output_dir=str(cascade_rl_dir),
                    num_epochs=args['cascade_rl_epochs'],
                    mini_batch_size=args['cascade_rl_batch'],
                    trace=args['cascade_trace'],
                )
            except Exception as exc:
                warnings.warn(f"[CASCADE RL] Run failed: {exc}")
                import traceback; traceback.print_exc()

        else:
            # ── Load from existing CSV files ──────────────────────────────────
            print("[CASCADE] --skip-run: loading existing points.csv files …")
            cascade_ga_csv = cascade_ga_dir / "points.csv"
            cascade_rl_csv = cascade_rl_dir / "points.csv"
            if cascade_ga_csv.exists():
                cascade_ga_obj = read_points_csv_cascade(str(cascade_ga_csv))
                print(f"  GA:  {len(cascade_ga_obj)} points loaded from {cascade_ga_csv}")
            else:
                warnings.warn(f"[CASCADE GA] CSV not found: {cascade_ga_csv}")
            if cascade_rl_csv.exists():
                cascade_rl_obj = read_points_csv_cascade(str(cascade_rl_csv))
                print(f"  RL:  {len(cascade_rl_obj)} points loaded from {cascade_rl_csv}")
            else:
                warnings.warn(f"[CASCADE RL] CSV not found: {cascade_rl_csv}")

        # ── Compute shared reference point ────────────────────────────────────
        all_cascade = [o for o in [cascade_ga_obj, cascade_rl_obj] if len(o) > 0]
        if all_cascade:
            cascade_ref = reference_point(all_cascade, margin=1.1)
            print(f"\n[CASCADE] Reference point: {cascade_ref}")

            # ── Plot 1 – Hypervolume vs. Evaluations ─────────────────────────
            hv_plot_path = str(plots_dir / "cascade_hypervolume.png")
            print("\n[CASCADE] Building hypervolume curves …")
            plot_hv_comparison(
                ga_objectives=cascade_ga_obj,
                rl_objectives=cascade_rl_obj,
                ref_point=cascade_ref,
                title="CASCADE – Hypervolume vs. Function Evaluations",
                save_path=hv_plot_path,
            )

            # ── Plot 2 – Pareto Front ─────────────────────────────────────────
            pf_plot_path = str(plots_dir / "cascade_pareto_front.png")
            print("[CASCADE] Building Pareto-front plot …")
            plot_pareto_fronts(
                ga_objectives=cascade_ga_obj,
                rl_objectives=cascade_rl_obj,
                title="CASCADE – Pareto Front (GA vs. RL)",
                save_path=pf_plot_path,
                xlabel="Execution Time (ms)",
                ylabel="Energy (mJ)",
            )
        else:
            warnings.warn("[CASCADE] No objectives collected – skipping plots.")

    # =========================================================================
    # 2.  PISTIL
    # =========================================================================
    pistil_ga_obj = np.empty((0, 2))
    pistil_rl_obj = np.empty((0, 2))
    pistil_rs_obj = np.empty((0, 2))

    pistil_batch_bounds  = (args['pistil_batch_min'], args['pistil_batch_max'])
    pistil_kv_bounds     = (args['pistil_kv_min'],   args['pistil_kv_max'])

    if run_pistil:
        print("\n" + "#" * 70)
        print("#  PISTIL")
        print("#" * 70)

        if not args['skip_run']:
            # ---------- GA ----------
            try:
                pistil_ga_obj = run_pistil_ga(
                    output_dir=str(pistil_ga_dir),
                    pop_size=args['pistil_ga_pop'],
                    n_gen=args['pistil_ga_gen'],
                    model_name=args['pistil_model'],
                    batch_bounds=pistil_batch_bounds,
                    kv_cache_bounds=pistil_kv_bounds,
                )
            except Exception as exc:
                warnings.warn(f"[PISTIL GA] Run failed: {exc}")
                import traceback; traceback.print_exc()

            # ---------- RL ----------
            try:
                pistil_rl_obj = run_pistil_rl(
                    output_dir=str(pistil_rl_dir),
                    num_epochs=args['pistil_rl_epochs'],
                    mini_batch_size=args['pistil_rl_batch'],
                    model_name=args['pistil_model'],
                    batch_bounds=pistil_batch_bounds,
                    kv_cache_bounds=pistil_kv_bounds,
                )
            except Exception as exc:
                warnings.warn(f"[PISTIL RL] Run failed: {exc}")
                import traceback; traceback.print_exc()

            # ---------- Random Search ----------
            try:
                pistil_rs_obj = run_pistil_random(
                    output_dir=str(pistil_rs_dir),
                    n_samples=args['pistil_rs_samples'],
                    model_name=args['pistil_model'],
                    batch_bounds=pistil_batch_bounds,
                    kv_cache_bounds=pistil_kv_bounds,
                )
            except Exception as exc:
                warnings.warn(f"[PISTIL RS] Run failed: {exc}")
                import traceback; traceback.print_exc()

        else:
            # ── Load from existing CSV files ──────────────────────────────────
            print("[PISTIL] --skip-run: loading existing points.csv files …")
            pistil_ga_csv = pistil_ga_dir / "points.csv"
            pistil_rl_csv = pistil_rl_dir / "points.csv"
            pistil_rs_csv = pistil_rs_dir / "points.csv"
            if pistil_ga_csv.exists():
                pistil_ga_obj = read_points_csv_pistil(str(pistil_ga_csv))
                print(f"  GA:  {len(pistil_ga_obj)} points loaded from {pistil_ga_csv}")
            else:
                warnings.warn(f"[PISTIL GA] CSV not found: {pistil_ga_csv}")
            if pistil_rl_csv.exists():
                pistil_rl_obj = read_points_csv_pistil(str(pistil_rl_csv))
                print(f"  RL:  {len(pistil_rl_obj)} points loaded from {pistil_rl_csv}")
            else:
                warnings.warn(f"[PISTIL RL] CSV not found: {pistil_rl_csv}")
            if pistil_rs_csv.exists():
                pistil_rs_obj = read_points_csv_pistil(str(pistil_rs_csv))
                print(f"  RS:  {len(pistil_rs_obj)} points loaded from {pistil_rs_csv}")
            else:
                warnings.warn(f"[PISTIL RS] CSV not found: {pistil_rs_csv}")

    # ── Compute shared reference point ────────────────────────────────────────
    all_pistil = [o for o in [pistil_ga_obj, pistil_rl_obj, pistil_rs_obj] if len(o) > 0]
    if all_pistil:
        # Log-transform PISTIL objectives for plotting and HV calculation
        # Raw values are saved to CSV; log values are only used for comparison
        pistil_ga_log = log_transform_objectives(pistil_ga_obj)
        pistil_rl_log = log_transform_objectives(pistil_rl_obj)
        pistil_rs_log = log_transform_objectives(pistil_rs_obj)

        pistil_ref = reference_point([pistil_ga_log, pistil_rl_log, pistil_rs_log], margin=1.1)
        print(f"\n[PISTIL] Reference point (log10 scale): {pistil_ref}")

        # ── Plot 3 – Hypervolume vs. Evaluations (log scale) ─────────────────
        hv_plot_path = str(plots_dir / "pistil_hypervolume.png")
        print("\n[PISTIL] Building hypervolume curves (log10 objectives) …")
        plot_hv_comparison(
            ga_objectives=pistil_ga_log,
            rl_objectives=pistil_rl_log,
            rs_objectives=pistil_rs_log,
            ref_point=pistil_ref,
            title="PISTIL – Hypervolume vs. Function Evaluations (log₁₀ objectives)",
            save_path=hv_plot_path,
        )

        # ── Plot 4 – Pareto Front (log scale) ────────────────────────────────
        pf_plot_path = str(plots_dir / "pistil_pareto_front.png")
        print("[PISTIL] Building Pareto-front plot (log10 objectives) …")
        plot_pareto_fronts(
            ga_objectives=pistil_ga_log,
            rl_objectives=pistil_rl_log,
            rs_objectives=pistil_rs_log,
            title="PISTIL – Pareto Front GA vs. RL vs. RS (log₁₀ objectives)",
            save_path=pf_plot_path,
            xlabel="log₁₀(Latency) [log ms]",
            ylabel="log₁₀(Energy) [log mJ]",
        )
    else:
        warnings.warn("[PISTIL] No objectives collected – skipping plots.")

    # =========================================================================
    # 3.  Summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def _summarise(
        name: str,
        ga_obj: np.ndarray,
        rl_obj: np.ndarray,
        ref: np.ndarray | None,
        rs_obj: np.ndarray = None,          # ADD
        hv_ga_obj: np.ndarray = None,
        hv_rl_obj: np.ndarray = None,
        hv_rs_obj: np.ndarray = None,       # ADD
    ) -> None:
        """Print a compact comparison table for one simulator."""
        hv_ga_src = hv_ga_obj if hv_ga_obj is not None else ga_obj
        hv_rl_src = hv_rl_obj if hv_rl_obj is not None else rl_obj
        hv_rs_src = hv_rs_obj if hv_rs_obj is not None else (rs_obj if rs_obj is not None else np.empty((0, 2)))  # ADD

        # CHANGE: widen the header to include RS column
        print(f"\n  {name}")
        print(f"  {'Metric':<35} {'GA':>14}  {'RL':>14}  {'RS':>14}")
        print(f"  {'-'*79}")

        def stat(arr, col):
            if arr is None or len(arr) == 0:
                return "N/A"
            valid = arr[arr[:, col] < 1e9]  # exclude penalty values
            if len(valid) == 0:
                return "N/A"
            return f"{np.min(valid[:, col]):.3e}"

        # CHANGE: add RS column to every printed row
        rs_len = len(rs_obj) if rs_obj is not None else 0
        print(f"  {'# evaluations':<35} {len(ga_obj):>14}  {len(rl_obj):>14}  {rs_len:>14}")
        print(f"  {'Best latency/exe-time (ms)':<35} "
              f"{stat(ga_obj, 0):>14}  {stat(rl_obj, 0):>14}  "
              f"{stat(rs_obj, 0) if rs_obj is not None else 'N/A':>14}")
        print(f"  {'Best energy (mJ)':<35} "
              f"{stat(ga_obj, 1):>14}  {stat(rl_obj, 1):>14}  "
              f"{stat(rs_obj, 1) if rs_obj is not None else 'N/A':>14}")

        if ref is not None:
            ga_hv = "N/A"
            rl_hv = "N/A"
            rs_hv = "N/A"  # ADD
            if len(hv_ga_src) > 0:
                pf = pareto_filter(hv_ga_src)
                ga_hv = f"{compute_hypervolume(pf, ref):.4e}"
            if len(hv_rl_src) > 0:
                pf = pareto_filter(hv_rl_src)
                rl_hv = f"{compute_hypervolume(pf, ref):.4e}"
            # ADD: RS hypervolume
            if len(hv_rs_src) > 0:
                pf = pareto_filter(hv_rs_src)
                rs_hv = f"{compute_hypervolume(pf, ref):.4e}"
            print(f"  {'Final hypervolume (log scale)':<35} "
                  f"{ga_hv:>14}  {rl_hv:>14}  {rs_hv:>14}")

        if len(ga_obj) > 0:
            pf_size = len(pareto_filter(ga_obj))
            print(f"  {'GA Pareto front size':<35} {pf_size:>14}")
        if len(rl_obj) > 0:
            pf_size = len(pareto_filter(rl_obj))
            print(f"  {'RL Pareto front size':<35} {pf_size:>14}")
        # ADD: RS Pareto front size
        if rs_obj is not None and len(rs_obj) > 0:
            pf_size = len(pareto_filter(rs_obj))
            print(f"  {'RS Pareto front size':<35} {pf_size:>14}")

    if run_cascade:
        # CASCADE has no RS, so rs_obj stays None (backward-compatible)
        _summarise(
            "CASCADE",
            cascade_ga_obj,
            cascade_rl_obj,
            cascade_ref if (len(cascade_ga_obj) > 0 or len(cascade_rl_obj) > 0) else None,
        )

    if run_pistil:
        pistil_ga_log = log_transform_objectives(pistil_ga_obj)
        pistil_rl_log = log_transform_objectives(pistil_rl_obj)
        pistil_rs_log = log_transform_objectives(pistil_rs_obj)  # ADD

        all_log = [arr for arr in [pistil_ga_log, pistil_rl_log, pistil_rs_log]  # CHANGE
                   if len(arr) > 0]
        pistil_ref_for_summary = (
            reference_point(all_log, margin=1.1) if all_log else None
        )

        # CHANGE: pass rs_obj and hv_rs_obj
        _summarise(
            "PISTIL (HV on log₁₀ scale)",
            pistil_ga_obj,
            pistil_rl_obj,
            pistil_ref_for_summary,
            rs_obj=pistil_rs_obj,           # ADD
            hv_ga_obj=pistil_ga_log,
            hv_rl_obj=pistil_rl_log,
            hv_rs_obj=pistil_rs_log,        # ADD
        )

    print(f"\nAll plots written to: {plots_dir.resolve()}")
    print("Done.")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()