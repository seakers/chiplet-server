"""
zoom_pistil_pareto.py

Loads existing PISTIL results from compare_optimizers output directories and
produces a zoomed-in Pareto-front plot showing only the non-dominated points
for GA, RL, and RS.

Usage:
    python zoom_pistil_pareto.py
    python zoom_pistil_pareto.py --output-dir comparison_results --save-path my_zoom.png
"""

import argparse
import warnings
from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import helpers from compare_optimizers.py
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))

from compare_optimizers import (
    read_points_csv_pistil,
    log_transform_objectives,
    pareto_filter,
    plot_pareto_fronts,
    COLORS,
    MARKERS,
)

def read_points_csv_pistil_zoom(csv_path: str, max_points: int = 1200) -> np.ndarray:
    """
    Read a Pistil points.csv (has a header row, columns include latency_ms
    and energy_mJ) and return an (N, 2) objectives array in order of
    appearance.

    Parameters
    ----------
    csv_path   : str  Path to the points.csv file.
    max_points : int  Maximum number of points to return, taken from the
                      **end** of the file (i.e. the most recent evaluations).
                      Defaults to 1200. Pass None to return all rows.
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

    if objectives and max_points is not None:
        objectives = objectives[-max_points:]  # keep only the last N points

    return np.array(objectives) if objectives else np.empty((0, 2))

# ---------------------------------------------------------------------------
# Zoomed Pareto-front plot
# ---------------------------------------------------------------------------

def plot_pistil_pareto_zoomed(
    ga_objectives: np.ndarray,
    rl_objectives: np.ndarray,
    rs_objectives: np.ndarray,
    save_path: str,
    padding: float = 0.05,
) -> None:
    """
    Plot only the Pareto-front points for GA, RL, and RS on the PISTIL
    problem (log10-transformed objectives), then zoom the axes tightly
    around those points with a small padding margin.

    Parameters
    ----------
    ga_objectives : np.ndarray  (N, 2) log10-transformed objectives for GA
    rl_objectives : np.ndarray  (N, 2) log10-transformed objectives for RL
    rs_objectives : np.ndarray  (N, 2) log10-transformed objectives for RS
    save_path     : str         File path to save the figure to
    padding       : float       Fractional axis padding around the tight
                                bounding box of all Pareto points (default 5%)
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    all_pf_points = []   # collect every Pareto point to compute axis limits

    def _plot_pf(objectives, name, color, marker):
        """Filter to Pareto front, plot connected line + scatter."""
        if objectives is None or len(objectives) == 0:
            print(f"  [{name}] No data – skipping.")
            return

        pf = np.unique(pareto_filter(objectives), axis=0)
        if len(pf) == 0:
            print(f"  [{name}] Empty Pareto front – skipping.")
            return

        # Sort by first objective (log-latency) for a clean step/line
        order = np.argsort(pf[:, 0])
        pf_sorted = pf[order]

        # Step-line connecting Pareto points
        ax.plot(
            pf_sorted[:, 0],
            pf_sorted[:, 1],
            color=color,
            linewidth=2.5,
            zorder=5,
            linestyle="-",
        )

        # Scatter the individual Pareto points on top
        ax.scatter(
            pf_sorted[:, 0],
            pf_sorted[:, 1],
            color=color,
            marker=marker,
            s=100,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=f"{name} – Pareto front ({len(pf)} pts)",
        )

        all_pf_points.append(pf_sorted)
        print(f"  [{name}] Plotted {len(pf)} Pareto-front points.")
        if name == "GA":
            print(f"    GA Pareto points (log-latency, log-energy):\n{pf_sorted}\n")

    _plot_pf(rs_objectives, "RS", COLORS["RS"], MARKERS["RS"])
    _plot_pf(rl_objectives, "RL", COLORS["RL"], MARKERS["RL"])
    _plot_pf(ga_objectives, "GA", COLORS["GA"], MARKERS["GA"])

    # ── Zoom axes tightly around all Pareto points ────────────────────────
    if all_pf_points:
        combined = np.vstack(all_pf_points)

        x_min, x_max = combined[:, 0].min(), combined[:, 0].max()
        y_min, y_max = combined[:, 1].min(), combined[:, 1].max()

        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0

        ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
        ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    else:
        warnings.warn("No Pareto-front points found for any method – axes not zoomed.")

    ax.set_xlabel("log₁₀(Latency) [log ms]", fontsize=13)
    ax.set_ylabel("log₁₀(Energy) [log mJ]", fontsize=13)
    ax.set_title(
        "PISTIL – Zoomed Pareto Front: GA vs. RL vs. RS (log₁₀ objectives)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.35)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"\n  Saved zoomed plot → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Zoomed Pareto-front plot for PISTIL (GA vs RL vs RS)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Root output directory used by compare_optimizers.py "
             "(default: comparison_results).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Where to save the plot. Defaults to "
             "<output-dir>/plots/pistil_pareto_zoomed.png.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Fractional padding around the Pareto-front bounding box "
             "(default: 0.05 = 5%%).",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)

    # ── Paths to the three points.csv files ──────────────────────────────────
    ga_csv = root / "pistil" / "ga" / "points.csv"
    rl_csv = root / "pistil" / "rl" / "points.csv"
    rs_csv = root / "pistil" / "rs" / "points.csv"

    save_path = (
        args.save_path
        if args.save_path
        else str(root / "plots" / "pistil_pareto_zoomed.png")
    )

    # ── Load raw objectives ───────────────────────────────────────────────────
    print("\nLoading PISTIL CSV files …")

    ga_obj = np.empty((0, 2))
    rl_obj = np.empty((0, 2))
    rs_obj = np.empty((0, 2))

    if ga_csv.exists():
        ga_obj = read_points_csv_pistil_zoom(str(ga_csv))
        print(f"  GA : {len(ga_obj):>5} points  ← {ga_csv}")
    else:
        warnings.warn(f"[GA] CSV not found: {ga_csv}")

    if rl_csv.exists():
        rl_obj = read_points_csv_pistil_zoom(str(rl_csv))
        print(f"  RL : {len(rl_obj):>5} points  ← {rl_csv}")
    else:
        warnings.warn(f"[RL] CSV not found: {rl_csv}")

    if rs_csv.exists():
        rs_obj = read_points_csv_pistil_zoom(str(rs_csv))
        print(f"  RS : {len(rs_obj):>5} points  ← {rs_csv}")
    else:
        warnings.warn(f"[RS] CSV not found: {rs_csv}")

    if len(ga_obj) == 0 and len(rl_obj) == 0 and len(rs_obj) == 0:
        print("\nNo data found. Run compare_optimizers.py first to generate results.")
        return

    # ── Log-transform (same as compare_optimizers.py does for PISTIL) ────────
    print("\nApplying log₁₀ transform to objectives …")
    ga_log = log_transform_objectives(ga_obj)
    rl_log = log_transform_objectives(rl_obj)
    rs_log = log_transform_objectives(rs_obj)

    # ── Build plot ────────────────────────────────────────────────────────────
    plots_dir = Path(save_path).parent
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_pareto_fronts(
        ga_objectives=ga_log,
        rl_objectives=rl_log,
        rs_objectives=rs_log,
        save_path=str(plots_dir / "pistil_pareto_front.png"),
        title="PISTIL – Pareto Front: GA vs. RL vs. RS (log₁₀ objectives)",
        xlabel="log₁₀(Latency) [log ms]",
        ylabel="log₁₀(Energy) [log mJ]",
    )

    # print("\nBuilding zoomed Pareto-front plot …")
    # plot_pistil_pareto_zoomed(
    #     ga_objectives=ga_log,
    #     rl_objectives=rl_log,
    #     rs_objectives=rs_log,
    #     save_path=save_path,
    #     padding=args.padding,
    # )

    print("\nDone.")


if __name__ == "__main__":
    main()