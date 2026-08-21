import pandas as pd
import numpy as np
from pathlib import Path

def verify_energy_scaling(csv_path):
    df = pd.read_csv(csv_path)
    
    print("=" * 70)
    print("ENERGY SCALING VERIFICATION")
    print("=" * 70)
    
    # ----------------------------------------------------------------
    # Check 1: Is energy_mJ consistent with average_power_W * latency?
    # (single-chiplet basis)
    # ----------------------------------------------------------------
    df["expected_energy_single_chiplet_mJ"] = (
        df["average_power_W"] * (df["latency_ms"] / 1000.0) * 1000.0
    )
    df["energy_ratio_single"] = (
        df["energy_mJ"] / df["expected_energy_single_chiplet_mJ"]
    )

    # ----------------------------------------------------------------
    # Check 2: Is energy_mJ consistent with system_power_W * latency?
    # (system-total basis)
    # ----------------------------------------------------------------
    df["expected_energy_system_mJ"] = (
        df["system_power_W"] * (df["latency_ms"] / 1000.0) * 1000.0
    )
    df["energy_ratio_system"] = (
        df["energy_mJ"] / df["expected_energy_system_mJ"]
    )

    # ----------------------------------------------------------------
    # Check 3: system_power_W == num_chiplets * average_power_W?
    # (verifies gaPistil.py scaling is applied correctly [1])
    # ----------------------------------------------------------------
    df["expected_system_power_W"] = df["num_chiplets"] * df["average_power_W"]
    df["system_power_correct"] = np.isclose(
        df["system_power_W"], df["expected_system_power_W"], rtol=1e-3
    )

    # ----------------------------------------------------------------
    # Check 4: Compute implied system energy for reference
    # ----------------------------------------------------------------
    df["implied_system_energy_mJ"] = (
        df["system_power_W"] * (df["latency_ms"] / 1000.0) * 1000.0
    )
    df["implied_system_energy_per_token_mJ"] = (
        df["implied_system_energy_mJ"] / df["batch_size"]
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n--- Check 1: energy_mJ vs single-chiplet power * latency ---")
    print(f"  Mean ratio   : {df['energy_ratio_single'].mean():.4f}  (expect ~1.0 if per-chiplet)")
    print(f"  Std  ratio   : {df['energy_ratio_single'].std():.4f}")
    print(f"  Min / Max    : {df['energy_ratio_single'].min():.4f} / {df['energy_ratio_single'].max():.4f}")

    print("\n--- Check 2: energy_mJ vs system power * latency ---")
    print(f"  Mean ratio   : {df['energy_ratio_system'].mean():.4f}  (expect ~1.0 if system-total)")
    print(f"  Std  ratio   : {df['energy_ratio_system'].std():.4f}")
    print(f"  Min / Max    : {df['energy_ratio_system'].min():.4f} / {df['energy_ratio_system'].max():.4f}")

    print("\n--- Check 3: system_power_W == num_chiplets * average_power_W ---")
    pct_correct = df["system_power_correct"].mean() * 100
    print(f"  Rows passing : {pct_correct:.1f}%")
    bad = df[~df["system_power_correct"]][
        ["num_chiplets", "average_power_W", "system_power_W", "expected_system_power_W"]
    ]
    if len(bad) > 0:
        print("  Failing rows:")
        print(bad.to_string(index=False))

    print("\n--- Check 4: Implied SYSTEM energy (for reference) ---")
    print(f"  Mean system energy/token (mJ) : "
          f"{df['implied_system_energy_per_token_mJ'].mean():.4f}")
    print(f"  Reported energy/token (mJ)   : "
          f"{df['energy_per_token_mJ'].mean():.4f}")
    print(f"  Scale factor (system/reported): "
          f"{(df['implied_system_energy_per_token_mJ'] / df['energy_per_token_mJ']).mean():.1f}x"
          f"  ← should be ~num_chiplets if energy is per-chiplet")

    # ----------------------------------------------------------------
    # Per-num_cus breakdown
    # ----------------------------------------------------------------
    print("\n--- Per num_chiplets breakdown ---")
    grp = df.groupby("num_chiplets").agg(
        n=("energy_mJ", "count"),
        avg_energy_ratio_single=("energy_ratio_single", "mean"),
        avg_energy_ratio_system=("energy_ratio_system", "mean"),
        avg_reported_energy_per_token=("energy_per_token_mJ", "mean"),
        avg_system_energy_per_token=("implied_system_energy_per_token_mJ", "mean"),
    ).reset_index()
    print(grp.to_string(index=False))

    # ----------------------------------------------------------------
    # Save augmented CSV
    # ----------------------------------------------------------------
    out_path = csv_path.replace(".csv", "_verified.csv")
    df.to_csv(out_path, index=False)
    print(f"\nAugmented CSV saved to: {out_path}")

    return df


# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------
RESULTS_ROOT = Path("api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results")

# All energy-related columns to scale by num_chiplets
# Based on the points.csv header [1] and the confirmed per-chiplet 
# energy finding from save_results.py [12]
ENERGY_COLS = [
    "energy_mJ",
    "energy_per_inference_mJ",
    "energy_per_token_mJ",
]

# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------
def fix_energy_scaling(csv_path: Path, dry_run: bool = False):
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    if "num_chiplets" not in df.columns:
        print(f"  [SKIP] No 'num_chiplets' column found: {csv_path}")
        return None

    # Check which energy cols are actually present in this file
    cols_to_fix = [c for c in ENERGY_COLS if c in df.columns]
    missing = [c for c in ENERGY_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] Columns not found (will skip): {missing}")

    if dry_run:
        print(f"  [DRY RUN] Would scale {cols_to_fix} by num_chiplets")
        return df

    # Apply scaling
    for col in cols_to_fix:
        df[col] = df[col] * df["num_chiplets"]

    return df


def main(dry_run=False):
    csv_files = sorted(RESULTS_ROOT.glob("*/points.csv"))

    if not csv_files:
        print(f"No points.csv files found under {RESULTS_ROOT}")
        return

    print(f"Found {len(csv_files)} points.csv file(s)\n")
    print("=" * 70)

    total_rows_fixed = 0

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path}")
        print(f"  Subfolder : {csv_path.parent.name}")

        df = fix_energy_scaling(csv_path, dry_run=dry_run)
        if df is None:
            continue

        n_rows = len(df)
        print(f"  Rows      : {n_rows}")
        print(f"  Chiplet counts: {sorted(df['num_chiplets'].unique())}")

        if not dry_run:
            # Backup original before overwriting
            backup_path = csv_path.with_name("points_original.csv")
            if not backup_path.exists():
                import shutil
                shutil.copy(csv_path, backup_path)
                print(f"  Backup    : {backup_path.name}")
            else:
                print(f"  Backup    : already exists, skipping")

            # Save corrected file in place
            df.to_csv(csv_path, index=False)
            print(f"  Saved     : {csv_path.name} (corrected)")
            total_rows_fixed += n_rows

            # Spot-check: print first row energy values
            row = df.iloc[0]
            print(f"  Spot check (row 0):")
            print(f"    num_chiplets          = {row['num_chiplets']}")
            if "energy_mJ" in df.columns:
                print(f"    energy_mJ             = {row['energy_mJ']:.4f}")
            if "energy_per_token_mJ" in df.columns:
                print(f"    energy_per_token_mJ   = {row['energy_per_token_mJ']:.4f}")
            if "average_power_W" in df.columns:
                print(f"    average_power_W       = {row['average_power_W']:.4f}")

    print("\n" + "=" * 70)
    if dry_run:
        print("DRY RUN complete — no files were modified.")
    else:
        print(f"Done. Fixed {total_rows_fixed} total rows across {len(csv_files)} file(s).")
        print("Originals saved as points_original.csv in each subfolder.")


if __name__ == "__main__":

    main(dry_run=False)


# if __name__ == "__main__":
#     csv_file = "api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/pistil_run_iso_batch_kv/points.csv"
#     df = verify_energy_scaling(csv_file)