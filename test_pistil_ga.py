#!/usr/bin/env python3
"""
Test script to run Pistil GA directly from terminal.
Tests with population=2, generations=2
"""

import sys
import os

# Add the Evaluator directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'chiplet-server', 'api', 'Evaluator'))

from gaPistil import runGAPistil
from pathlib import Path
import datetime

# Create output directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parent / "chiplet-server" / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean" / "dse" / "results" / f"test_run_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TESTING PISTIL GA")
print("=" * 80)
print(f"Population Size: 2")
print(f"Generations: 2")
print(f"Model: llama3-8b")
print(f"Output Directory: {output_dir}")
print("=" * 80)
print()

try:
    # Run the GA
    result = runGAPistil(
        pop_size=2,
        n_gen=2,
        model_name="llama3-8b",
        output_dir=str(output_dir),
    )
    
    print()
    print("=" * 80)
    print("GA RUN COMPLETED")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Points CSV: {output_dir / 'points.csv'}")
    
    # Check if points.csv exists and count points
    points_csv = output_dir / "points.csv"
    if points_csv.exists():
        with open(points_csv, 'r') as f:
            lines = f.readlines()
            header = lines[0] if lines else None
            data_rows = len(lines) - 1 if len(lines) > 1 else 0
        print(f"Points saved: {data_rows}")
        if data_rows > 0:
            print("✓ SUCCESS: Points were saved to CSV")
        else:
            print("⚠ WARNING: CSV exists but contains no data points (only header)")
    else:
        print("✗ ERROR: points.csv was not created")
    
    print("=" * 80)
    
except Exception as e:
    print()
    print("=" * 80)
    print("ERROR OCCURRED")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("=" * 80)
    sys.exit(1)

