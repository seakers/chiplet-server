import os
import re
from itertools import product

# Update this to your target directory
target_dir = "/n/home13/madiletta/workspace/netscratch_ln/transactional-sim-logs/traces"


# Sweep definitions
batch_sizes = [1, 2] + list(range(4, 129, 4))
sequence_lengths = list(range(2048, 8193, 1024))

model_chiplet_limits = {
#     "llama3-8b": 512,
#    "llama3-70b": 512,
    "llama3-405b": 1024,
}

def get_chiplets_for_model(model):
    max_chiplets = model_chiplet_limits[model]
    chiplets = list(range(16, 129, 4))
    if max_chiplets > 128:
        chiplets += list(range(128, max_chiplets + 1, 64))
    return chiplets

# Gather existing .pkl filenames
existing_files = set(f for f in os.listdir(target_dir) if f.endswith(".pkl"))

# Generate all expected combinations
expected_files = set()
for model in model_chiplet_limits:
    chiplets_list = get_chiplets_for_model(model)
    for chiplets, bs, sl in product(chiplets_list, batch_sizes, sequence_lengths):
        filename = f"{model}-chiplets-{chiplets}-bs-{bs}-sl-{sl}.pkl"
        expected_files.add(filename)

# Compare
missing_files = sorted(expected_files - existing_files)


# Report
if not missing_files:
    print("✅ All expected trace files are present.")
else:
    print(f"❌ Missing {len(missing_files)} trace files:")
    for f in missing_files:
        print(f"  {f}")

print("expected:", len(expected_files))
print("missing:", len(missing_files))
