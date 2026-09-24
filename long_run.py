import numpy as np

from api.Evaluator.gaPistil import runGAPistil


pop_size = 5
n_gen = 5
model_name = "llama3-8b"
objectives = ["Energy per Token", "Latency per Token", "System Cost"]
# objectives = ['system_compute_TOPS', 'system_bandwidth_TBps', 'system_capacity_GB']
output_dir = f"results/pistil_run_{model_name}_{pop_size}pop_{n_gen}gen"


runGAPistil(
    pop_size=pop_size,
    n_gen=n_gen,
    model_name=model_name,
    objectives=objectives,
    output_dir=output_dir,
    batch_bounds=(1, 1),
    kv_cache_bounds=(8192, 8192),
)
