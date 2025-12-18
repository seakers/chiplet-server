import os
import csv
import time
from pathlib import Path

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Dynamically load PistilSimulator from pistil_runner.py since the directory
# name contains hyphens and is not an importable Python package.
import importlib.util as _importlib_util

_SIM_ROOT = Path(__file__).parent / "sim-v2-4-pistil-sim-clean"
_PISTIL_RUNNER_PATH = _SIM_ROOT / "pistil_runner.py"
_spec = _importlib_util.spec_from_file_location("pistil_runner", _PISTIL_RUNNER_PATH)
_pistil_module = _importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pistil_module)
PistilSimulator = _pistil_module.PistilSimulator


def _round_to_nearest(value, choices):
    """
    Helper to snap a (possibly mutated) numeric value to the closest allowed choice.
    """
    choices = np.array(choices, dtype=float)
    idx = (np.abs(choices - float(value))).argmin()
    return choices[idx]


class PistilProblem(ElementwiseProblem):
    """
    GA problem wrapper for the Pistil simulator.

    Decision vector x encodes:
        x[0] -> num_cus                  (int, multiple of 4, from allowed_num_cus)
        x[1] -> tmacs                    (int, from TMAC_CHOICES)
        x[2] -> mem_buf_cap              (float, from MEM_BUF_CHOICES)
        x[3] -> net_buf_cap              (float, from MEM_BUF_CHOICES)
        x[4] -> mem_bank_groups          (int, from BANK_GROUP_CHOICES)
        x[5] -> mem_ranks                (int, from RANK_CHOICES)
        x[6] -> mem_frac_bank_cap        (float, from FRAC_BANK_CHOICES)
        x[7] -> batch_size               (int, power of 2, from BATCH_CHOICES)
        x[8] -> kv_cache                 (int, power of 2, from KV_CACHE_CHOICES)
    """

    TMAC_CHOICES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    MEM_BUF_CHOICES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    BANK_GROUP_CHOICES = [1, 2, 3, 4]
    RANK_CHOICES = [1, 2, 3, 4]
    FRAC_BANK_CHOICES = [0.5, 0.75, 1.0]
    
    @staticmethod
    def _generate_power_of_2_choices(min_val, max_val):
        """Generate list of powers of 2 within the given bounds."""
        choices = []
        power = 0
        while True:
            value = 2 ** power
            if value < min_val:
                power += 1
                continue
            if value > max_val:
                break
            choices.append(value)
            power += 1
        return choices if choices else [min_val]  # Fallback to min_val if no powers of 2 in range

    def __init__(
        self,
        model_name: str,
        allowed_num_cus=None,
        batch_bounds=(1, 64),
        kv_cache_bounds=(1024, 8192),
        output_dir: str = None,
    ):
        if allowed_num_cus is None:
            # Default reasonable Pistil-friendly CU counts (must be multiple of 4)
            allowed_num_cus = [16, 32, 64, 96, 128]

        self.model_name = model_name
        self.allowed_num_cus = sorted(set(int(v) for v in allowed_num_cus if v % 4 == 0))
        if not self.allowed_num_cus:
            raise ValueError("allowed_num_cus must contain at least one multiple-of-4 value.")

        self.batch_min, self.batch_max = batch_bounds
        self.kvcache_min, self.kvcache_max = kv_cache_bounds
        
        # Generate power-of-2 choices for batch_size and kv_cache
        self.BATCH_CHOICES = self._generate_power_of_2_choices(self.batch_min, self.batch_max)
        self.KV_CACHE_CHOICES = self._generate_power_of_2_choices(self.kvcache_min, self.kvcache_max)
        
        # Update bounds to match the actual power-of-2 choices
        self.batch_min = min(self.BATCH_CHOICES)
        self.batch_max = max(self.BATCH_CHOICES)
        self.kvcache_min = min(self.KV_CACHE_CHOICES)
        self.kvcache_max = max(self.KV_CACHE_CHOICES)

        self.sim = PistilSimulator()

        # Where points.csv and context files will be written for this run
        self.output_dir = output_dir
        if self.output_dir is None:
            # Fallback to a default under the Pistil sim root
            self.output_dir = os.path.join(self.sim.sim_root, "dse", "results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track CSV header initialization
        self.points_csv_initialized = False
        self.points_csv_path = os.path.join(self.output_dir, "points.csv")
        
        # Initialize CSV file with header immediately
        self._initialize_points_csv()

        # 9 decision variables (see class docstring)
        n_var = 9
        # Lower/upper bounds are broad; we will snap to Pistil-native discrete sets in _evaluate
        xl = np.array(
            [
                min(self.allowed_num_cus),
                min(self.TMAC_CHOICES),
                min(self.MEM_BUF_CHOICES),
                min(self.MEM_BUF_CHOICES),
                min(self.BANK_GROUP_CHOICES),
                min(self.RANK_CHOICES),
                min(self.FRAC_BANK_CHOICES),
                self.batch_min,
                self.kvcache_min,
            ],
            dtype=float,
        )
        xu = np.array(
            [
                max(self.allowed_num_cus),
                max(self.TMAC_CHOICES),
                max(self.MEM_BUF_CHOICES),
                max(self.MEM_BUF_CHOICES),
                max(self.BANK_GROUP_CHOICES),
                max(self.RANK_CHOICES),
                max(self.FRAC_BANK_CHOICES),
                self.batch_max,
                self.kvcache_max,
            ],
            dtype=float,
        )

        super().__init__(
            n_var=n_var,
            n_obj=2,  # time (ms), energy (mJ)
            n_constr=0,
            xl=xl,
            xu=xu,
            vtype=float,
        )

    def _decode_vector(self, x):
        """
        Map raw GA vector to a valid Pistil parameter dictionary.
        """
        (
            num_cus,
            tmacs,
            mem_buf_cap,
            net_buf_cap,
            bank_groups,
            ranks,
            frac_bank_cap,
            batch_size,
            kv_cache,
        ) = x

        # Snap to valid discrete sets where Pistil defines choices
        num_cus = int(_round_to_nearest(num_cus, self.allowed_num_cus))
        # enforce multiple-of-4 just in case
        if num_cus % 4 != 0:
            num_cus = max(4, 4 * round(num_cus / 4))

        tmacs = int(_round_to_nearest(tmacs, self.TMAC_CHOICES))
        mem_buf_cap = float(_round_to_nearest(mem_buf_cap, self.MEM_BUF_CHOICES))
        net_buf_cap = float(_round_to_nearest(net_buf_cap, self.MEM_BUF_CHOICES))
        bank_groups = int(_round_to_nearest(bank_groups, self.BANK_GROUP_CHOICES))
        ranks = int(_round_to_nearest(ranks, self.RANK_CHOICES))
        frac_bank_cap = float(_round_to_nearest(frac_bank_cap, self.FRAC_BANK_CHOICES))

        # Snap batch_size and kv_cache to power-of-2 choices
        batch_size = int(_round_to_nearest(batch_size, self.BATCH_CHOICES))
        kv_cache = int(_round_to_nearest(kv_cache, self.KV_CACHE_CHOICES))

        params = {
            "num_cus": num_cus,
            "num_tmacs": tmacs,
            "mem_buf_cap": mem_buf_cap,
            "net_buf_cap": net_buf_cap,
            "mem_banks_per_group": bank_groups,
            "mem_ranks": ranks,
            "mem_frac_bank_cap": frac_bank_cap,
            "model": self.model_name,
            "batch_size": batch_size,
            "kv_cache": kv_cache,
            # Fixed / run-level constants aligned with arg_parser defaults
            "w_dtype": 0.5,
            "kv_dtype": 1.0,
            "prefill": "False",
            "prefill_chunk_size": 0,
            "prefill_cached": 0,
            "sim_num_layers": -1,
            "lm_head": "False",
            "plot_exe": "False",
            "gen_trace": True,
            "sim_standalone": True,
            "base_config": "pistil-sys-base.json",
        }
        return params

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Run a single Pistil simulation for design vector x and return [time_ms, energy_mJ].
        Also saves all metrics to points.csv for plotting/analysis.
        """
        params = self._decode_vector(x)
        
        # Track evaluation count for logging
        if not hasattr(self, '_eval_count'):
            self._eval_count = 0
        self._eval_count += 1

        import time as time_module
        eval_start_time = time_module.time()
        
        print(f"[PISTIL GA] Evaluation #{self._eval_count}: Starting simulation...")
        print(f"  Design: {params['num_cus']} CUs, {params['num_tmacs']} TMACs, "
              f"batch={params['batch_size']}, kv_cache={params['kv_cache']}, "
              f"mem_buf={params['mem_buf_cap']}, net_buf={params['net_buf_cap']}")
        print(f"  [Note: Simulation may take several minutes. Pistil is running...]")

        try:
            self.sim.run_dse_point(params)
            eval_elapsed = time_module.time() - eval_start_time
            print(f"  [Simulation completed in {eval_elapsed:.1f} seconds]")
            metrics_dict = self._load_all_metrics(params)
            time_ms = metrics_dict["latency_ms"]
            energy_mJ = metrics_dict["energy_mJ"]
            
            # Save to points.csv (similar to cascade)
            self._save_to_points_csv(params, metrics_dict)
            
            print(f"[PISTIL GA] Evaluation #{self._eval_count}: ✓ Completed successfully")
            print(f"  Results: latency={time_ms:.2f}ms, energy={energy_mJ:.2f}mJ")
            print(f"  Per-token: latency={metrics_dict.get('latency_per_token_ms', 0):.3f}ms, "
                  f"energy={metrics_dict.get('energy_per_token_mJ', 0):.3f}mJ")
            print(f"  Saved to CSV: {self.points_csv_path}")
        except Exception as e:
            # On failure, penalize this design heavily so GA avoids it
            print(f"[PISTIL GA] Evaluation #{self._eval_count}: ✗ ERROR - {e}")
            import traceback
            traceback.print_exc()
            time_ms, energy_mJ = 1e9, 1e9

        out["F"] = [time_ms, energy_mJ]

    def _load_all_metrics(self, params):
        """
        Load ALL metrics from the Pistil results CSV and compute derived metrics.
        Returns a comprehensive dictionary with all metrics for plotting/analysis.
        """
        results_dir = os.path.join(self.sim.sim_root, self.sim.results_dir)
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Pistil results directory not found: {results_dir}")

        # Heuristic: choose the most recently modified CSV in results_dir
        csv_files = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError(f"No Pistil result CSVs found in: {results_dir}")

        latest_csv = max(csv_files, key=os.path.getmtime)

        # Parse the summary CSV which has "Metric,Value" rows from save_stats_to_csv
        metrics = {}
        with open(latest_csv, "r") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) != 2:
                    continue
                key, value = row
                try:
                    # Try to convert to float, but keep as string if it fails (e.g., "False", model names)
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

        # Derive latency and energy consistent with save_results' print logic
        total_latency_s = float(metrics.get("total_latency_s", 0.0))
        total_energy_J = float(metrics.get("total_energy_J", 0.0))
        num_layers = float(metrics.get("num_layers", 1.0))
        sim_num_layers = float(metrics.get("sim_num_layers", -1.0))
        batch_size = float(metrics.get("batch_size", params["batch_size"]))

        # Scale if simulation only ran subset of layers
        if sim_num_layers != num_layers and sim_num_layers != -1:
            latency_ms = total_latency_s * num_layers / sim_num_layers * 1000.0
            energy_mJ = total_energy_J * num_layers / sim_num_layers * 1000.0
        else:
            latency_ms = total_latency_s * 1000.0
            energy_mJ = total_energy_J * 1000.0

        # Compute derived metrics
        latency_per_token_ms = latency_ms / batch_size if batch_size > 0 else 0.0
        energy_per_inference_mJ = energy_mJ
        energy_per_token_mJ = energy_mJ / batch_size if batch_size > 0 else 0.0
        
        # Average power (system-wide)
        num_cus = float(metrics.get("num_chiplets", params["num_cus"]))
        avg_power_W = float(metrics.get("average_power_W", 0.0))
        system_power_W = num_cus * avg_power_W
        
        # Cost metrics
        system_cost = float(metrics.get("system_cost", 0.0))
        
        # Prefill tokens per second (if applicable)
        prefill_batch = metrics.get("prefill_batch", False)
        prefill_chunk_size = float(metrics.get("prefill_chunk_size", 0.0))
        if prefill_batch and prefill_chunk_size > 0 and latency_ms > 0:
            prefill_tokens_per_sec = (prefill_chunk_size * batch_size) / (latency_ms / 1000.0)
        else:
            prefill_tokens_per_sec = batch_size / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        
        # Utilization metrics
        avg_comp_util = float(metrics.get("avg_comp_util", 0.0))
        avg_mem_util = float(metrics.get("avg_mem_util", 0.0))
        
        # Build comprehensive metrics dictionary
        # Start with ALL metrics from the CSV (this ensures we capture everything)
        all_metrics = dict(metrics)
        
        # Override/add computed/derived metrics (these are more accurate or normalized)
        all_metrics.update({
            # Core objectives (for GA) - computed from raw metrics
            "latency_ms": float(latency_ms),
            "energy_mJ": float(energy_mJ),
            
            # Derived performance metrics
            "latency_per_token_ms": latency_per_token_ms,
            "energy_per_inference_mJ": energy_per_inference_mJ,
            "energy_per_token_mJ": energy_per_token_mJ,
            "prefill_tokens_per_sec": prefill_tokens_per_sec,
            
            # Power metrics (ensure we have computed values)
            "average_power_W": avg_power_W,
            "system_power_W": system_power_W,
            # Keep original domain-specific power if available
            "average_power_mem_W": float(metrics.get("average_power_mem_W", 0.0)),
            "average_power_comp_W": float(metrics.get("average_power_comp_W", 0.0)),
            "average_power_net_W": float(metrics.get("average_power_net_W", 0.0)),
            
            # Cost metrics (ensure we have all cost breakdowns)
            "system_cost": system_cost,
            "chiplet_silicon_cost": float(metrics.get("chiplet_silicon_cost", 0.0)),
            "memory_cost_2xHBLC": float(metrics.get("memory_cost_2xHBLC", 0.0)),
            "package_cost": float(metrics.get("package_cost", 0.0)),
            "package_silicon_cost": float(metrics.get("package_silicon_cost", 0.0)),
            "package_memory_cost": float(metrics.get("package_memory_cost", 0.0)),
            "package_substrate_cost": float(metrics.get("package_substrate_cost", 0.0)),
            "system_silicon_cost": float(metrics.get("system_silicon_cost", 0.0)),
            "system_memory_cost": float(metrics.get("system_memory_cost", 0.0)),
            "system_substrate_cost": float(metrics.get("system_substrate_cost", 0.0)),
            "system_pcb_cost": float(metrics.get("system_pcb_cost", 0.0)),
            
            # Utilization metrics
            "avg_comp_util": avg_comp_util,
            "avg_mem_util": avg_mem_util,
            
            # System configuration (ensure we have correct values)
            "num_chiplets": num_cus,
            "system_compute_TOPS": float(metrics.get("system_compute_TOPS", 0.0)),
            "system_bandwidth_TBps": float(metrics.get("system_bandwidth_TBps", 0.0)),
            "system_capacity_GB": float(metrics.get("system_capacity_GB", 0.0)),
            
            # Additional system metrics from save_results
            "num_packages": float(metrics.get("num_packages", 0.0)),
            "hblc_capacity_GB": float(metrics.get("hblc_capacity_GB", 0.0)),
            "hblc_bandwidth_GBps": float(metrics.get("hblc_bandwidth_GBps", 0.0)),
            "hblc_bw_per_capacity": float(metrics.get("hblc_bw_per_capacity", 0.0)),
            "peak_buffer_MB": float(metrics.get("peak_buffer_MB", 0.0)),
            "avg_cache_util_MB": float(metrics.get("avg_cache_util_MB", 0.0)),
            "model_weight_capacity_GB": float(metrics.get("model_weight_capacity_GB", 0.0)),
            "kv_cache_capacity_per_batch_GB": float(metrics.get("kv_cache_capacity_per_batch_GB", 0.0)),
            "total_model_capacity_GB": float(metrics.get("total_model_capacity_GB", 0.0)),
        })
        
        return all_metrics
    
    def _initialize_points_csv(self):
        """
        Initialize points.csv with header row.
        Includes ALL metrics from pistil_runner for comprehensive data capture.
        """
        decision_cols = [
            "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
            "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
            "batch_size", "kv_cache"
        ]
        
        # Include ALL metrics from pistil_runner for comprehensive data capture
        metric_cols = [
            # Core objectives (for GA)
            "latency_ms", "energy_mJ",
            # Derived performance metrics
            "latency_per_token_ms", "energy_per_inference_mJ", "energy_per_token_mJ",
            "prefill_tokens_per_sec",
            # Power metrics
            "average_power_W", "system_power_W",
            "average_power_mem_W", "average_power_comp_W", "average_power_net_W",
            # Cost metrics (all cost breakdowns)
            "system_cost",
            "chiplet_silicon_cost", "memory_cost_2xHBLC", "package_cost",
            "package_silicon_cost", "package_memory_cost", "package_substrate_cost",
            "system_silicon_cost", "system_memory_cost", "system_substrate_cost", "system_pcb_cost",
            # Utilization metrics
            "avg_comp_util", "avg_mem_util",
            # System configuration
            "system_compute_TOPS", "system_bandwidth_TBps", "system_capacity_GB",
            "num_packages", "num_chiplets",
            # Memory/HBLC metrics
            "hblc_capacity_GB", "hblc_bandwidth_GBps", "hblc_bw_per_capacity",
            "peak_buffer_MB", "avg_cache_util_MB",
            # Model capacity metrics
            "model_weight_capacity_GB", "kv_cache_capacity_per_batch_GB", "total_model_capacity_GB",
            # Additional system parameters
            "cores_per_cu", "mem_buffer_size", "net_buffer_size", "tmacs_per_core",
            "hblc_bank_groups", "hblc_ranks", "hblc_frac_bank_cap"
        ]
        
        # Only write header if file doesn't exist or is empty
        if not os.path.exists(self.points_csv_path) or os.path.getsize(self.points_csv_path) == 0:
            with open(self.points_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = decision_cols + metric_cols
                writer.writerow(header)
            self.points_csv_initialized = True
            print(f"Initialized points.csv at {self.points_csv_path}")
    
    def _save_to_points_csv(self, params, metrics_dict):
        """
        Save design point with all metrics to points.csv (similar to cascade's approach).
        Includes decision variables + ALL performance/cost metrics from pistil_runner.
        """
        # Define CSV columns: decision variables first, then metrics
        decision_cols = [
            "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
            "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
            "batch_size", "kv_cache"
        ]
        
        # Include ALL metrics from pistil_runner for comprehensive data capture
        metric_cols = [
            # Core objectives (for GA)
            "latency_ms", "energy_mJ",
            # Derived performance metrics
            "latency_per_token_ms", "energy_per_inference_mJ", "energy_per_token_mJ",
            "prefill_tokens_per_sec",
            # Power metrics
            "average_power_W", "system_power_W",
            "average_power_mem_W", "average_power_comp_W", "average_power_net_W",
            # Cost metrics (all cost breakdowns)
            "system_cost",
            "chiplet_silicon_cost", "memory_cost_2xHBLC", "package_cost",
            "package_silicon_cost", "package_memory_cost", "package_substrate_cost",
            "system_silicon_cost", "system_memory_cost", "system_substrate_cost", "system_pcb_cost",
            # Utilization metrics
            "avg_comp_util", "avg_mem_util",
            # System configuration
            "system_compute_TOPS", "system_bandwidth_TBps", "system_capacity_GB",
            "num_packages", "num_chiplets",
            # Memory/HBLC metrics
            "hblc_capacity_GB", "hblc_bandwidth_GBps", "hblc_bw_per_capacity",
            "peak_buffer_MB", "avg_cache_util_MB",
            # Model capacity metrics
            "model_weight_capacity_GB", "kv_cache_capacity_per_batch_GB", "total_model_capacity_GB",
            # Additional system parameters
            "cores_per_cu", "mem_buffer_size", "net_buffer_size", "tmacs_per_core",
            "hblc_bank_groups", "hblc_ranks", "hblc_frac_bank_cap"
        ]
        
        # Ensure header exists (should already be initialized, but check just in case)
        if not self.points_csv_initialized:
            self._initialize_points_csv()
        
        # Prepare row data
        row_data = [
            params["num_cus"],
            params["num_tmacs"],
            params["mem_buf_cap"],
            params["net_buf_cap"],
            params["mem_banks_per_group"],
            params["mem_ranks"],
            params["mem_frac_bank_cap"],
            params["batch_size"],
            params["kv_cache"],
        ]
        
        # Add metrics
        for col in metric_cols:
            row_data.append(metrics_dict.get(col, 0.0))
        
        # Append row
        try:
            with open(self.points_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            print(f"Saved point to CSV: {params['num_cus']} CUs, latency={metrics_dict.get('latency_ms', 0):.2f}ms, energy={metrics_dict.get('energy_mJ', 0):.2f}mJ")
        except Exception as e:
            print(f"Error saving to points.csv: {e}")
            raise

# sample code for running GA for Pistil
def runGAPistil(
    pop_size=50,
    n_gen=10,
    model_name="llama3-8b",
    allowed_num_cus=None,
    batch_bounds=(1, 64),
    kv_cache_bounds=(1024, 8192),
    initial_population=None,
    return_decisions=False,
    output_dir=None,
):
    """
    Run the Genetic Algorithm for the Pistil simulator.

    Returns either:
        - objectives array (n_points x 2) if return_decisions is False
        - dict with {"objectives": F, "decisions": X} otherwise
    """
    import time
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print(f"[PISTIL GA] ===== GA RUN INITIALIZATION =====")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Parameters:")
    print(f"    - Population Size: {pop_size}")
    print(f"    - Generations: {n_gen}")
    print(f"    - Model: {model_name}")
    print(f"    - Batch Bounds: {batch_bounds}")
    print(f"    - KV Cache Bounds: {kv_cache_bounds}")
    print("=" * 80 + "\n")
    
    # Create output directory for this run
    if output_dir is None:
        # Default under Pistil sim tree
        sim = PistilSimulator()
        base_results = os.path.join(sim.sim_root, "dse", "results")
        os.makedirs(base_results, exist_ok=True)
        output_dir = base_results
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"[PISTIL GA] Using provided output directory: {output_dir}")

    problem = PistilProblem(
        model_name=model_name,
        allowed_num_cus=allowed_num_cus,
        batch_bounds=batch_bounds,
        kv_cache_bounds=kv_cache_bounds,
        output_dir=output_dir,
    )

    # points.csv will be initialized with header on first evaluation by PistilProblem

    # Reset evaluation counter
    problem._eval_count = 0
    
    # Create callback to track generation progress
    class GenerationCallback(Callback):
        def __init__(self, problem_instance, total_gens, pop_size):
            super().__init__()
            self.problem = problem_instance
            self.total_gens = total_gens
            self.pop_size = pop_size
            
        def _count_csv_points(self):
            """Count points in CSV file (excluding header)"""
            if not os.path.exists(self.problem.points_csv_path):
                return 0
            try:
                with open(self.problem.points_csv_path, 'r') as f:
                    return sum(1 for line in f) - 1  # Subtract header
            except:
                return 0
            
        def update(self, algorithm):
            """Called after each generation"""
            current_gen = algorithm.n_gen
            # Get evaluation count from algorithm's evaluator (more accurate than problem._eval_count)
            try:
                if hasattr(algorithm, 'evaluator') and hasattr(algorithm.evaluator, 'n_eval'):
                    current_eval = algorithm.evaluator.n_eval
                else:
                    current_eval = getattr(self.problem, '_eval_count', 0)
            except:
                current_eval = getattr(self.problem, '_eval_count', 0)
            
            csv_points = self._count_csv_points()
            progress_pct = (current_gen / self.total_gens) * 100
            
            print(f"\n[PISTIL GA] {'='*60}")
            print(f"[PISTIL GA] Generation {current_gen}/{self.total_gens} completed ({progress_pct:.1f}%)")
            print(f"  Total evaluations attempted: {current_eval}/{self.pop_size * self.total_gens}")
            print(f"  Points successfully saved to CSV: {csv_points}")
            if current_eval > csv_points:
                failed_count = current_eval - csv_points
                print(f"  ⚠️  {failed_count} evaluation(s) failed (errors occurred)")
            print(f"[PISTIL GA] {'='*60}")

    print("=" * 80)
    print(f"[PISTIL GA] Starting Genetic Algorithm Optimization")
    print(f"  Model: {model_name}")
    print(f"  Population Size: {pop_size}")
    print(f"  Generations: {n_gen}")
    print(f"  Total Evaluations Expected: {pop_size * n_gen}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Points CSV: {problem.points_csv_path}")
    print("=" * 80)
    
    # Reset evaluation counter
    problem._eval_count = 0
    
    # Create callback to track generation progress
    class GenerationCallback(Callback):
        def __init__(self, problem_instance, total_gens, pop_size):
            super().__init__()
            self.problem = problem_instance
            self.total_gens = total_gens
            self.pop_size = pop_size
            
        def _count_csv_points(self):
            """Count points in CSV file (excluding header)"""
            if not os.path.exists(self.problem.points_csv_path):
                return 0
            try:
                with open(self.problem.points_csv_path, 'r') as f:
                    return sum(1 for line in f) - 1  # Subtract header
            except:
                return 0
            
        def update(self, algorithm):
            """Called after each generation"""
            current_gen = algorithm.n_gen
            # Get evaluation count from algorithm's evaluator (more accurate than problem._eval_count)
            try:
                if hasattr(algorithm, 'evaluator') and hasattr(algorithm.evaluator, 'n_eval'):
                    current_eval = algorithm.evaluator.n_eval
                else:
                    current_eval = getattr(self.problem, '_eval_count', 0)
            except:
                current_eval = getattr(self.problem, '_eval_count', 0)
            
            csv_points = self._count_csv_points()
            progress_pct = (current_gen / self.total_gens) * 100
            
            print(f"\n[PISTIL GA] {'='*60}")
            print(f"[PISTIL GA] Generation {current_gen}/{self.total_gens} completed ({progress_pct:.1f}%)")
            print(f"  Total evaluations attempted: {current_eval}/{self.pop_size * self.total_gens}")
            print(f"  Points successfully saved to CSV: {csv_points}")
            if current_eval > csv_points:
                failed_count = current_eval - csv_points
                print(f"  ⚠️  {failed_count} evaluation(s) failed (errors occurred)")
            print(f"[PISTIL GA] {'='*60}")
    
    # Sampling: either use provided initial_population or fall back to random
    if initial_population is not None:
        sampling = np.array(initial_population, dtype=float)
    else:
        sampling = IntegerRandomSampling()

    # Create callback instance
    callback = GenerationCallback(problem, n_gen, pop_size)
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=sampling,
        crossover=SBX(eta=15, prob=0.9, repair=RoundingRepair()),
        mutation=PM(eta=20, repair=RoundingRepair()),
        callback=callback,  # Pass callback to algorithm
    )
            
    res = pymoo_minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        verbose=True,
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print(f"[PISTIL GA] ===== OPTIMIZATION COMPLETE =====")
    print(f"  Total Evaluations: {getattr(problem, '_eval_count', 0)}")
    print(f"  Final Pareto Front Size: {len(res.F)}")
    print(f"  Total Runtime: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    print(f"  Average Time per Evaluation: {elapsed_time/max(getattr(problem, '_eval_count', 1), 1):.2f} seconds")
    print(f"  Points CSV: {problem.points_csv_path}")
    
    # Count actual points in CSV
    csv_point_count = 0
    if os.path.exists(problem.points_csv_path):
        with open(problem.points_csv_path, 'r') as f:
            csv_point_count = sum(1 for line in f) - 1  # Subtract header
    print(f"  Points in CSV file: {csv_point_count}")
    print("=" * 80 + "\n")

    if return_decisions:
        return {"objectives": res.F, "decisions": res.X}
    return res.F


