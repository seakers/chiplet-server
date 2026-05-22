"""
randomPistil.py

Random Search baseline for the Pistil chiplet design problem.
Randomly samples designs from the feasible space and evaluates them.
Follows the same structure and CSV format as gaPistil.py and rlPistil.py.
"""

import os
import csv
import time
import numpy as np
from pathlib import Path

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
    """Helper to snap a numeric value to the closest allowed choice."""
    choices = np.array(choices, dtype=float)
    idx = (np.abs(choices - float(value))).argmin()
    return choices[idx]


class PistilRandomSearcher:
    """
    Random Search evaluator for the Pistil simulator.

    Decision variables (identical to gaPistil.py / rlPistil.py):
        0  -> num_cus            (int, multiple of 4, from allowed_num_cus)
        1  -> num_tmacs          (int, from TMAC_CHOICES)
        2  -> mem_buf_cap        (float, from MEM_BUF_CHOICES)
        3  -> net_buf_cap        (float, from MEM_BUF_CHOICES)
        4  -> mem_banks_per_group(int, from BANK_GROUP_CHOICES)
        5  -> mem_ranks          (int, from RANK_CHOICES)
        6  -> mem_frac_bank_cap  (float, from FRAC_BANK_CHOICES)
        7  -> batch_size         (int, power of 2, from BATCH_CHOICES)
        8  -> kv_cache           (int, power of 2, from KV_CACHE_CHOICES)
    """

    # Shared choice sets (must stay in sync with gaPistil.py / rlPistil.py)
    TMAC_CHOICES        = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    MEM_BUF_CHOICES     = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    BANK_GROUP_CHOICES  = [1, 2, 3, 4]
    RANK_CHOICES        = [1, 2, 3, 4]
    FRAC_BANK_CHOICES   = [0.5, 0.75, 1.0]

    @staticmethod
    def _generate_power_of_2_choices(min_val, max_val):
        choices, power = [], 0
        while True:
            value = 2 ** power
            if value < min_val:
                power += 1
                continue
            if value > max_val:
                break
            choices.append(value)
            power += 1
        return choices if choices else [min_val]

    def __init__(
        self,
        model_name: str = "llama3-8b",
        allowed_num_cus: list = None,
        batch_bounds: tuple = (1, 64),
        kv_cache_bounds: tuple = (1024, 8192),
        output_dir: str = None,
        seed: int = None,
    ):
        if allowed_num_cus is None:
            allowed_num_cus = [16, 32, 64, 96, 128]

        self.model_name     = model_name
        self.allowed_num_cus = sorted(
            set(int(v) for v in allowed_num_cus if v % 4 == 0)
        )
        if not self.allowed_num_cus:
            raise ValueError(
                "allowed_num_cus must contain at least one multiple-of-4 value."
            )

        self.BATCH_CHOICES    = self._generate_power_of_2_choices(*batch_bounds)
        self.KV_CACHE_CHOICES = self._generate_power_of_2_choices(*kv_cache_bounds)

        # Ordered list of choice-sets — index must match decision variable order
        self.all_choices = [
            self.allowed_num_cus,    # 0: num_cus
            self.TMAC_CHOICES,       # 1: num_tmacs
            self.MEM_BUF_CHOICES,    # 2: mem_buf_cap
            self.MEM_BUF_CHOICES,    # 3: net_buf_cap
            self.BANK_GROUP_CHOICES, # 4: mem_banks_per_group
            self.RANK_CHOICES,       # 5: mem_ranks
            self.FRAC_BANK_CHOICES,  # 6: mem_frac_bank_cap
            self.BATCH_CHOICES,      # 7: batch_size
            self.KV_CACHE_CHOICES,   # 8: kv_cache
        ]

        self.sim = PistilSimulator()

        # CSV bookkeeping (mirrors gaPistil.py / rlPistil.py)
        self.output_dir = output_dir
        if self.output_dir is None:
            self.output_dir = os.path.join(self.sim.sim_root, "dse", "results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.points_csv_path        = os.path.join(self.output_dir, "points.csv")
        self.points_csv_initialized = False
        self._initialize_points_csv()

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # CSV helpers (identical column layout to gaPistil / rlPistil)
    # ------------------------------------------------------------------

    # Column definitions kept as class-level constants so they are
    # defined once and reused in both _initialize and _save methods.
    _DECISION_COLS = [
        "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
        "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
        "batch_size", "kv_cache",
    ]
    _METRIC_COLS = [
        "latency_ms", "energy_mJ",
        "latency_per_token_ms", "energy_per_inference_mJ", "energy_per_token_mJ",
        "prefill_tokens_per_sec",
        "average_power_W", "system_power_W",
        "average_power_mem_W", "average_power_comp_W", "average_power_net_W",
        "system_cost",
        "chiplet_silicon_cost", "memory_cost_2xHBLC", "package_cost",
        "package_silicon_cost", "package_memory_cost", "package_substrate_cost",
        "system_silicon_cost", "system_memory_cost", "system_substrate_cost",
        "system_pcb_cost",
        "avg_comp_util", "avg_mem_util",
        "system_compute_TOPS", "system_bandwidth_TBps", "system_capacity_GB",
        "num_packages", "num_chiplets",
        "hblc_capacity_GB", "hblc_bandwidth_GBps", "hblc_bw_per_capacity",
        "peak_buffer_MB", "avg_cache_util_MB",
        "model_weight_capacity_GB", "kv_cache_capacity_per_batch_GB",
        "total_model_capacity_GB",
        "cores_per_cu", "mem_buffer_size", "net_buffer_size", "tmacs_per_core",
        "hblc_bank_groups", "hblc_ranks", "hblc_frac_bank_cap",
    ]

    def _initialize_points_csv(self):
        if not os.path.exists(self.points_csv_path) or \
                os.path.getsize(self.points_csv_path) == 0:
            with open(self.points_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._DECISION_COLS + self._METRIC_COLS)
            self.points_csv_initialized = True
            print(f"[PISTIL RS] Initialized points.csv at {self.points_csv_path}")

    def _save_to_points_csv(self, params, metrics_dict):
        if not self.points_csv_initialized:
            self._initialize_points_csv()

        row = [
            params["num_cus"],
            params["num_tmacs"],
            params["mem_buf_cap"],
            params["net_buf_cap"],
            params["mem_banks_per_group"],
            params["mem_ranks"],
            params["mem_frac_bank_cap"],
            params["batch_size"],
            params["kv_cache"],
        ] + [metrics_dict.get(col, 0.0) for col in self._METRIC_COLS]

        try:
            with open(self.points_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print(
                f"[PISTIL RS] Saved: {params['num_cus']} CUs, "
                f"latency={metrics_dict.get('latency_ms', 0):.2f}ms, "
                f"energy={metrics_dict.get('energy_mJ', 0):.2f}mJ"
            )
        except Exception as exc:
            print(f"[PISTIL RS] Error saving to points.csv: {exc}")
            raise

    # ------------------------------------------------------------------
    # Design sampling
    # ------------------------------------------------------------------

    def sample_random_design(self) -> dict:
        """
        Uniformly sample one design from the full feasible space.
        Returns a parameter dict ready to pass to PistilSimulator.run_dse_point.
        """
        num_cus           = int(self.rng.choice(self.allowed_num_cus))
        num_tmacs         = int(self.rng.choice(self.TMAC_CHOICES))
        mem_buf_cap       = float(self.rng.choice(self.MEM_BUF_CHOICES))
        net_buf_cap       = float(self.rng.choice(self.MEM_BUF_CHOICES))
        mem_banks_pg      = int(self.rng.choice(self.BANK_GROUP_CHOICES))
        mem_ranks         = int(self.rng.choice(self.RANK_CHOICES))
        mem_frac_bank_cap = float(self.rng.choice(self.FRAC_BANK_CHOICES))
        batch_size        = int(self.rng.choice(self.BATCH_CHOICES))
        kv_cache          = int(self.rng.choice(self.KV_CACHE_CHOICES))

        return {
            "num_cus":             num_cus,
            "num_tmacs":           num_tmacs,
            "mem_buf_cap":         mem_buf_cap,
            "net_buf_cap":         net_buf_cap,
            "mem_banks_per_group": mem_banks_pg,
            "mem_ranks":           mem_ranks,
            "mem_frac_bank_cap":   mem_frac_bank_cap,
            "model":               self.model_name,
            "batch_size":          batch_size,
            "kv_cache":            kv_cache,
            # Fixed run-level constants (same as GA / RL)
            "w_dtype":             0.5,
            "kv_dtype":            1.0,
            "prefill":             "False",
            "prefill_chunk_size":  0,
            "prefill_cached":      0,
            "sim_num_layers":      -1,
            "lm_head":             "False",
            "plot_exe":            "False",
            "gen_trace":           True,
            "sim_standalone":      True,
            "base_config":         "pistil-sys-base.json",
        }

    # ------------------------------------------------------------------
    # Metric loading (identical logic to gaPistil.py / rlPistil.py)
    # ------------------------------------------------------------------

    def _load_all_metrics(self, params) -> dict:
        results_dir = os.path.join(self.sim.sim_root, self.sim.results_dir)
        if not os.path.exists(results_dir):
            raise FileNotFoundError(
                f"Pistil results directory not found: {results_dir}"
            )

        csv_files = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError(
                f"No Pistil result CSVs found in: {results_dir}"
            )

        latest_csv = max(csv_files, key=os.path.getmtime)

        metrics = {}
        with open(latest_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) != 2:
                    continue
                key, value = row
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

        total_latency_s  = float(metrics.get("total_latency_s", 0.0))
        total_energy_J   = float(metrics.get("total_energy_J", 0.0))
        num_layers       = float(metrics.get("num_layers", 1.0))
        sim_num_layers   = float(metrics.get("sim_num_layers", -1.0))
        batch_size       = float(metrics.get("batch_size", params["batch_size"]))

        if sim_num_layers != num_layers and sim_num_layers != -1:
            latency_ms = total_latency_s * num_layers / sim_num_layers * 1000.0
            energy_mJ  = total_energy_J  * num_layers / sim_num_layers * 1000.0
        else:
            latency_ms = total_latency_s * 1000.0
            energy_mJ  = total_energy_J  * 1000.0

        latency_per_token_ms    = latency_ms / batch_size if batch_size > 0 else 0.0
        energy_per_token_mJ     = energy_mJ  / batch_size if batch_size > 0 else 0.0
        num_cus                 = float(metrics.get("num_chiplets", params["num_cus"]))
        avg_power_W             = float(metrics.get("average_power_W", 0.0))

        prefill_batch      = metrics.get("prefill_batch", False)
        prefill_chunk_size = float(metrics.get("prefill_chunk_size", 0.0))
        if prefill_batch and prefill_chunk_size > 0 and latency_ms > 0:
            prefill_tps = (prefill_chunk_size * batch_size) / (latency_ms / 1000.0)
        else:
            prefill_tps = batch_size / (latency_ms / 1000.0) if latency_ms > 0 else 0.0

        all_metrics = dict(metrics)
        all_metrics.update({
            "latency_ms":                   float(latency_ms),
            "energy_mJ":                    float(energy_mJ),
            "latency_per_token_ms":         latency_per_token_ms,
            "energy_per_inference_mJ":      energy_mJ,
            "energy_per_token_mJ":          energy_per_token_mJ,
            "prefill_tokens_per_sec":       prefill_tps,
            "average_power_W":              avg_power_W,
            "system_power_W":               num_cus * avg_power_W,
            "average_power_mem_W":          float(metrics.get("average_power_mem_W",  0.0)),
            "average_power_comp_W":         float(metrics.get("average_power_comp_W", 0.0)),
            "average_power_net_W":          float(metrics.get("average_power_net_W",  0.0)),
            "system_cost":                  float(metrics.get("system_cost",           0.0)),
            "chiplet_silicon_cost":         float(metrics.get("chiplet_silicon_cost",  0.0)),
            "memory_cost_2xHBLC":           float(metrics.get("memory_cost_2xHBLC",   0.0)),
            "package_cost":                 float(metrics.get("package_cost",          0.0)),
            "package_silicon_cost":         float(metrics.get("package_silicon_cost",  0.0)),
            "package_memory_cost":          float(metrics.get("package_memory_cost",   0.0)),
            "package_substrate_cost":       float(metrics.get("package_substrate_cost",0.0)),
            "system_silicon_cost":          float(metrics.get("system_silicon_cost",   0.0)),
            "system_memory_cost":           float(metrics.get("system_memory_cost",    0.0)),
            "system_substrate_cost":        float(metrics.get("system_substrate_cost", 0.0)),
            "system_pcb_cost":              float(metrics.get("system_pcb_cost",       0.0)),
            "avg_comp_util":                float(metrics.get("avg_comp_util",         0.0)),
            "avg_mem_util":                 float(metrics.get("avg_mem_util",          0.0)),
            "num_chiplets":                 num_cus,
            "system_compute_TOPS":          float(metrics.get("system_compute_TOPS",   0.0)),
            "system_bandwidth_TBps":        float(metrics.get("system_bandwidth_TBps", 0.0)),
            "system_capacity_GB":           float(metrics.get("system_capacity_GB",    0.0)),
            "num_packages":                 float(metrics.get("num_packages",          0.0)),
            "hblc_capacity_GB":             float(metrics.get("hblc_capacity_GB",      0.0)),
            "hblc_bandwidth_GBps":          float(metrics.get("hblc_bandwidth_GBps",   0.0)),
            "hblc_bw_per_capacity":         float(metrics.get("hblc_bw_per_capacity",  0.0)),
            "peak_buffer_MB":               float(metrics.get("peak_buffer_MB",        0.0)),
            "avg_cache_util_MB":            float(metrics.get("avg_cache_util_MB",        0.0)),
            "model_weight_capacity_GB":     float(metrics.get("model_weight_capacity_GB",  0.0)),
            "kv_cache_capacity_per_batch_GB": float(metrics.get("kv_cache_capacity_per_batch_GB", 0.0)),
            "total_model_capacity_GB":      float(metrics.get("total_model_capacity_GB",   0.0)),
            "cores_per_cu":         float(metrics.get("cores_per_cu",         0.0)),
            "mem_buffer_size":      float(metrics.get("mem_buffer_size",      0.0)),
            "net_buffer_size":      float(metrics.get("net_buffer_size",      0.0)),
            "tmacs_per_core":       float(metrics.get("tmacs_per_core",       0.0)),
            "hblc_bank_groups":     float(metrics.get("hblc_bank_groups",     0.0)),
            "hblc_ranks":           float(metrics.get("hblc_ranks",           0.0)),
            "hblc_frac_bank_cap":   float(metrics.get("hblc_frac_bank_cap",   0.0)),
        })

        return all_metrics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        n_samples: int = 100,
        seed: int = None,
    ):
        """
        Sample and evaluate `n_samples` random designs.

        Parameters
        ----------
        n_samples : int
            Number of random designs to evaluate.
        seed : int, optional
            Override the RNG seed set at construction time.

        Returns
        -------
        objectives : np.ndarray, shape (N, 2)
            [latency_ms, energy_mJ] for each successfully evaluated design,
            in evaluation order.  Failed evaluations are recorded as (1e9, 1e9)
            and included so that the evaluation-order index stays meaningful for
            hypervolume curves.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        print("\n" + "=" * 80)
        print(f"[PISTIL RS] ===== RANDOM SEARCH INITIALIZATION =====")
        print(f"  Timestamp:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  n_samples:   {n_samples}")
        print(f"  model:       {self.model_name}")
        print(f"  output_dir:  {self.output_dir}")
        print("=" * 80 + "\n")

        objectives = []
        start = time.time()

        for i in range(n_samples):
            params = self.sample_random_design()

            print(f"[PISTIL RS] Sample {i + 1}/{n_samples}")
            print(
                f"  Design: {params['num_cus']} CUs, {params['num_tmacs']} TMACs, "
                f"batch={params['batch_size']}, kv_cache={params['kv_cache']}, "
                f"mem_buf={params['mem_buf_cap']}, net_buf={params['net_buf_cap']}"
            )

            try:
                # Same pre-flight fix used in gaPistil / rlPistil [2][3]
                if (
                    params.get("prefill") == "False"
                    and int(params.get("prefill_cached", 0)) == 0
                    and int(params.get("kv_cache", 0)) > 0
                ):
                    params = dict(params)
                    params["prefill_cached"] = 1
                    print(
                        "[PISTIL RS] Pre-flight: set prefill_cached=1 "
                        "to avoid seq_len=0 crash"
                    )

                t0 = time.time()
                self.sim.run_dse_point(params)
                print(f"  Simulation completed in {time.time() - t0:.1f}s")

                metrics_dict = self._load_all_metrics(params)
                latency_ms = metrics_dict["latency_ms"]
                energy_mJ  = metrics_dict["energy_mJ"]

                self._save_to_points_csv(params, metrics_dict)
                objectives.append([latency_ms, energy_mJ])

                print(
                    f"  ✓  latency={latency_ms:.2f}ms  "
                    f"energy={energy_mJ:.2f}mJ"
                )

            except Exception as exc:
                print(f"  ✗  ERROR – {exc}")
                import traceback
                traceback.print_exc()
                # Penalise failed designs identically to GA / RL so the CSV
                # row count stays in sync with n_samples.
                objectives.append([1e9, 1e9])

        elapsed = time.time() - start
        obj_arr = np.array(objectives)

        print("\n" + "=" * 80)
        print(f"[PISTIL RS] ===== RANDOM SEARCH COMPLETE =====")
        print(f"  Samples evaluated : {n_samples}")
        print(f"  Successful        : {int(np.sum(obj_arr[:, 0] < 1e9))}")
        print(f"  Failed            : {int(np.sum(obj_arr[:, 0] >= 1e9))}")
        if np.any(obj_arr[:, 0] < 1e9):
            valid = obj_arr[obj_arr[:, 0] < 1e9]
            print(f"  Best latency  (ms): {np.min(valid[:, 0]):.2f}")
            print(f"  Best energy  (mJ) : {np.min(valid[:, 1]):.2f}")
        print(f"  Total runtime     : {elapsed:.1f}s  ({elapsed / 60:.2f} min)")
        print(f"  Points CSV        : {self.points_csv_path}")
        print("=" * 80 + "\n")

        return obj_arr


# ---------------------------------------------------------------------------
# Convenience entry-point (mirrors runGAPistil / runPPOPistil)
# ---------------------------------------------------------------------------

def runRandomPistil(
    n_samples: int = 100,
    model_name: str = "llama3-8b",
    allowed_num_cus: list = None,
    batch_bounds: tuple = (1, 64),
    kv_cache_bounds: tuple = (1024, 8192),
    output_dir: str = None,
    seed: int = None,
) -> np.ndarray:
    """
    Top-level entry point for random search on the Pistil simulator.

    Mirrors the signatures of ``runGAPistil`` [2] and ``runPPOPistil`` [3]
    so it can be dropped into the same comparison harness.

    Returns
    -------
    objectives : np.ndarray, shape (N, 2)
        [latency_ms, energy_mJ] in evaluation order.
    """

    points_file = os.path.join(output_dir, "points.csv")
    with open(points_file, 'w') as f:
        pass  # Truncate / reset the file

    searcher = PistilRandomSearcher(
        model_name=model_name,
        allowed_num_cus=allowed_num_cus,
        batch_bounds=batch_bounds,
        kv_cache_bounds=kv_cache_bounds,
        output_dir=output_dir,
        seed=seed,
    )
    return searcher.run(n_samples=n_samples)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     objectives = runRandomPistil(
#         n_samples=5,
#         model_name="llama3-8b",
#         seed=42,
#     )
#     print("Objectives shape:", objectives.shape)
#     print(objectives)