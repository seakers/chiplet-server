import os
import sys
import subprocess
import shutil
from pathlib import Path

class PistilSimulator:
    def __init__(self):
        # 1. Automatically determine the absolute path to THIS file's directory
        #    This resolves to: /path/to/your/project/api/Evaluator/sim-v2-4-pistil-sim-clean
        self.sim_root = Path(__file__).parent.resolve()

        # 2. Define standard paths RELATIVE to the sim_root
        #    We keep these relative because we will execute the subprocesses 
        #    with cwd=self.sim_root, mimicking the behavior of running "./run.sh" inside the folder.
        self.base_config_dir = "./configs/sys_configs"
        self.output_config_dir = "./configs/gen_configs"
        self.model_config_dir = "./configs/model_configs"
        self.trace_dir = "./traces"
        self.results_dir = "./trace-results"
        
        # Ensure output directories exist inside the sim folder
        (self.sim_root / self.output_config_dir).mkdir(parents=True, exist_ok=True)
        (self.sim_root / self.results_dir).mkdir(parents=True, exist_ok=True)

    def run_dse_point(self, params):
        """
        params: dict containing hardware and app knobs
        """
        # Construct the Configuration Name
        config_name = (
            f"pistil-config-num_cus-{params['num_cus']}-"
            f"tmacs-{params['num_tmacs']}-"
            f"mem_buf_cap-{params['mem_buf_cap']}-"
            f"net_buf_cap-{params['net_buf_cap']}-"
            f"mem_bank_groups-{params['mem_banks_per_group']}-"
            f"mem_ranks-{params['mem_ranks']}-"
            f"mem_frac_bank_cap-{params['mem_frac_bank_cap']}.json"
        )

        trace_name = f"{params['model']}-chiplets-{params['num_cus']}-bs-{params['batch_size']}-sl-{params['kv_cache']}"

        print(f"--- [Pistil Runner] Processing: {config_name} ---")

        # --- STEP 1: Config Generation ---
        cmd_gen_config = [
            sys.executable, "main-gen-config.py",
            f"--name={config_name}",
            f"--output-config-dir={self.output_config_dir}",
            f"--base-sys-cfg={self.base_config_dir}/{params.get('base_config', 'pistil-sys-base.json')}",
            f"--tmacs={params['num_tmacs']}",
            f"--mem-buf-cap={params['mem_buf_cap']}",
            f"--net-buf-cap={params['net_buf_cap']}",
            f"--mem-bank-groups={params['mem_banks_per_group']}",
            f"--mem-ranks={params['mem_ranks']}",
            f"--mem-frac-bank-cap={params['mem_frac_bank_cap']}",
            f"--num-cus={params['num_cus']}"
        ]
        
        self._run_command(cmd_gen_config, "Config Gen")

        # --- STEP 2: Trace Generation ---
        if params.get('gen_trace', True):
            cmd_gen_trace = [
                sys.executable, "main-gen-trace.py",
                f"--model-name={params['model']}",
                f"--model-cfg={self.model_config_dir}/{params['model']}.json",
                f"--sys-cfg={self.output_config_dir}/{config_name}",
                f"--trace-file={self.trace_dir}/{trace_name}",
                f"--w-dtype={params['w_dtype']}",
                f"--kv-dtype={params['kv_dtype']}",
                f"--sim-batch-size={params['batch_size']}",
                f"--sim-prefill-batch={params['prefill']}",
                f"--sim-prefill-chunk-size={params['prefill_chunk_size']}",
                f"--sim-prefill-cached={params['prefill_cached']}",
                f"--sim-kv-cache={params['kv_cache']}",
                f"--sim-vocab-size=128000",
                f"--sim-num-chiplets={params['num_cus']}",
                "--verbose=1",
                "--inference=True"
            ]
            self._run_command(cmd_gen_trace, "Trace Gen")

        # --- STEP 3: Simulation ---
        if params.get('sim_standalone', True):
            cmd_sim = []
            # Only use numactl if it actually exists in the system path
            if shutil.which("numactl"):
                 cmd_sim = ["numactl", "-m", "0"]
            
            cmd_sim += [
                sys.executable, "main-sim-trace.py",
                f"--model-name={params['model']}",
                f"--model-cfg={self.model_config_dir}/{params['model']}.json",
                f"--sys-cfg={self.output_config_dir}/{config_name}",
                f"--trace-file={self.trace_dir}/{trace_name}",
                f"--results-dir={self.results_dir}",
                f"--sim-batch-size={params['batch_size']}",
                f"--sim-prefill-batch={params['prefill']}",
                f"--sim-prefill-chunk-size={params['prefill_chunk_size']}",
                f"--sim-prefill-cached={params['prefill_cached']}",
                f"--sim-kv-cache={params['kv_cache']}",
                f"--sim-num-layers={params.get('sim_num_layers', -1)}",
                f"--sim-lm-head={params.get('lm_head', False)}",
                f"--w-dtype={params['w_dtype']}",
                f"--kv-dtype={params['kv_dtype']}",
                f"--sim-num-chiplets={params['num_cus']}",
                f"--plot-exe={params.get('plot_exe', False)}",
                "--verbose=1",
                "--inference=False"
            ]
            self._run_command(cmd_sim, "Simulation")

    def _run_command(self, cmd_list, stage_name):
        try:
            # IMPORTANT: cwd=self.sim_root ensures the script runs AS IF we were 
            # inside the api/Evaluator/... folder. 
            # This fixes all relative path imports inside the target scripts.
            subprocess.run(cmd_list, check=True, cwd=self.sim_root)
        except subprocess.CalledProcessError as e:
            print(f"!!! Error during {stage_name} !!!")
            # We re-raise so the DSE tool knows this design point failed
            raise e
        except KeyboardInterrupt:
            print("\nRun interrupted by user.")
            sys.exit(1)

# --- Verification Block (Run this file directly to test) ---
if __name__ == "__main__":
    # Example params used for testing the logic
    dse_params = {
        "num_tmacs": 8,
        "mem_buf_cap": 0.25,
        "net_buf_cap": 1.0,
        "mem_banks_per_group": 2,
        "mem_ranks": 4,
        "mem_frac_bank_cap": 1.0,
        "num_cus": 64,
        "model": "llama3-8b", 
        "batch_size": 16,
        "kv_cache": 8192, 
        "w_dtype": 0.5,
        "kv_dtype": 1.0,
        "prefill": "False",
        "prefill_chunk_size": 256,
        "prefill_cached": 1024,
        "sim_num_layers": 1, # Run just 1 layer for speed test
        "lm_head": "False",
        "plot_exe": "True",
        "gen_trace": True,
        "sim_standalone": True
    }

    print(f"Running Pistil Runner from: {Path(__file__).resolve()}")
    runner = PistilSimulator()
    runner.run_dse_point(dse_params)
    print("Test Complete.")