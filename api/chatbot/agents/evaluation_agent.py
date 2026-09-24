"""
Evaluation Agent for evaluating chiplet designs using the CASCADE or PISTIL evaluator.
"""
import re
from typing import Dict, Any, Optional

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry


@AgentRegistry.register
class EvaluationAgent(BaseAgent):
    """
    Agent that evaluates a chiplet design configuration.

    Supports two evaluators (selected via self.evaluator):
      - "cascade": evaluates a mix of 12 chiplets (GPU/Attention/Sparse/Convolution).
      - "pistil":  evaluates a continuous hardware configuration (CUs, TMACs, memory, etc.)
                   for a named LLM model.
    """

    # ------------------------------------------------------------------
    # Cascade constants
    # ------------------------------------------------------------------
    TOTAL_CHIPLETS = 12
    CHIPLET_TYPES = ["GPU", "Attention", "Sparse", "Convolution"]

    # ------------------------------------------------------------------
    # Pistil discrete choices (mirrors PistilProblem in gaPistil.py)
    # ------------------------------------------------------------------
    PISTIL_TMAC_CHOICES        = [2, 4, 8, 12, 16]
    PISTIL_MEM_BUF_CHOICES     = [0.25, 0.5, 0.75, 1.0]
    PISTIL_BANK_GROUP_CHOICES  = [1, 2, 3, 4]
    PISTIL_RANK_CHOICES        = [1, 2, 3, 4]
    PISTIL_FRAC_BANK_CHOICES   = [0.5, 0.75, 1.0]

    # ------------------------------------------------------------------
    # BaseAgent interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "evaluation_agent"

    @property
    def description(self) -> str:
        return (
            f"Evaluates a chiplet design configuration and returns objective values. "
            f"The CURRENT evaluator is '{self.evaluator.upper()}'. "
            f"For CASCADE, supply 'chiplets' dict with GPU/Attention/Sparse/Convolution (must sum to 12) "
            f"and optionally a 'trace'. "
            f"For PISTIL, supply 'pistil_params' with num_cus, num_tmacs, mem_buf_cap, net_buf_cap, "
            f"mem_banks_per_group, mem_ranks, mem_frac_bank_cap, batch_size, kv_cache, and optionally 'model_name'. "
            f"You MAY propose a new design yourself — use rule mining or distance correlation insights to "
            f"choose chiplet ratios / Pistil parameters that are likely to be Pareto-optimal."
            f"IMPORTANT: If a design is returned as INFEASIBLE, you MUST modify the "
            f"design parameters before retrying — never submit the identical configuration again. "
            f"Use rule mining or distance correlation insights to guide the perturbation."
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Evaluate a chiplet design.

        For the CASCADE evaluator, context must contain either:
            - 'chiplets': dict with keys GPU, Attention, Sparse, Convolution
            - 'query':    raw string to parse chiplet counts from
          Optionally:
            - 'trace':    workload trace name (default: gpt-j-65536-weighted)

        For the PISTIL evaluator, context must contain either:
            - 'pistil_params': dict with Pistil hardware parameters (see get_parameters_schema)
            - 'query':         raw string to parse Pistil parameters from
          Optionally:
            - 'model_name':    LLM model name (default: llama3-8b)
            - 'output_dir':    directory for results/CSV output
        """
        try:
            evaluator = getattr(self, "evaluator", "cascade")

            if evaluator == "cascade":
                return self._execute_cascade(context)
            elif evaluator == "pistil":
                return self._execute_pistil(context)
            else:
                return AgentResult(
                    success=False,
                    message=f"Unsupported evaluator: '{evaluator}'. Choose 'cascade' or 'pistil'.",
                    error=f"Unsupported evaluator: '{evaluator}'"
                )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"This design is INFEASIBLE. Please try a slightly different design. Error message: {str(e)}",
                error=str(e)
            )

    # ==================================================================
    # CASCADE path
    # ==================================================================

    def _execute_cascade(self, context: Dict[str, Any]) -> AgentResult:
        
        meta = self._get_run_metadata()
        trace = (
            context.get("trace")
            or meta.get("trace_name")
            or "gpt-j-65536-weighted"
        )

        if not trace and self.run_id:
            try:
                from api.models import OptimizationRun
                run = OptimizationRun.objects.filter(run_id=self.run_id).first()
                if run and getattr(run, 'trace_name', None):
                    trace = run.trace_name
            except Exception as e:
                print(f"[EvaluationAgent] Could not look up run trace: {e}")
        if not trace:
            trace = "gpt-j-65536-weighted"

        chiplets = context.get("chiplets")
        if not chiplets:
            query = context.get("query", "")
            chiplets = self._parse_chiplets_from_query(query)

        if not chiplets:
            return AgentResult(
                success=False,
                message=(
                    "Could not determine chiplet configuration. "
                    "Please specify counts for GPU, Attention, Sparse, and Convolution "
                    f"chiplets (must total {self.TOTAL_CHIPLETS})."
                ),
                error="No chiplet configuration found"
            )

        validation_error = self._validate_chiplets(chiplets)
        if validation_error:
            return AgentResult(success=False, message=validation_error, error=validation_error)

        raw_objectives = self._run_cascade(chiplets, trace)

        # Use flexible objectives matching the other agents [7][9]
        from api.config.objectives import OBJECTIVE_FIELD_MAP
        requested_objectives = context.get('objectives') or ['Energy', 'Runtime']

        # CASCADE always returns [energy, runtime] — map by position then by name
        cascade_defaults = {
            'Energy':   raw_objectives[0],
            'Runtime':  raw_objectives[1],
        }
        objective_results = {}
        for obj_name in requested_objectives:
            if obj_name in cascade_defaults:
                objective_results[obj_name] = cascade_defaults[obj_name]
            else:
                field = OBJECTIVE_FIELD_MAP.get(obj_name)
                objective_results[obj_name] = cascade_defaults.get(field)

        evaluated_point = {
            "objectives": objective_results,       # dynamic
            "energy":     raw_objectives[0],       # kept for backward compat
            "exe_time":   raw_objectives[1],       # kept for backward compat
            "gpu":        chiplets["GPU"],
            "attn":       chiplets["Attention"],
            "sparse":     chiplets["Sparse"],
            "conv":       chiplets["Convolution"],
            "trace":      trace,
            "algorithm":  "Chatbot",
        }

        return AgentResult(
            success=True,
            message=self._format_cascade_result(evaluated_point),
            data=evaluated_point
        )

    # ==================================================================
    # PISTIL path
    # ==================================================================

    def _execute_pistil(self, context: Dict[str, Any]) -> AgentResult:

        meta = self._get_run_metadata()

        # Priority: explicit override > active run's trace_name > safe default
        model_name = (
            context.get("model_name")
            or context.get("trace_or_model")
            or meta.get("trace_name")
            or "llama3-8b"
        )
        if not meta.get('trace_name'):
            print(f"[EvaluationAgent] No active PISTIL run found; using default model '{model_name}'.")

        # Route the eval to the ACTIVE run's directory so the plot polling picks it up
        output_dir = meta.get('output_dir')
        if output_dir is None:
            run_id = context.get('run_id') or getattr(self, 'run_id', None)
            if run_id:
                from api.config.settings import get_results_dir
                output_dir = get_results_dir('pistil', run_id)  # .../dse/results/<run_id>
                import os
                os.makedirs(output_dir, exist_ok=True)
                print(f"[EvaluationAgent] Resolved output_dir from run_id: {output_dir}")

        pistil_params = context.get("pistil_params")
        if not pistil_params:
            query = context.get("query", "")
            pistil_params = self._parse_pistil_params_from_query(query)

        if not pistil_params:
            return AgentResult(
                success=False,
                message=(
                    "Could not determine Pistil hardware configuration. "
                    "Please specify num_cus, num_tmacs, mem_buf_cap, net_buf_cap, "
                    "mem_banks_per_group, mem_ranks, mem_frac_bank_cap, batch_size, and kv_cache."
                ),
                error="No Pistil configuration found"
            )

        # Inject model name into params (required by PistilSimulator)
        pistil_params.setdefault("model", model_name)

        validation_error = self._validate_pistil_params(pistil_params)
        if validation_error:
            print("C ", pistil_params)
            return AgentResult(success=False, message=validation_error, error=validation_error)

        # If output_dir was not provided by the LLM, build a deterministic default
        # that mirrors the path gaPistil uses so results land in the right place [5]
        if output_dir is None:
            import os
            from api.Evaluator.gaPistil import PistilSimulator
            sim = PistilSimulator()
            output_dir = os.path.join(sim.sim_root, "dse", "results", "agent_eval")
            os.makedirs(output_dir, exist_ok=True)
            print(f"[EvaluationAgent] output_dir was None, defaulting to: {output_dir}")

        from api.config.objectives import OBJECTIVE_FIELD_MAP
        objectives = context.get('objectives') or ['Latency per Token', 'Energy per Inference']
        pistil_params["model"] = model_name
        metrics = self._run_pistil(pistil_params, output_dir=output_dir, objectives=objectives)


        # Build objective results dynamically from whatever objectives were requested
        objective_results = {}
        for obj_name in objectives:
            field = OBJECTIVE_FIELD_MAP.get(obj_name, obj_name)
            if field in metrics:
                objective_results[obj_name] = metrics[field]
            else:
                print(f"[EvaluationAgent] Warning: field '{field}' not found in metrics for objective '{obj_name}'")
                objective_results[obj_name] = None

        evaluated_point = {
            "objectives":  objective_results,      # dynamic — whatever was requested
            "energy":      metrics.get("energy_mJ"),   # kept for backward compat
            "exe_time":    metrics.get("latency_ms"),   # kept for backward compat
            "model":       pistil_params["model"],
            "num_cus":     pistil_params["num_cus"],
            "num_tmacs":   pistil_params["num_tmacs"],
            "batch_size":  pistil_params["batch_size"],
            "kv_cache":    pistil_params["kv_cache"],
            "algorithm":   "Chatbot",
            # Preserve full metrics for callers that want them
            "metrics":   metrics,
        }

        return AgentResult(
            success=True,
            message=self._format_pistil_result(evaluated_point),
            data=evaluated_point
        )

    # ==================================================================
    # Parsing
    # ==================================================================

    def _parse_chiplets_from_query(self, query: str) -> Optional[Dict[str, int]]:
        """
        Extract chiplet counts from a natural language query.

        Handles patterns like:
          "evaluate 4 GPU, 3 Attention, 3 Sparse, 2 Convolution"
          "GPU=4 Attention=3 Sparse=3 Convolution=2"
          "4 gpu 3 attention 3 sparse 2 convolution"
        """
        query_lower = query.lower()
        patterns = {
            "GPU":         r'(\d+)\s*gpu',
            "Attention":   r'(\d+)\s*attention',
            "Sparse":      r'(\d+)\s*sparse',
            "Convolution": r'(\d+)\s*conv(?:olution)?',
        }

        chiplets = {}
        for chiplet_type, pattern in patterns.items():
            match = re.search(pattern, query_lower)
            chiplets[chiplet_type] = int(match.group(1)) if match else 0

        return None if all(v == 0 for v in chiplets.values()) else chiplets

    def _parse_pistil_params_from_query(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Extract Pistil hardware parameters from a natural language query.

        Handles patterns like:
          "64 CUs, 16 tmacs, mem_buf 1.0, net_buf 0.5, 2 bank groups, 2 ranks,
           frac_bank 0.75, batch 8, kv_cache 2048"
        """
        query_lower = query.lower()

        int_patterns = {
            "num_cus":            r'(\d+)\s*cu(?:s)?',
            "num_tmacs":          r'(\d+)\s*tmac(?:s)?',
            "mem_banks_per_group": r'(\d+)\s*bank\s*group(?:s)?',
            "mem_ranks":          r'(\d+)\s*rank(?:s)?',
            "batch_size":         r'batch(?:_size)?\s*[=:]?\s*(\d+)',
            "kv_cache":           r'kv[_\s]cache\s*[=:]?\s*(\d+)',
        }
        float_patterns = {
            "mem_buf_cap":       r'mem(?:ory)?[_\s]buf(?:fer)?[_\s]cap(?:acity)?\s*[=:]?\s*([0-9]*\.?[0-9]+)',
            "net_buf_cap":       r'net(?:work)?[_\s]buf(?:fer)?[_\s]cap(?:acity)?\s*[=:]?\s*([0-9]*\.?[0-9]+)',
            "mem_frac_bank_cap": r'frac(?:_bank(?:_cap)?)?\s*[=:]?\s*([0-9]*\.?[0-9]+)',
        }

        params: Dict[str, Any] = {}

        for key, pattern in int_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                params[key] = int(match.group(1))

        for key, pattern in float_patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                params[key] = float(match.group(1))

        return None if not params else params

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_chiplets(self, chiplets: Dict[str, int]) -> Optional[str]:
        """Return an error string if chiplets are invalid, else None."""
        missing = [t for t in self.CHIPLET_TYPES if t not in chiplets]
        if missing:
            return f"Missing chiplet types: {missing}"

        negative = [t for t, v in chiplets.items() if v < 0]
        if negative:
            return f"Chiplet counts cannot be negative: {negative}"

        total = sum(chiplets.values())
        if total != self.TOTAL_CHIPLETS:
            return (
                f"Total chiplets must equal {self.TOTAL_CHIPLETS}, got {total}. "
                f"Current config: {chiplets}"
            )

        return None

    def _validate_pistil_params(self, params: Dict[str, Any]) -> Optional[str]:
        """Return an error string if Pistil params are invalid, else None."""
        required_keys = [
            "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
            "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
            "batch_size", "kv_cache",
        ]
        missing = [k for k in required_keys if k not in params]
        if missing:
            return f"Missing Pistil parameters: {missing}"

        if params["num_cus"] % 4 != 0 or params["num_cus"] <= 0:
            return f"num_cus must be a positive multiple of 4, got {params['num_cus']}."

        if params["num_tmacs"] not in self.PISTIL_TMAC_CHOICES:
            return (
                f"num_tmacs must be one of {self.PISTIL_TMAC_CHOICES}, "
                f"got {params['num_tmacs']}."
            )

        for buf_key in ("mem_buf_cap", "net_buf_cap"):
            if params[buf_key] not in self.PISTIL_MEM_BUF_CHOICES:
                return (
                    f"{buf_key} must be one of {self.PISTIL_MEM_BUF_CHOICES}, "
                    f"got {params[buf_key]}."
                )

        if params["mem_banks_per_group"] not in self.PISTIL_BANK_GROUP_CHOICES:
            return (
                f"mem_banks_per_group must be one of {self.PISTIL_BANK_GROUP_CHOICES}, "
                f"got {params['mem_banks_per_group']}."
            )

        if params["mem_ranks"] not in self.PISTIL_RANK_CHOICES:
            return (
                f"mem_ranks must be one of {self.PISTIL_RANK_CHOICES}, "
                f"got {params['mem_ranks']}."
            )

        if params["mem_frac_bank_cap"] not in self.PISTIL_FRAC_BANK_CHOICES:
            return (
                f"mem_frac_bank_cap must be one of {self.PISTIL_FRAC_BANK_CHOICES}, "
                f"got {params['mem_frac_bank_cap']}."
            )

        return None

    # ==================================================================
    # Evaluator calls
    # ==================================================================

    def _run_cascade(self, chiplets: Dict[str, int], trace: str, output_dir: Optional[str] = None):
        from api.Evaluator.gaCascade import runSingleCascade
        return runSingleCascade(chiplets, trace, save_to_csv=True, source='Chatbot')

    def _run_pistil(self, params: Dict[str, Any], output_dir: Optional[str], objectives) -> Dict[str, Any]:
        """
        Run a single Pistil simulation and return the full metrics dictionary.

        Reuses PistilProblem's simulation and metrics-loading logic without
        running a full GA: instantiates the problem, calls sim.run_dse_point,
        and loads results via _load_all_metrics.
        """
        import os
        from api.Evaluator.gaPistil import PistilProblem, PistilSimulator

        # Second safety layer — should rarely trigger given the fix above [5]
        if output_dir is None:
            sim = PistilSimulator()
            output_dir = os.path.join(sim.sim_root, "dse", "results", "agent_eval")
            os.makedirs(output_dir, exist_ok=True)
            print(f"[EvaluationAgent._run_pistil] output_dir was None, defaulting to: {output_dir}")

        # Build a minimal PistilProblem just for its simulator and metrics loader.
        # num_cus from params is the only CU we need; supply it as the sole allowed value
        # so bounds are well-defined without affecting anything else.
        params = dict(params)  # don't mutate caller's dict
        
        if (str(params.get("prefill")) == "False"
                and int(params.get("prefill_cached", 0)) == 0
                and int(params.get("kv_cache", 0)) > 0):
            params["prefill_cached"] = 1
            print("[EvaluationAgent] Pre-flight: set prefill_cached=1 to avoid seq_len=0 crash")

        from api.Evaluator.gaPistil import _round_to_nearest
        # Snap to the same discrete choices the GA uses, so chatbot-proposed
        # values land on valid, tested design points.
        problem = PistilProblem(
            model_name=params["model"],
            allowed_num_cus=[params["num_cus"]],
            output_dir=output_dir,
            objectives=objectives
        )
        problem.algorithm_label = "Chatbot"
        if hasattr(problem, 'BATCH_CHOICES'):
            params["batch_size"] = int(_round_to_nearest(params["batch_size"], problem.BATCH_CHOICES))
            params["kv_cache"]   = int(_round_to_nearest(params["kv_cache"],   problem.KV_CACHE_CHOICES))

        # Guard: kv_cache drives random_tokens_init -> seq_len. A zero (or boundary)
        # value produces prefill_input_ids of shape (1, 0) and crashes write_kv [22].
        valid_kv = [k for k in problem.KV_CACHE_CHOICES if k > 0]
        if int(params.get("kv_cache", 0)) <= 0 and valid_kv:
            params["kv_cache"] = int(min(valid_kv))
            print(f"[EvaluationAgent] Pre-flight: forced kv_cache to {params['kv_cache']} (was <= 0)")

        print("Pistil Setup Params:", params['model'], params['num_cus'], output_dir)
        # Inject fixed/default constants that gaPistil always adds (mirrors _decode_vector)
        full_params = dict(params)
        full_params.setdefault("w_dtype", 0.5)
        full_params.setdefault("kv_dtype", 1.0)
        full_params.setdefault("prefill", "False")
        full_params.setdefault("prefill_chunk_size", 0)
        full_params.setdefault("prefill_cached", 0)
        full_params.setdefault("sim_num_layers", -1)
        full_params.setdefault("lm_head", "False")
        full_params.setdefault("plot_exe", "False")
        full_params.setdefault("gen_trace", True)
        full_params.setdefault("sim_standalone", True)
        full_params.setdefault("base_config", "pistil-sys-base.json")

        print("Running Pistil simulation with params:", full_params)

        try:
            problem.sim.run_dse_point(full_params)
        except RuntimeError as e:
            if "non-singleton dimension" in str(e) or "seq_len" in str(e):
                return AgentResult(
                    success=False,
                    message=(
                        "That Pistil configuration hit an invalid sequence-length "
                        "edge case in the simulator (often caused by a kv_cache value "
                        "at the boundary). Try a different kv_cache or batch_size."
                    ),
                    error=str(e)
                )
            raise

        metrics = problem._load_all_metrics(full_params)
        problem._save_to_points_csv(full_params, metrics)
        return metrics

    def _get_run_metadata(self) -> dict:
        """
        Look up the active OptimizationRun's metadata. Returns dict with:
            model (CASCADE/PISTIL), trace_name (or LLM model for PISTIL),
            output_dir (absolute path to the run's results dir), objectives.
        Returns empty dict if run_id is missing or not found.
        """
        if not self.run_id:
            return {}
        try:
            from api.models import OptimizationRun
            import sys
            from pathlib import Path

            run = OptimizationRun.objects.filter(run_id=self.run_id).first()
            if not run:
                return {}

            meta = {
                'model': run.model,
                'trace_name': run.trace_name,
                'objectives': run.objectives,
            }
            if run.model == 'PISTIL':
                meta['output_dir'] = str(
                    Path(sys.path[0]) / "api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results" / self.run_id
                )
            return meta
        except Exception as e:
            print(f"[EvaluationAgent] Could not look up run metadata: {e}")
            return {}

    # ==================================================================
    # Formatting
    # ==================================================================

    def _format_cascade_result(self, result: Dict[str, Any]) -> str:
        return (
            f"Design Evaluation Results (trace: {result['trace']}):\n\n"
            f"  Objectives:\n"
            f"    Energy:       {result['energy']:.4f} mJ\n"
            f"    Runtime:      {result['exe_time']:.4f} ms\n\n"
            f"  Chiplet Configuration (total: {self.TOTAL_CHIPLETS}):\n"
            f"    GPU:          {result['gpu']}\n"
            f"    Attention:    {result['attn']}\n"
            f"    Sparse:       {result['sparse']}\n"
            f"    Convolution:  {result['conv']}\n"
        )

    def _format_pistil_result(self, result: Dict[str, Any]) -> str:
        m = result.get("metrics", {})
        return (
            f"Design Evaluation Results (model: {result['model']}):\n\n"
            f"  Objectives:\n"
            f"    Energy:              {result['energy']:.4f} mJ\n"
            f"    Latency:             {result['exe_time']:.4f} ms\n\n"
            f"  Per-token Metrics:\n"
            f"    Latency/token:       {m.get('latency_per_token_ms', 0):.4f} ms\n"
            f"    Energy/token:        {m.get('energy_per_token_mJ', 0):.4f} mJ\n\n"
            f"  Hardware Configuration:\n"
            f"    Compute Units (CUs): {result['num_cus']}\n"
            f"    TMACs:               {result['num_tmacs']}\n"
            f"    Batch Size:          {result['batch_size']}\n"
            f"    KV Cache:            {result['kv_cache']}\n"
        )

    # ==================================================================
    # Schema & routing
    # ==================================================================

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                # --- Cascade parameters ---
                "chiplets": {
                    "type": "object",
                    "description": f"(Cascade) Chiplet counts — must sum to {self.TOTAL_CHIPLETS}",
                    "properties": {
                        "GPU":         {"type": "integer", "minimum": 0},
                        "Attention":   {"type": "integer", "minimum": 0},
                        "Sparse":      {"type": "integer", "minimum": 0},
                        "Convolution": {"type": "integer", "minimum": 0},
                    },
                    "required": ["GPU", "Attention", "Sparse", "Convolution"]
                },
                "trace": {
                    "type": "string",
                    "description": "(Cascade) Workload trace name",
                    "default": "gpt-j-65536-weighted"
                },
                # --- Pistil parameters ---
                "pistil_params": {
                    "type": "object",
                    "description": "(Pistil) Hardware configuration for the Pistil simulator",
                    "properties": {
                        "num_cus":             {"type": "integer",
                                               "description": "Number of Compute Units (CUs) — one of [16, 32, 64, 96, 128]"},
                        "num_tmacs":           {"type": "integer",
                                               "description": f"TMACs — one of {self.PISTIL_TMAC_CHOICES}"},
                        "mem_buf_cap":         {"type": "number",
                                               "description": f"Memory buffer capacity — one of {self.PISTIL_MEM_BUF_CHOICES}"},
                        "net_buf_cap":         {"type": "number",
                                               "description": f"Network buffer capacity — one of {self.PISTIL_MEM_BUF_CHOICES}"},
                        "mem_banks_per_group": {"type": "integer",
                                               "description": f"Memory banks per group — one of {self.PISTIL_BANK_GROUP_CHOICES}"},
                        "mem_ranks":           {"type": "integer",
                                               "description": f"Memory ranks — one of {self.PISTIL_RANK_CHOICES}"},
                        "mem_frac_bank_cap":   {"type": "number",
                                               "description": f"Fractional bank capacity — one of {self.PISTIL_FRAC_BANK_CHOICES}"},
                        "batch_size":          {"type": "integer",
                                               "description": "Batch size (power of 2) between 1 and 64"},
                        "kv_cache":            {"type": "integer",
                                               "description": "KV cache size (power of 2) between 1024 and 8192"},
                    },
                    "required": [
                        "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
                        "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
                        "batch_size", "kv_cache"
                    ]
                },
                "model_name": {
                    "type": "string",
                    "description": "(Pistil) LLM model name",
                    "default": "llama3-8b"
                },
                "output_dir": {
                    "type": "string",
                    "description": "(Pistil) Output directory for results/CSV (optional)"
                },
            },
            "required": []
        }

    def can_handle(self, query: str) -> bool:
        keywords = ['evaluate', 'evaluation', 'test design', 'run design',
                    'simulate', 'calculate objectives', 'what would']
        return any(kw in query.lower() for kw in keywords)