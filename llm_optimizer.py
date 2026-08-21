"""
llmOptimizer.py

LLM-guided evolutionary optimizer for CASCADE and PISTIL chiplet design.
Mirrors the return/CSV contract of gaCascade / gaPistil / rlPistil so it can
be dropped into compare_optimizers.py alongside the other optimizers.

Loop per generation:
  1. Ask the LLM for `pop_size` designs (seeded with best-so-far + analysis).
  2. Evaluate each via EvaluationAgent (snaps to GA discrete choices, handles infeasible).
  3. Run rule mining + distance correlation on all evaluated points.
  4. Feed those insights back to the LLM to propose the next population.
"""

import os
import csv
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple

from openai import OpenAI
import dotenv

from api.chatbot.agents import (
    AgentRegistry,
    EvaluationAgent,
    RuleMiningAgent,
    DistanceCorrelationAgent,
)
from api.data.loaders import PointsLoader

dotenv.load_dotenv()

FAILED_OBJ = [1e9, 1e9]   # same penalty convention as rsPistil / GA


class LLMOptimizer:
    """
    Evolutionary optimizer whose 'operators' are LLM completions.

    Parameters mirror the GA runners so compare_optimizers can call it
    with the same pop_size / n_gen knobs.
    """

    def __init__(
        self,
        evaluator: str = "cascade",
        model: str = "gpt-5.4-mini",
        run_id: str = None,
        output_dir: str = None,
        # ---- PISTIL-specific pass-throughs (ignored for CASCADE) ----
        model_name: str = "llama3-8b",
        allowed_num_cus: list = None,
        batch_bounds: tuple = (1, 64),
        kv_cache_bounds: tuple = (1024, 8192),
        # ---- CASCADE-specific ----
        trace: str = "gpt-j-65536-weighted",
        objectives: list = None,
        keep_elites: bool = True,
    ):
        self.evaluator = evaluator.lower()
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = model
        self.run_id = run_id
        self.output_dir = output_dir
        self.trace = trace
        self.model_name = model_name
        self.allowed_num_cus = allowed_num_cus or [16, 32, 64, 96, 128]
        self.batch_bounds = batch_bounds
        self.kv_cache_bounds = kv_cache_bounds
        self.keep_elites = keep_elites
        self.objectives = objectives or (
            ["Runtime", "Energy"] if self.evaluator == "cascade"
            else ["Latency per Token", "Energy per Inference"]
        )

        # Instantiate the agents we reuse (they read self.evaluator / self.run_id)
        self._eval_agent = self._make_agent(EvaluationAgent)
        self._rule_agent = self._make_agent(RuleMiningAgent)
        self._dcorr_agent = self._make_agent(DistanceCorrelationAgent)

        # Bookkeeping — match the (designs, objectives, design_points) contract
        self.all_designs: List[Dict[str, Any]] = []
        self.all_objectives: List[List[float]] = []

        # points.csv, written incrementally for hypervolume reconstruction
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.points_csv_path = os.path.join(self.output_dir, "points.csv")
        else:
            self.points_csv_path = None
        self._init_csv()

    # ------------------------------------------------------------------ #
    # Agent + CSV setup
    # ------------------------------------------------------------------ #
    def _make_agent(self, agent_cls):
        agent = agent_cls()
        # BaseAgent reads these attributes (see dcorr_agent / rule_mining_agent).
        agent.evaluator = self.evaluator
        agent.run_id = self.run_id
        return agent

    def _init_csv(self):
        if not self.points_csv_path:
            return
        # Truncate; header only needed for the RL-style CASCADE writer.
        with open(self.points_csv_path, "w", newline="") as f:
            if self.evaluator == "cascade":
                csv.writer(f).writerow(["exe_time_ms", "energy_mJ"])

    def _append_csv(self, obj: List[float]):
        if not self.points_csv_path:
            return
        # Append every evaluation (even failures/dupes) to preserve order,
        # exactly like gaCascade / rsPistil do.
        with open(self.points_csv_path, "a", newline="") as f:
            csv.writer(f).writerow([obj[0], obj[1]])

    # ------------------------------------------------------------------ #
    # Evaluation — delegate to EvaluationAgent (handles snapping + infeasible)
    # ------------------------------------------------------------------ #
    def _evaluate_design(self, design: Dict[str, Any]) -> List[float]:
        if self.evaluator == "cascade":
            ctx = {"chiplets": design, "trace": self.trace,
                "objectives": self.objectives}
        else:
            ctx = {"pistil_params": design, "model_name": self.model_name,
                "output_dir": self.output_dir, "objectives": self.objectives}

        result = self._eval_agent.execute(ctx)
        if not result.success:
            return FAILED_OBJ

        data = result.data or {}

        if self.evaluator == "cascade":
            # EvaluationAgent returns flat keys "exe_time" and "energy",
            # plus "objectives" as a NAME-keyed dict. raw_objectives order
            # is [energy, runtime], so read the flat keys explicitly and
            # emit [exe_time, energy] to match the CSV/plot convention. [10][1]
            exe_time = data.get("exe_time")
            energy = data.get("energy")
            if exe_time is None or energy is None:
                # Fallback: pull from the name-keyed objectives dict
                obj_dict = data.get("objectives", {})
                exe_time = obj_dict.get("Runtime")
                energy = obj_dict.get("Energy")
            if exe_time is None or energy is None:
                return FAILED_OBJ
            return [float(exe_time), float(energy)]

        # PISTIL path — verify these keys against your _execute_pistil return!
        lat = data.get("latency_ms")
        eng = data.get("energy_mJ")
        if lat is None or eng is None:
            obj_dict = data.get("objectives", {})
            lat = obj_dict.get("Latency per Token", obj_dict.get("latency_ms"))
            eng = obj_dict.get("Energy per Inference", obj_dict.get("energy_mJ"))
        if lat is None or eng is None:
            return FAILED_OBJ
        return [float(lat), float(eng)]

    def _record(self, design: Dict[str, Any], obj: List[float]):
        self.all_designs.append(design)
        self.all_objectives.append(obj)
        self._append_csv(obj)

    # ------------------------------------------------------------------ #
    # In-loop analysis — reuse the existing agents
    # ------------------------------------------------------------------ #
    def _gather_insights(self) -> str:
        """Run rule mining + distance correlation on all points so far."""
        insights = []
        loader = PointsLoader(self.evaluator, self.run_id)
        try:
            points = loader.load_points_as_dicts()
        except Exception:
            points = []

        # Need a minimum number of points for meaningful analysis
        # (dcorr needs >=5, rule mining >=3 per the agents' guards).
        if len(points) >= 5:
            rule_res = self._rule_agent.execute(
                {"objectives": self.objectives, "max_pareto_rank": 3}
            )
            if rule_res.success:
                insights.append("RULE MINING:\n" + rule_res.message)

            dcorr_res = self._dcorr_agent.execute(
                {"objectives": self.objectives, "use_all_points": True}
            )
            if dcorr_res.success:
                insights.append("DISTANCE CORRELATION:\n" + dcorr_res.message)

        return "\n\n".join(insights) if insights else "(no analysis yet — first generation)"

    # ------------------------------------------------------------------ #
    # LLM proposal
    # ------------------------------------------------------------------ #
    def _design_schema_text(self) -> str:
        if self.evaluator == "cascade":
            return (
                "Each design is a JSON object with integer keys GPU, Attention, "
                "Sparse, Convolution that MUST sum to exactly 12."
            )
        return (
            "Each design is a JSON object with keys: num_cus (one of "
            f"{self.allowed_num_cus}), num_tmacs, mem_buf_cap, net_buf_cap, "
            "mem_banks_per_group, mem_ranks, mem_frac_bank_cap, "
            "batch_size (power of 2 in 1..64), kv_cache (power of 2 in 1024..8192)."
        )

    def _best_designs_text(self, k: int = 10) -> str:
        if not self.all_objectives:
            return "(none yet)"
        objs = np.array(self.all_objectives)
        valid = objs[:, 0] < 1e9
        if not np.any(valid):
            return "(no feasible designs yet)"
        # Cheap Pareto-ish ranking: sort by summed normalized objectives.
        idx = np.where(valid)[0]
        scored = sorted(idx, key=lambda i: objs[i, 0] + objs[i, 1])[:k]
        lines = []
        for i in scored:
            lines.append(f"design={json.dumps(self.all_designs[i])} "
                         f"→ latency={objs[i,0]:.2f}, energy={objs[i,1]:.2f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # LLM proposal (continued)
    # ------------------------------------------------------------------ #
    def _build_prompt(self, pop_size: int, insights: str) -> str:
        """Assemble the generation prompt for the LLM."""
        return (
            f"You are optimizing {self.evaluator.upper()} chiplet designs to "
            f"MINIMIZE two objectives: {self.objectives[0]} and {self.objectives[1]}.\n\n"
            f"DESIGN SCHEMA:\n{self._design_schema_text()}\n\n"
            f"BEST FEASIBLE DESIGNS SO FAR (lower is better):\n"
            f"{self._best_designs_text()}\n\n"
            f"ANALYSIS FROM EVALUATED POINTS:\n{insights}\n\n"
            f"Propose EXACTLY {pop_size} NEW, DIVERSE designs that you expect to be "
            f"Pareto-optimal. Use the rule-mining and distance-correlation insights "
            f"to guide your choices. Do NOT repeat designs already listed above.\n\n"
            f"Return ONLY a JSON array of {pop_size} design objects — no prose, no "
            f"markdown fences."
        )

    def _propose_population(self, pop_size: int, insights: str) -> List[Dict[str, Any]]:
        """Ask the LLM for a new population and parse the JSON response."""
        # Same direct-completion pattern as ChatBot.summarize() — no tool calls [7].
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "developer",
                 "content": "You are a hardware design optimizer. "
                            "Respond with valid JSON only."},
                {"role": "user", "content": self._build_prompt(pop_size, insights)},
            ],
        )
        content = resp.choices[0].message.content.strip()

        # Strip accidental markdown fences before parsing.
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        try:
            designs = json.loads(content)
            if isinstance(designs, dict):          # single object → wrap it
                designs = [designs]
        except json.JSONDecodeError as e:
            print(f"[LLM OPT] Failed to parse LLM response: {e}\nRaw: {content[:500]}")
            designs = []

        # Pad short responses with elites so we always evaluate pop_size designs.
        designs = designs[:pop_size]
        while len(designs) < pop_size and self.all_designs:
            designs.append(self.all_designs[-1])
        return designs

    # ------------------------------------------------------------------ #
    # Main optimization loop
    # ------------------------------------------------------------------ #
    def run(self, pop_size: int = 10, n_gen: int = 5) -> np.ndarray:
        """
        Run the LLM-guided loop and return all evaluated objective points
        as an (N, 2) array in evaluation order — matching gaPistil / rlPistil
        / rsPistil so hypervolume curves line up in compare_optimizers.py [1][4].
        """
        print("\n" + "=" * 80)
        print(f"[LLM OPT] ===== LLM-GUIDED OPTIMIZATION =====")
        print(f"  Timestamp:  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Evaluator:  {self.evaluator}")
        print(f"  Model:      {self._model}")
        print(f"  pop_size:   {pop_size}")
        print(f"  n_gen:      {n_gen}")
        print(f"  output_dir: {self.output_dir}")
        print("=" * 80 + "\n")

        start = time.time()
        insights = "(no analysis yet — first generation)"

        for gen in range(n_gen):
            print(f"\n[LLM OPT] {'='*60}")
            print(f"[LLM OPT] Generation {gen + 1}/{n_gen}")
            print(f"[LLM OPT] {'='*60}")

            # 1. Ask the LLM for a new population.
            population = self._propose_population(pop_size, insights)
            print(f"[LLM OPT] LLM proposed {len(population)} designs.")

            # 2. Evaluate each design (EvaluationAgent snaps + flags infeasible [10]).
            for i, design in enumerate(population):
                print(f"[LLM OPT] Gen {gen + 1} — evaluating design {i + 1}/{len(population)}")
                obj = self._evaluate_design(design)
                if obj[0] >= 1e9:
                    print(f"  ✗ INFEASIBLE / failed — penalized as {FAILED_OBJ}")
                else:
                    print(f"  ✓ {self.objectives[0]}={obj[0]:.2f}, "
                          f"{self.objectives[1]}={obj[1]:.2f}")
                self._record(design, obj)

            # 3. Run rule mining + distance correlation on all points so far [8][9].
            insights = self._gather_insights()
            print(f"\n[LLM OPT] Insights refreshed for next generation.")

        elapsed = time.time() - start
        obj_arr = (np.array(self.all_objectives)
                   if self.all_objectives else np.empty((0, 2)))

        print("\n" + "=" * 80)
        print(f"[LLM OPT] ===== OPTIMIZATION COMPLETE =====")
        print(f"  Designs evaluated : {len(obj_arr)}")
        if len(obj_arr):
            valid = obj_arr[obj_arr[:, 0] < 1e9]
            print(f"  Successful        : {len(valid)}")
            print(f"  Failed            : {len(obj_arr) - len(valid)}")
            if len(valid):
                print(f"  Best {self.objectives[0]:>18}: {np.min(valid[:, 0]):.2f}")
                print(f"  Best {self.objectives[1]:>18}: {np.min(valid[:, 1]):.2f}")
        print(f"  Total runtime     : {elapsed:.1f}s  ({elapsed / 60:.2f} min)")
        print(f"  Points CSV        : {self.points_csv_path}")
        print("=" * 80 + "\n")

        return obj_arr


# --------------------------------------------------------------------------- #
# Top-level entry points — mirror runGAPistil / runRandomPistil signatures so
# compare_optimizers.py can drop these in [1][4][6].
# --------------------------------------------------------------------------- #
def runLLMPistil(
    pop_size: int = 10,
    n_gen: int = 5,
    model_name: str = "llama3-8b",
    llm_model: str = "gpt-5.4-mini",
    allowed_num_cus: list = None,
    batch_bounds: tuple = (1, 64),
    kv_cache_bounds: tuple = (1024, 8192),
    output_dir: str = None,
    run_id: str = None,
    objectives: list = None,
) -> np.ndarray:
    """
    Top-level entry point for LLM-guided search on the Pistil simulator.
    Returns objectives (N, 2) = [latency_ms, energy_mJ] in evaluation order.
    """
    opt = LLMOptimizer(
        evaluator="pistil",
        model=llm_model,
        run_id=run_id,
        output_dir=output_dir,
        model_name=model_name,
        allowed_num_cus=allowed_num_cus,
        batch_bounds=batch_bounds,
        kv_cache_bounds=kv_cache_bounds,
        objectives=objectives,
    )
    return opt.run(pop_size=pop_size, n_gen=n_gen)


def runLLMCascade(
    pop_size: int = 10,
    n_gen: int = 5,
    trace: str = "gpt-j-65536-weighted",
    llm_model: str = "gpt-5.4-mini",
    output_dir: str = None,
    run_id: str = None,
    objectives: list = None,
) -> np.ndarray:
    """
    Top-level entry point for LLM-guided search on the CASCADE model.
    Returns objectives (N, 2) = [exe_time_ms, energy_mJ] in evaluation order.
    """
    opt = LLMOptimizer(
        evaluator="cascade",
        model=llm_model,
        run_id=run_id,
        output_dir=output_dir,
        trace=trace,
        objectives=objectives,
    )
    return opt.run(pop_size=pop_size, n_gen=n_gen)