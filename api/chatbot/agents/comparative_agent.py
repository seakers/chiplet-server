"""
Comparative Analysis Agent — runs comparative analysis between two optimization runs.
"""
from typing import Dict, Any
from .base import BaseAgent, AgentResult
from .registry import AgentRegistry


@AgentRegistry.register
class ComparativeAnalysisAgent(BaseAgent):
    """Compares two existing optimization runs. Both must already exist."""

    @property
    def name(self) -> str:
        return "comparative_analysis_agent"

    @property
    def description(self) -> str:
        return (
            "Compares two existing optimization runs side-by-side. "
            "REQUIRED: run_a_id and run_b_id (both must be completed or loaded runs). "
            "If the user has not specified two runs, ask them which two to compare — "
            "do NOT guess. Use list_runs_agent (or ask the user) to find run IDs. "
            "Returns Pareto comparison and a per-run rule mining summary."
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        try:
            run_a_id = context.get('run_a_id')
            run_b_id = context.get('run_b_id')

            if not run_a_id or not run_b_id:
                return AgentResult(
                    success=False,
                    message=(
                        "I need two run IDs to run a comparative analysis. "
                        "Please tell me which two runs to compare."
                    ),
                    error="Missing run_a_id or run_b_id"
                )

            from api.models import OptimizationRun
            for rid in (run_a_id, run_b_id):
                if not OptimizationRun.objects.filter(run_id=rid).exists():
                    return AgentResult(
                        success=False,
                        message=f"Run '{rid}' does not exist.",
                        error=f"Unknown run_id: {rid}"
                    )

            # Load points for both runs and run a lightweight comparison.
            # Reuses the same per-run loader the rest of the system uses.
            from api.data.loaders import PointsLoader
            from api.analysis.pareto import ParetoCalculator
            import numpy as np

            loader_a = PointsLoader(self.evaluator, run_a_id)
            loader_b = PointsLoader(self.evaluator, run_b_id)
            points_a = loader_a.load_points_as_dicts()
            points_b = loader_b.load_points_as_dicts()

            if not points_a or not points_b:
                return AgentResult(
                    success=False,
                    message=f"One of the runs has no points (A={len(points_a)}, B={len(points_b)}).",
                    error="Empty run"
                )

            pareto_a = ParetoCalculator.get_pareto_front(points_a)
            pareto_b = ParetoCalculator.get_pareto_front(points_b)
            a_dominates_b = ParetoCalculator.count_dominated_points(pareto_a, pareto_b)
            b_dominates_a = ParetoCalculator.count_dominated_points(pareto_b, pareto_a)

            summary = (
                f"Comparative Analysis: {run_a_id} vs {run_b_id}\n\n"
                f"Run A: {len(points_a)} points, {len(pareto_a)} on Pareto front\n"
                f"Run B: {len(points_b)} points, {len(pareto_b)} on Pareto front\n\n"
                f"Run A's Pareto front dominates {a_dominates_b} of Run B's Pareto points.\n"
                f"Run B's Pareto front dominates {b_dominates_a} of Run A's Pareto points.\n"
            )
            if a_dominates_b > b_dominates_a:
                summary += "\n→ Run A is generally Pareto-superior."
            elif b_dominates_a > a_dominates_b:
                summary += "\n→ Run B is generally Pareto-superior."
            else:
                summary += "\n→ Neither run dominates the other clearly."

            return AgentResult(
                success=True,
                message=summary,
                data={
                    'run_a_id': run_a_id,
                    'run_b_id': run_b_id,
                    'run_a_points': len(points_a),
                    'run_b_points': len(points_b),
                    'run_a_pareto': len(pareto_a),
                    'run_b_pareto': len(pareto_b),
                    'a_dominates_b': a_dominates_b,
                    'b_dominates_a': b_dominates_a,
                    'run_a_data': points_a,
                    'run_b_data': points_b,
                }
            )

        except Exception as e:
            import traceback; traceback.print_exc()
            return AgentResult(
                success=False,
                message=f"Error during comparative analysis: {e}",
                error=str(e)
            )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "run_a_id": {"type": "string", "description": "ID of the first run to compare."},
                "run_b_id": {"type": "string", "description": "ID of the second run to compare."},
            },
            "required": ["run_a_id", "run_b_id"],
        }