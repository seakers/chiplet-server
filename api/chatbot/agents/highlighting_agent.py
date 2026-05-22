"""
Highlighting Agent for visually highlighting points on the plot via LLM commands.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.data.loaders import PointsLoader


@AgentRegistry.register
class HighlightingAgent(BaseAgent):
    """
    Agent that highlights design points on the scatter plot based on
    Pareto rank ranges, objective value ranges, or design variable ranges.
    """

    @property
    def name(self) -> str:
        return "highlighting_agent"

    @property
    def description(self) -> str:
        return (
            "Highlights design points on the scatter plot. Can highlight by: "
            "(1) Pareto rank range (e.g., top 1-3 ranks), "
            "(2) objective value range (e.g., energy between 5 and 10 mJ), "
            "(3) design variable range (e.g., GPU count >= 4). "
            "Use this whenever the user asks to see, show, or highlight specific designs."
            "Can also clear existing highlights."
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        try:
            loader = PointsLoader(self.evaluator, self.run_id)
            points = context.get('points') or loader.load_points_as_dicts()

            if not points:
                return AgentResult(
                    success=False,
                    message="No data available for highlighting.",
                    error="No points data found"
                )

            mode = context.get('mode', 'pareto_rank')
            highlighted_indices = []

            if mode == 'pareto_rank':
                min_rank = context.get('min_rank', 1)
                max_rank = context.get('max_rank', 1)
                highlighted_indices = self._by_pareto_rank(points, min_rank, max_rank)
                description = f"Pareto ranks {min_rank}–{max_rank}"

            elif mode == 'objective_range':
                objective = context.get('objective', 'energy')
                min_val = context.get('min_val')
                max_val = context.get('max_val')
                highlighted_indices = self._by_objective_range(points, objective, min_val, max_val)
                description = f"{objective} in [{min_val}, {max_val}]"

            elif mode == 'design_range':
                variable = context.get('variable', 'gpu')
                min_val = context.get('min_val')
                max_val = context.get('max_val')
                highlighted_indices = self._by_design_range(points, variable, min_val, max_val)
                description = f"{variable} in [{min_val}, {max_val}]"

            elif mode == 'top_n':
                objective = context.get('objective', 'energy')
                n = context.get('n', 5)
                minimize = context.get('minimize', True)
                highlighted_indices = self._top_n(points, objective, n, minimize)
                description = f"Top {n} {'lowest' if minimize else 'highest'} {objective}"

            elif mode == 'clear':
                # Return empty highlighted_points to clear all highlighting
                highlighted_points = [
                    {**pt, 'highlighted': False, 'index': i}
                    for i, pt in enumerate(points)
                ]
                return AgentResult(
                    success=True,
                    message="Cleared all highlighting.",
                    data={
                        'highlighted_points': highlighted_points,
                        'highlighted_indices': [],
                        'count': 0,
                        'total': len(points),
                        'description': 'cleared',
                        'mode': 'clear',
                    }
                )

            else:
                return AgentResult(
                    success=False,
                    message=f"Unknown highlighting mode: {mode}",
                    error=f"Unknown mode: {mode}"
                )

            # Build highlighted_points array for the frontend
            highlighted_points = []
            for i, pt in enumerate(points):
                entry = {**pt, 'highlighted': i in highlighted_indices, 'index': i}
                # Ensure model tag is present for frontend findGlobalIndex branching
                if self.evaluator.lower() == 'pistil' and 'model' not in entry:
                    entry['model'] = 'PISTIL'
                highlighted_points.append(entry)

            message = (
                f"Highlighted {len(highlighted_indices)} out of {len(points)} designs "
                f"matching: {description}."
            )

            return AgentResult(
                success=True,
                message=message,
                data={
                    'highlighted_points': highlighted_points,
                    'highlighted_indices': highlighted_indices,
                    'count': len(highlighted_indices),
                    'total': len(points),
                    'description': description,
                    'mode': mode,
                }
            )

        except Exception as e:
            import traceback
            print(f"[HighlightingAgent] Error during highlighting: {e}")
            traceback.print_exc()  # ← This will show you the real error
            return AgentResult(
                success=False,
                message=f"Error during highlighting: {str(e)}",
                error=str(e)
            )

    def _by_pareto_rank(self, points, min_rank, max_rank):
        try:
            from api.analysis.pareto import ParetoCalculator
            calculator = ParetoCalculator()
            ranks = calculator.calculate_pareto_ranks(points, self.evaluator)
            return [i for i, r in enumerate(ranks) if min_rank <= r <= max_rank]
        except Exception as e:
            print(f"[HighlightingAgent] ParetoCalculator failed: {e}, falling back to manual calculation")
            # Manual Pareto rank calculation as fallback
            return self._manual_pareto_rank(points, min_rank, max_rank)

    def _manual_pareto_rank(self, points, min_rank, max_rank):
        """Manual Pareto ranking fallback using x/y objectives."""
        # Choose correct objective keys based on evaluator
        if self.evaluator.lower() == 'pistil':
            x_key, y_key = 'latency_per_token_ms', 'energy_per_inference_mJ'
        else:
            x_key, y_key = 'x', 'y'

        # Extract objective values
        objectives = []
        for i, pt in enumerate(points):
            x_val = pt.get(x_key) or pt.get('x')
            y_val = pt.get(y_key) or pt.get('y')
            if x_val is not None and y_val is not None:
                objectives.append((i, float(x_val), float(y_val)))

        # Assign Pareto ranks iteratively
        ranks = {}
        remaining = list(objectives)
        rank = 1

        while remaining:
            # Find non-dominated points in current remaining set
            pareto_front = []
            for i, x, y in remaining:
                dominated = False
                for j, xj, yj in remaining:
                    if i == j:
                        continue
                    if xj <= x and yj <= y and (xj < x or yj < y):
                        dominated = True
                        break
                if not dominated:
                    pareto_front.append((i, x, y))

            for i, x, y in pareto_front:
                ranks[i] = rank
            remaining = [(i, x, y) for i, x, y in remaining
                        if i not in {idx for idx, _, _ in pareto_front}]
            rank += 1

        return [i for i, r in ranks.items() if min_rank <= r <= max_rank]

    def _by_objective_range(self, points, objective, min_val, max_val):
        key_map = {
            'energy': 'y', 'runtime': 'x', 'exe_time': 'x', 'time': 'x',
            'latency_per_token': 'latency_per_token_ms',
            'energy_per_inference': 'energy_per_inference_mJ',
        }
        key = key_map.get(objective, objective)
        indices = []
        for i, pt in enumerate(points):
            val = pt.get(key)
            if val is None:
                continue
            if (min_val is None or val >= min_val) and (max_val is None or val <= max_val):
                indices.append(i)
        return indices

    def _by_design_range(self, points, variable, min_val, max_val):
        variable = variable.lower()
        indices = []
        for i, pt in enumerate(points):
            val = pt.get(variable)
            if val is None:
                continue
            if (min_val is None or val >= min_val) and (max_val is None or val <= max_val):
                indices.append(i)
        return indices

    def _top_n(self, points, objective, n, minimize=True):
        key_map = {
            'energy': 'y', 'runtime': 'x', 'exe_time': 'x', 'time': 'x',
        }
        key = key_map.get(objective, objective)
        scored = [(i, pt.get(key, float('inf') if minimize else float('-inf')))
                  for i, pt in enumerate(points) if pt.get(key) is not None]
        scored.sort(key=lambda x: x[1], reverse=not minimize)
        return [i for i, _ in scored[:n]]

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["pareto_rank", "objective_range", "design_range", "top_n", "clear"],
                    "description": "Highlighting mode. Use 'clear' to remove all highlights."
                },
                "min_rank": {"type": "integer", "description": "Min Pareto rank (for pareto_rank mode)"},
                "max_rank": {"type": "integer", "description": "Max Pareto rank (for pareto_rank mode)"},
                "objective": {"type": "string", "description": "Objective name (for objective_range/top_n)"},
                "variable": {"type": "string", "description": "Design variable name (for design_range)"},
                "min_val": {"type": "number", "description": "Min value (for range modes)"},
                "max_val": {"type": "number", "description": "Max value (for range modes)"},
                "n": {"type": "integer", "description": "Number of points (for top_n mode)"},
                "minimize": {"type": "boolean", "description": "Whether lower is better (for top_n)"}
            },
            "required": ["mode"]
        }

    def can_handle(self, query: str) -> bool:
        keywords = ['highlight', 'show me', 'mark', 'which points', 'filter',
                     'top designs', 'best designs', 'pareto rank']
        return any(kw in query.lower() for kw in keywords)