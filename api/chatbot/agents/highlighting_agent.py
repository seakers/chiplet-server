"""
Highlighting Agent for visually highlighting points on the plot via LLM commands.
"""
import numpy as np
from typing import Dict, Any, List, Optional
from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.data.loaders import PointsLoader
from api.config.objectives import OBJECTIVE_FIELD_MAP, to_field


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
            "(3) design variable range (e.g., num_cus = 16, GPU count >= 4 — "
            "use design_range mode with min_val == max_val for exact matches), "
            "(4) top-N by objective. Can also clear existing highlights. "
            "Use 'combined' for AND-ed conditions."
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
                # Pass active objectives through
                active_objectives = context.get('objectives')
                highlighted_indices = self._by_pareto_rank(
                    points, min_rank, max_rank, active_objectives
                )
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

            elif mode == 'combined':
                conditions = context.get('conditions', [])
                if not isinstance(conditions, list) or len(conditions) == 0:
                    return AgentResult(
                        success=False,
                        message="Combined mode requires a non-empty 'conditions' list.",
                        error="Missing conditions",
                    )

                index_sets = []
                descriptions = []
                active_objectives = context.get('objectives')

                # print(f"[HighlightingAgent] Processing combined conditions: {conditions}")

                for cond in conditions:
                    cmode = cond.get('mode')
                    if cmode == 'pareto_rank':
                        idxs = self._by_pareto_rank(
                            points,
                            cond.get('min_rank', 1),
                            cond.get('max_rank', 1),
                            active_objectives,
                        )
                        descriptions.append(
                            f"Pareto ranks {cond.get('min_rank', 1)}–{cond.get('max_rank', 1)}"
                        )
                    elif cmode == 'objective_range':
                        idxs = self._by_objective_range(
                            points,
                            cond.get('objective', 'energy'),
                            cond.get('min_val'),
                            cond.get('max_val'),
                        )
                        # print(f"[HighlightingAgent] Objective range condition: {cond}, matched indices: {idxs}, minimum: {cond.get('min_val')}, maximum: {cond.get('max_val')}")
                        descriptions.append(
                            f"{cond.get('objective')} in [{cond.get('min_val')}, {cond.get('max_val')}]"
                        )
                    elif cmode == 'design_range':
                        idxs = self._by_design_range(
                            points,
                            cond.get('variable', 'gpu'),
                            cond.get('min_val'),
                            cond.get('max_val'),
                        )
                        descriptions.append(
                            f"{cond.get('variable')} in [{cond.get('min_val')}, {cond.get('max_val')}]"
                        )
                    elif cmode == 'top_n':
                        idxs = self._top_n(
                            points,
                            cond.get('objective', 'energy'),
                            cond.get('n', 5),
                            cond.get('minimize', True),
                        )
                        descriptions.append(
                            f"top {cond.get('n', 5)} {'lowest' if cond.get('minimize', True) else 'highest'} {cond.get('objective')}"
                        )
                    else:
                        return AgentResult(
                            success=False,
                            message=f"Unknown sub-condition mode '{cmode}' in combined highlighting.",
                            error=f"Unknown sub-mode: {cmode}",
                        )
                    index_sets.append(set(idxs))

                # AND semantics: intersect all condition index sets
                if index_sets:
                    intersection = set.intersection(*index_sets)
                else:
                    intersection = set()
                highlighted_indices = sorted(intersection)
                description = " AND ".join(descriptions)

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
                print(f"[HighlightingAgent] Unknown mode: {mode}")
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
            print(f"[HighlightingAgent] {message}")

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

    def _by_pareto_rank(self, points, min_rank, max_rank, objectives=None):
        from api.analysis.pareto import ParetoCalculator
        from api.config.objectives import to_fields, DEFAULT_OBJECTIVES

        # Resolve which fields to use for Pareto ranking
        # print(f"[HighlightingAgent] Resolving fields for Pareto ranking with objectives: {objectives}")
        if objectives:
            # Map friendly names → internal keys, then fall back to x/y if not in point
            fields = to_fields(objectives)
        elif self.evaluator.lower() == 'pistil':
            fields = DEFAULT_OBJECTIVES.get('pistil', [])
            fields = to_fields(fields) if fields else ['latency_per_token_ms', 'energy_per_inference_mJ']
        else:
            fields = ['x', 'y']

        # Fallback: if a field isn't in the points, try x/y
        sample = points[0] if points else {}
        resolved = []
        for f in fields:
            if f in sample:
                resolved.append(f)
            elif 'x' in sample:
                resolved.append('x' if len(resolved) == 0 else 'y')

        if not resolved:
            resolved = ['x', 'y']

        try:
            obj_array = np.array([
                [float(pt.get(f, 0) or 0) for f in resolved]
                for pt in points
            ])

            if len(obj_array) == 0:
                return []

            # print(f"[HighlightingAgent] Calculating Pareto ranks for {len(obj_array)} points")
            # print(f"[HighlightingAgent] Full Obj_array: {obj_array}")
            ranks = ParetoCalculator.calculate_pareto_ranks(obj_array, max_ranks=max_rank)
            # print(f"[HighlightingAgent] Pareto ranks calculated: {ranks}")
            return [i for i, r in enumerate(ranks) if (min_rank - 1) <= r <= (max_rank - 1)]

        except Exception as e:
            print(f"[HighlightingAgent] ParetoCalculator failed: {e}, falling back to manual")
            return self._manual_pareto_rank(points, min_rank, max_rank, resolved[0],
                                            resolved[1] if len(resolved) > 1 else resolved[0], None)

    def _manual_pareto_rank(self, points, min_rank, max_rank, x_key, y_key, z_key):
        """Manual Pareto ranking fallback using x/y objectives."""
        # Choose correct objective keys based on evaluator
        objectives = []
        for i, pt in enumerate(points):
            x_val = pt.get(x_key) or pt.get('x')
            y_val = pt.get(y_key) or pt.get('y')
            z_val = pt.get(z_key) or pt.get('z')
            if x_val is not None and y_val is not None and z_val is not None:
                objectives.append((i, float(x_val), float(y_val), float(z_val)))
            elif x_val is not None and y_val is not None:
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
    
    def _resolve_objective_key(self, objective: str, sample_point: dict) -> Optional[str]:
        """
        Resolve a user-supplied objective name to the actual data field key on a point.

        Resolution order:
        1. OBJECTIVE_FIELD_MAP (e.g. 'Energy per Inference' -> 'energy_per_inference_mJ')
        2. Case-insensitive match against OBJECTIVE_FIELD_MAP friendly names
        3. Legacy CASCADE aliases (energy/runtime/time → y/x)
        4. The objective name itself if it already matches a field on the point
        5. Case-insensitive match against the point's own keys
        """
        if not objective:
            return None

        available_keys = set(sample_point.keys()) if sample_point else set()

        # 1. Direct lookup in the canonical map
        mapped = OBJECTIVE_FIELD_MAP.get(objective)
        if mapped and mapped in available_keys:
            return mapped

        # 2. Case-insensitive friendly-name match
        obj_lower = objective.lower()
        for friendly, field in OBJECTIVE_FIELD_MAP.items():
            if friendly.lower() == obj_lower and field in available_keys:
                return field

        # 3. Legacy aliases for CASCADE (kept so old prompts still work)
        legacy_aliases = {
            'energy': 'y',
            'runtime': 'x',
            'exe_time': 'x',
            'time': 'x',
            'latency': 'x',
        }
        legacy_key = legacy_aliases.get(obj_lower)
        if legacy_key and legacy_key in available_keys:
            return legacy_key

        # 4. Exact field name passed in directly (e.g. 'system_cost')
        if objective in available_keys:
            return objective

        # 5. Case-insensitive field-key match
        for k in available_keys:
            if isinstance(k, str) and k.lower() == obj_lower:
                return k

        return None

    def _by_objective_range(self, points, objective, min_val, max_val):
        """
        Highlight points whose value for the given objective falls within [min_val, max_val].
        Works with any objective declared in OBJECTIVE_FIELD_MAP, plus legacy aliases.
        """
        if not points:
            return []

        key = self._resolve_objective_key(objective, points[0])
        if key is None:
            print(f"[HighlightingAgent] Could not resolve objective '{objective}' "
                f"to a data field. Available point keys: {list(points[0].keys())}")
            return []

        indices = []
        for i, pt in enumerate(points):
            val = pt.get(key)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            if (min_val is None or val >= min_val) and (max_val is None or val <= max_val):
                indices.append(i)
        return indices

    def _by_design_range(self, points, variable, min_val, max_val):
        if not points:
            return []

        # Build a list of candidate keys to try
        candidates = [
            variable,
            variable.lower(),
            variable.replace(' ', '_').lower(),
        ]
        # CASCADE friendly-name aliases
        cascade_aliases = {
            'gpu': 'gpu', 'attention': 'attn', 'attn': 'attn',
            'sparse': 'sparse', 'convolution': 'conv', 'conv': 'conv',
        }
        if variable.lower() in cascade_aliases:
            candidates.append(cascade_aliases[variable.lower()])

        available = points[0].keys()
        key = next((c for c in candidates if c in available), None)
        if key is None:
            print(f"[HighlightingAgent] design_range: variable '{variable}' not in "
                f"point keys {list(available)}")
            return []

        indices = []
        for i, pt in enumerate(points):
            val = pt.get(key)
            if val is None:
                continue
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            lo = float(min_val) if min_val is not None else None
            hi = float(max_val) if max_val is not None else None
            # Small tolerance for float drift on equality queries
            eps = 1e-6
            if (lo is None or val >= lo - eps) and (hi is None or val <= hi + eps):
                indices.append(i)
        return indices

    def _top_n(self, points, objective, n, minimize=True):
        """
        Highlight the top-N points for a given objective. Works for any objective in
        OBJECTIVE_FIELD_MAP plus legacy aliases.
        """
        if not points:
            return []

        key = self._resolve_objective_key(objective, points[0])
        if key is None:
            print(f"[HighlightingAgent] Could not resolve objective '{objective}' "
                f"for top_n. Available point keys: {list(points[0].keys())}")
            return []

        scored = []
        for i, pt in enumerate(points):
            val = pt.get(key)
            if val is None:
                continue
            try:
                scored.append((i, float(val)))
            except (TypeError, ValueError):
                continue

        scored.sort(key=lambda x: x[1], reverse=not minimize)
        return [i for i, _ in scored[:n]]

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["pareto_rank", "objective_range", "design_range",
                            "top_n", "combined", "clear"],
                    "description": (
                        "Highlighting mode. Use 'combined' when the user gives multiple "
                        "conditions joined by AND (e.g. 'top 3 ranks AND cost below $1000'). "
                        "Use 'clear' to remove all highlights."
                    )
                },
                "conditions": {
                    "type": "array",
                    "description": (
                        "Only used when mode='combined'. Each entry is a sub-condition with "
                        "its own 'mode' (pareto_rank|objective_range|design_range|top_n) "
                        "and the matching parameters. All conditions are AND-ed together."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["pareto_rank", "objective_range", "design_range", "top_n"]
                            },
                            "min_rank": {"type": "integer"},
                            "max_rank": {"type": "integer"},
                            "objective": {
                                "type": "string",
                                "description": (
                                    "Objective name for objective_range/top_n modes. Accepts friendly names like "
                                    "'Energy', 'Runtime', 'Latency per Token', 'Energy per Inference', "
                                    "'Energy per Token', 'Average Power', 'System Power', 'System Cost', "
                                    "'Avg Compute Util', 'Avg Memory Util', 'Prefill Tokens/sec', "
                                    "'System Compute', 'System Bandwidth', 'System Capacity'. "
                                    "Also accepts raw data field names (e.g. 'system_cost', 'latency_per_token_ms')."
                                )
                            },
                            "variable": {
                                "type": "string",
                                "description": (
                                    "Design variable name for design_range mode. CASCADE: 'gpu', 'attn', "
                                    "'sparse', 'conv'. PISTIL: 'num_cus', 'num_tmacs', 'mem_buf_cap', "
                                    "'net_buf_cap', 'mem_banks_per_group', 'mem_ranks', 'mem_frac_bank_cap', "
                                    "'batch_size', 'kv_cache'. Use design_range (NOT objective_range) for "
                                    "these variables, even for exact-value queries (set min_val == max_val)."
                                )
                            },
                            "min_val": {"type": "number"},
                            "max_val": {"type": "number"},
                            "n": {"type": "integer"},
                            "minimize": {"type": "boolean"}
                        },
                        "required": ["mode"]
                    }
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
        keywords = [
            'highlight', 'show me', 'mark', 'which points', 'filter',
            'top designs', 'best designs', 'pareto rank',
            'and', 'with cost', 'below', 'above', 'less than', 'greater than'
        ]
        return any(kw in query.lower() for kw in keywords)