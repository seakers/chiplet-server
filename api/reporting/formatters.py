"""
Formatting utilities for reports and data display.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from api.analysis.rule_mining import RuleFormatter
from api.config.evaluators import EvaluatorConfig, get_evaluator_config
from api.config.objectives import to_field, to_fields, to_axis_label, DEFAULT_OBJECTIVES


# CASCADE-specific flattened keys emitted by DesignPoint.to_dict()
_CASCADE_FLAT_KEYS = {
    'GPU': 'gpu',
    'Attention': 'attn',
    'Sparse': 'sparse',
    'Convolution': 'conv',
}


def get_point_objective_value(point: Dict[str, Any],
                              friendly_name: str,
                              obj_index: int) -> float:
    """
    Resolve an objective value from a point dict, using several fallbacks.

    Order:
      1. point[<internal field key>]      e.g. 'latency_per_token_ms'
      2. point['x'] / point['y']          for objective indices 0 and 1
      3. 0.0
    """
    field = to_field(friendly_name)
    if field in point and point[field] is not None:
        try:
            return float(point[field])
        except (TypeError, ValueError):
            pass

    # Fallback for the first two objectives (DesignPoint.to_dict always emits x/y)
    if obj_index == 0 and 'x' in point:
        try:
            return float(point['x'])
        except (TypeError, ValueError):
            pass
    if obj_index == 1 and 'y' in point:
        try:
            return float(point['y'])
        except (TypeError, ValueError):
            pass

    return 0.0


def get_point_decision_value(point: Dict[str, Any], dec_name: str) -> Any:
    """
    Resolve a decision-variable value from a point dict.

    Order:
      1. point['chiplets'][dec_name]     (always emitted by DesignPoint.to_dict)
      2. point[<cascade flattened key>]  e.g. 'gpu' for 'GPU'
      3. point[dec_name]                 direct hit (PISTIL keys)
      4. 0
    """
    chiplets = point.get('chiplets')
    if isinstance(chiplets, dict) and dec_name in chiplets:
        return chiplets[dec_name]

    flat = _CASCADE_FLAT_KEYS.get(dec_name)
    if flat and flat in point:
        return point[flat]

    if dec_name in point:
        return point[dec_name]

    return 0


def resolve_objectives(run_params: Dict[str, Any],
                       config: EvaluatorConfig) -> List[str]:
    """Pick objectives from run_params, falling back to evaluator defaults."""
    objs = run_params.get('objectives')
    if objs:
        # Guard against comma-joined strings
        if isinstance(objs, str):
            objs = [o.strip() for o in objs.split(',') if o.strip()]
        return list(objs)
    return list(DEFAULT_OBJECTIVES.get(config.name, ['Runtime', 'Energy']))


class ReportFormatter:
    """
    Formats data for report generation.
    """

    @staticmethod
    def format_run_params(run_params: Dict[str, Any]) -> Dict[str, str]:
        """Format run parameters for display."""
        objs = run_params.get('objectives', [])
        if isinstance(objs, str):
            objs_str = objs
        else:
            objs_str = ', '.join(objs)
        return {
            'model': str(run_params.get('model', 'Unknown')),
            'algorithm': str(run_params.get('algorithm', 'Unknown')),
            'objectives': objs_str,
            'population_size': str(run_params.get('population_size', 'N/A')),
            'generations': str(run_params.get('generations', 'N/A')),
            'trace_name': str(run_params.get('trace_name', 'Unknown')),
        }

    @staticmethod
    def format_statistics(points: List[Dict[str, Any]],
                          objectives: List[str]) -> Dict[str, Any]:
        """
        Calculate and format statistics from points, one entry per selected
        objective.

        Returns:
            {
              'total': <int>,
              'per_objective': {
                 <friendly_name>: {
                    'avg': <float>, 'min': <float>, 'max': <float>,
                    'std': <float>, 'label': '<axis label with units>'
                 },
                 ...
              }
            }
        """
        result: Dict[str, Any] = {
            'total': len(points),
            'per_objective': {},
        }

        if not points:
            for name in objectives:
                result['per_objective'][name] = {
                    'avg': 0.0, 'min': 0.0, 'max': 0.0, 'std': 0.0,
                    'label': to_axis_label(name),
                }
            return result

        import math

        for idx, name in enumerate(objectives):
            vals = [get_point_objective_value(p, name, idx) for p in points]
            n = len(vals)
            avg = sum(vals) / n
            variance = sum((v - avg) ** 2 for v in vals) / n
            result['per_objective'][name] = {
                'avg': avg,
                'min': min(vals),
                'max': max(vals),
                'std': math.sqrt(variance),
                'label': to_axis_label(name),
            }

        return result

    @staticmethod
    def format_correlations_table(correlations: Dict[str, float],
                                  config: EvaluatorConfig,
                                  objectives: List[str]) -> List[Dict[str, Any]]:
        """
        Format correlations as table rows.

        Rows = decision variables (from config), columns = selected objectives.
        Keys in `correlations` are expected to look like '<decision>_vs_<objective>'
        (produced by DistanceCorrelationAnalyzer using friendly names) [26].
        """
        rows = []
        for dec in config.decision_columns:
            row = {'decision': dec, 'values': []}
            for obj in objectives:
                key = f"{dec}_vs_{obj}"
                row['values'].append({
                    'objective': obj,
                    'label': to_axis_label(obj),
                    'value': correlations.get(key, 0.0),
                })
            rows.append(row)
        return rows