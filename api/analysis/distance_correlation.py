"""
Distance correlation analysis for chiplet design optimization.
Consolidates the repeated distance correlation logic from model.py [2].
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from dcor import distance_correlation

from api.config.evaluators import get_evaluator_config
from api.config.objectives import OBJECTIVE_FIELD_MAP, to_fields, DEFAULT_OBJECTIVES


class DistanceCorrelationAnalyzer:
    """
    Analyzes distance correlations between design variables and objectives.
    """
    
    def __init__(self, evaluator: str = 'cascade'):
        self.evaluator = evaluator.lower()
        self.config = get_evaluator_config(evaluator)

        self.OBJECTIVE_KEY_MAP = OBJECTIVE_FIELD_MAP
    
    def calculate_correlations(self, 
                               objective_vals: np.ndarray, 
                               design_vals: np.ndarray,
                               metric_names: List[str] = None,
                               decision_names: List[str] = None) -> Dict[str, float]:
        """
        Calculate distance correlations between objectives and design variables.
        
        Args:
            objective_vals: (n_points, n_objectives) array.
            design_vals: (n_points, n_decisions) array.
            metric_names: Names of objective columns (optional, uses config default).
            decision_names: Names of decision columns (optional, uses config default).
            
        Returns:
            Dictionary mapping "{decision}_vs_{objective}" to correlation value.
        """
        if metric_names is None:
            metric_names = self.config.objective_columns
        if decision_names is None:
            decision_names = self.config.decision_columns
        
        correlations = {}
        
        for obj_idx, obj_name in enumerate(metric_names):
            for dec_idx, dec_name in enumerate(decision_names):
                obj_column = np.array(objective_vals[:, obj_idx], dtype=float)
                dec_column = np.array(design_vals[:, dec_idx], dtype=float)
                
                # Handle constant columns (would cause division by zero)
                if np.std(obj_column) == 0 or np.std(dec_column) == 0:
                    corr = 0.0
                else:
                    corr = float(distance_correlation(obj_column, dec_column))
                
                # Store with standardized key format
                key = f"{dec_name}_vs_{self._normalize_metric_name(obj_name)}"
                correlations[key] = corr
        
        return correlations
    
    def _normalize_metric_name(self, metric_name: str) -> str:
        # Friendly names come pre-formatted now; just return as-is.
        return metric_name
    
    def get_correlation_string(self, correlations: Dict[str, float]) -> str:
        """
        Format correlations as a human-readable string.
        
        This matches the format expected by the chatbot [2].
        """
        lines = []
        for key, value in sorted(correlations.items()):
            parts = key.split('_vs_')
            if len(parts) == 2:
                decision, objective = parts
                lines.append(
                    f"Distance correlation between objective '{objective}' "
                    f"and variable '{decision}' is {value:.4f}."
                )
        return "\n".join(lines)
    
    def get_high_impact_variables(self, 
                                   correlations: Dict[str, float], 
                                   threshold: float = 0.3) -> Dict[str, List[str]]:
        """
        Identify high-impact design variables for each objective.
        
        Args:
            correlations: Dictionary of correlation values.
            threshold: Minimum correlation to be considered high impact.
            
        Returns:
            Dictionary mapping objective names to lists of high-impact variables.
        """
        high_impact = {}
        
        for key, value in correlations.items():
            if value >= threshold:
                parts = key.split('_vs_')
                if len(parts) == 2:
                    decision, objective = parts
                    if objective not in high_impact:
                        high_impact[objective] = []
                    high_impact[objective].append({
                        'variable': decision,
                        'correlation': value
                    })
        
        # Sort by correlation value descending
        for objective in high_impact:
            high_impact[objective].sort(key=lambda x: x['correlation'], reverse=True)
        
        return high_impact
    
    
    def analyze_from_points(self, points, requested_objectives=None, metric_names=None):
        """
        Perform full distance correlation analysis from point dictionaries.

        Args:
            points: list of point dicts loaded by PointsLoader.
            requested_objectives: list of *friendly* names (e.g. ['Energy','Runtime']).
            metric_names: list of *internal* field keys (e.g. ['energy','exe_time']).
                          Takes priority over requested_objectives if both given.
        """
        if not points:
            return {'correlations': {}, 'high_impact': {}, 'summary': 'No data available.'}

        # Resolve metric (objective) field keys
        if metric_names:
            obj_fields = metric_names
            obj_labels = metric_names  # already internal; use as-is for keys
        elif requested_objectives:
            obj_fields = to_fields(requested_objectives)
            obj_labels = requested_objectives
        else:
            defaults = DEFAULT_OBJECTIVES.get(self.evaluator, [])
            obj_fields = to_fields(defaults) or self.config.objective_columns
            obj_labels = defaults or self.config.objective_columns

        decision_names = self.config.decision_columns

        # Build objective matrix using internal field keys
        objective_vals = np.array([
            [float(p.get(field, 0) or 0) for field in obj_fields]
            for p in points
        ])

        # Decision keys: CASCADE points use 'gpu','attn','sparse','conv';
        # PISTIL points use the same names as decision_columns.
        cascade_map = {'GPU': 'gpu', 'Attention': 'attn',
                       'Sparse': 'sparse', 'Convolution': 'conv'}
        def _dec_key(name):
            return cascade_map.get(name, name)

        design_vals = np.array([
            [float(p.get(_dec_key(dec), 0) or 0) for dec in decision_names]
            for p in points
        ])

        # Pass FRIENDLY labels as metric_names so correlation keys end up
        # human-readable (e.g. "GPU_vs_Energy" instead of "GPU_vs_energy")
        correlations = self.calculate_correlations(
            objective_vals, design_vals,
            metric_names=obj_labels,
            decision_names=decision_names
        )
        # print(f"Calculated correlations: {correlations}")
        high_impact = self.get_high_impact_variables(correlations)
        summary = self.get_correlation_string(correlations)

        return {
            'correlations': correlations,
            'high_impact': high_impact,
            'summary': summary,
        }