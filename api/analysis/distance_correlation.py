"""
Distance correlation analysis for chiplet design optimization.
Consolidates the repeated distance correlation logic from model.py [2].
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from dcor import distance_correlation

from api.config.evaluators import get_evaluator_config


class DistanceCorrelationAnalyzer:
    """
    Analyzes distance correlations between design variables and objectives.
    """
    
    def __init__(self, evaluator: str = 'cascade'):
        self.evaluator = evaluator.lower()
        self.config = get_evaluator_config(evaluator)
    
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
        """Normalize metric names for consistent key formatting."""
        name_map = {
            'exe_time': 'Time',
            'energy': 'Energy',
            'latency_ms': 'Latency',
            'energy_mJ': 'Energy',
        }
        return name_map.get(metric_name, metric_name.title())
    
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
    
    def analyze_from_points(self, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform full distance correlation analysis from point dictionaries.
        
        This is a convenience method that handles the full analysis pipeline [1].
        
        Args:
            points: List of point dictionaries with x, y, and chiplet values.
            
        Returns:
            Dictionary with correlations and analysis results.
        """
        if not points:
            return {'correlations': {}, 'high_impact': {}, 'summary': 'No data available.'}
        
        # Extract values based on evaluator type
        if self.evaluator == 'cascade':
            objective_vals = np.array([[p['x'], p['y']] for p in points])
            design_vals = np.array([
                [p.get('gpu', 0), p.get('attn', 0), p.get('sparse', 0), p.get('conv', 0)]
                for p in points
            ])
            metric_names = ['exe_time', 'energy']
            decision_names = ['GPU', 'Attention', 'Sparse', 'Convolution']
        elif self.evaluator == 'pistil':
            objective_vals = np.array([[p['x'], p['y']] for p in points])
            design_vals = np.array([
                [p.get('num_cus', 0), p.get('num_tmacs', 0), p.get('mem_buf_cap', 0),
                 p.get('net_buf_cap', 0), p.get('mem_banks_per_group', 0),
                 p.get('mem_ranks', 0), p.get('mem_frac_bank_cap', 0),
                 p.get('batch_size', 0), p.get('kv_cache', 0)]
                for p in points
            ])
            metric_names = ['latency_ms', 'energy_mJ']
            decision_names = self.config.decision_columns
        else:
            raise ValueError(f"Unknown evaluator: {self.evaluator}")
        
        correlations = self.calculate_correlations(
            objective_vals, design_vals, metric_names, decision_names
        )
        
        high_impact = self.get_high_impact_variables(correlations)
        summary = self.get_correlation_string(correlations)
        
        return {
            'correlations': correlations,
            'high_impact': high_impact,
            'summary': summary,
        }