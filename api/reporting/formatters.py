"""
Formatting utilities for reports and data display.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime

from api.analysis.rule_mining import RuleFormatter


class ReportFormatter:
    """
    Formats data for report generation.
    """
    
    @staticmethod
    def format_run_params(run_params: Dict[str, Any]) -> Dict[str, str]:
        """Format run parameters for display."""
        return {
            'model': str(run_params.get('model', 'Unknown')),
            'algorithm': str(run_params.get('algorithm', 'Unknown')),
            'objectives': ', '.join(run_params.get('objectives', [])),
            'population_size': str(run_params.get('population_size', 'N/A')),
            'generations': str(run_params.get('generations', 'N/A')),
            'trace_name': str(run_params.get('trace_name', 'Unknown')),
        }
    
    @staticmethod
    def format_statistics(points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate and format statistics from points."""
        if not points:
            return {
                'total': 0,
                'avg_time': 0,
                'avg_energy': 0,
                'min_time': 0,
                'min_energy': 0,
                'max_time': 0,
                'max_energy': 0,
            }
        
        times = [p.get('x', 0) for p in points]
        energies = [p.get('y', 0) for p in points]
        
        return {
            'total': len(points),
            'avg_time': sum(times) / len(times),
            'avg_energy': sum(energies) / len(energies),
            'min_time': min(times),
            'min_energy': min(energies),
            'max_time': max(times),
            'max_energy': max(energies),
        }
    
    @staticmethod
    def format_correlations_table(correlations: Dict[str, float], 
                                   chiplet_types: List[str] = None) -> List[Dict[str, Any]]:
        """Format correlations as table rows."""
        if chiplet_types is None:
            chiplet_types = ['GPU', 'Attention', 'Sparse', 'Convolution']
        
        rows = []
        for chiplet in chiplet_types:
            rows.append({
                'chiplet': chiplet,
                'vs_energy': correlations.get(f'{chiplet}_vs_Energy', 0),
                'vs_time': correlations.get(f'{chiplet}_vs_Time', 0),
            })
        return rows