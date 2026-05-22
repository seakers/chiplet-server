"""
Distance Correlation Agent for analyzing relationships between design variables and objectives.
"""
import numpy as np
from typing import Dict, Any

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.analysis.distance_correlation import DistanceCorrelationAnalyzer
from api.data.loaders import PointsLoader


@AgentRegistry.register
class DistanceCorrelationAgent(BaseAgent):
    """
    Agent that calculates distance correlations between design variables and objectives.
    Consolidates the dcor_manager logic from model.py [2].
    """
    
    @property
    def name(self) -> str:
        return "dcorr_agent"
    
    @property
    def description(self) -> str:
        return "Calculates distance correlations between design variables and performance objectives."
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute distance correlation analysis.
        
        Args:
            context: Dictionary containing points data or file path.
            
        Returns:
            AgentResult with correlation analysis.
        """
        try:
            # Load data
            loader = PointsLoader(self.evaluator, self.run_id)
            points = context.get('points') or loader.load_points_as_dicts()
            
            if not points:
                return AgentResult(
                    success=False,
                    message="No data available for distance correlation analysis.",
                    error="No points data found"
                )
            
            # Perform analysis
            analyzer = DistanceCorrelationAnalyzer(self.evaluator)
            results = analyzer.analyze_from_points(points)
            
            # Format response message
            message = self._format_correlation_message(results)
            
            return AgentResult(
                success=True,
                message=message,
                data=results
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error during distance correlation analysis: {str(e)}",
                error=str(e)
            )
        
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "metric_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Objective metric names to analyze"
                }
            },
            "required": []
        }
    
    def _format_correlation_message(self, results: Dict[str, Any]) -> str:
        """Format correlation results as a human-readable message."""
        correlations = results.get('correlations', {})
        high_impact = results.get('high_impact', {})
        
        message = "Distance Correlation Analysis Results:\n\n"
        
        # Add summary of high-impact variables
        if high_impact:
            message += "High-Impact Variables:\n"
            for objective, variables in high_impact.items():
                message += f"\n  {objective}:\n"
                for var_info in variables[:3]:  # Top 3
                    message += f"    - {var_info['variable']}: {var_info['correlation']:.4f}\n"
        
        # Add full correlation table
        message += "\nFull Correlation Table:\n"
        message += results.get('summary', '')
        
        return message
    
    def can_handle(self, query: str) -> bool:
        """Check if query is about distance correlations."""
        keywords = ['correlation', 'dcorr', 'relationship', 'impact', 'affect', 'influence']
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)