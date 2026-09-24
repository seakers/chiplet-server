"""
Distance Correlation Agent for analyzing relationships between design variables and objectives.
"""
import numpy as np
from typing import Dict, Any

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.analysis.distance_correlation import DistanceCorrelationAnalyzer
from api.data.loaders import PointsLoader
from api.config.objectives import OBJECTIVE_FIELD_MAP


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
        try:
            loader = PointsLoader(self.evaluator, self.run_id)
            points = context.get('points') or loader.load_points_as_dicts()

            if not points:
                return AgentResult(
                    success=False,
                    message="No data available for distance correlation analysis.",
                    error="No points data found"
                )

            # NEW: restrict to highlighted/selected indices if provided
            selected_indices = context.get('selected_indices')
            use_all = bool(context.get('use_all_points', True))  # Default to True if not specified
            subset_note = ""
            if selected_indices and not use_all:
                idx_set = set(selected_indices)
                points = [p for i, p in enumerate(points) if i in idx_set]
                subset_note = f" (restricted to {len(points)} selected points)"
                if len(points) < 5:
                    return AgentResult(
                        success=False,
                        message=f"Only {len(points)} highlighted points — need at least 5 for "
                                f"meaningful distance correlation. Ask user to expand selection "
                                f"or set use_all_points=true.",
                        error="Selection too small"
                    )

            analyzer = DistanceCorrelationAnalyzer(self.evaluator)
            objectives = context.get('objectives')

            # Pass friendly names through; analyzer.analyze_from_points
            # (updated in step 2) handles the field mapping itself.
            results = analyzer.analyze_from_points(
                points,
                requested_objectives=objectives,
            )

            return AgentResult(
                success=True,
                message=("Distance correlation analysis" + subset_note + ":\n\n"
                        + self._format_correlation_message(results)),
                data=results,
)

        except Exception as e:
            import traceback; traceback.print_exc()
            return AgentResult(
                success=False,
                message=f"Error during distance correlation analysis: {e}",
                error=str(e),
            )
        
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "objectives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Friendly objective names to analyze."
                },
                "use_all_points": {
                    "type": "boolean",
                    "description": (
                        "If true, ignore the current highlighted selection and use ALL "
                        "points. Default is true (use all points). Set to false to restrict analysis to the current highlighted points. "
                        "If the user wants a CUSTOM selection different from current "
                        "highlights, do NOT set this — instead call highlighting_agent "
                        "first to re-highlight, then call this agent."
                    ),
                    "default": True
                }
            },
            "required": [],
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