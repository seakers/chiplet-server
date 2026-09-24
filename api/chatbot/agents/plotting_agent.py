"""
Plotting Agent for dynamically generating plots based on user requests.
"""
import numpy as np
from typing import Dict, Any, List, Optional

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.data.loaders import PointsLoader
from api.config.objectives import OBJECTIVE_FIELD_MAP, to_field


@AgentRegistry.register
class PlottingAgent(BaseAgent):
    """
    Agent that builds plot specifications (scatter, histogram, line, box)
    from design-space data based on natural-language user requests.
    Returns Plotly-ready traces + layout for the frontend to render.
    """

    @property
    def name(self) -> str:
        return "plotting_agent"

    @property
    def description(self) -> str:
        return (
            "Creates a plot from the current run's design-space data. "
            "Supports plot_type: 'scatter', 'histogram', 'line', 'box'. "
            "For scatter/line, provide 'x' and 'y' (objective names like 'Energy', "
            "'Runtime', 'Latency per Token', or design variables like 'gpu', 'num_cus'). "
            "For histogram/box, provide a single 'x'. Optionally 'color_by' to color "
            "points by a third variable, and 'selected_indices' to restrict to "
            "highlighted points."
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        try:
            loader = PointsLoader(self.evaluator, self.run_id)
            points = context.get('points') or loader.load_points_as_dicts()

            if not points:
                return AgentResult(
                    success=False,
                    message="No data available for plotting.",
                    error="No points data found"
                )

            # Optional: restrict to highlighted subset (mirrors dcorr_agent) [6]
            selected_indices = context.get('selected_indices')
            use_all = bool(context.get('use_all_points', True))
            subset_note = ""
            if selected_indices and not use_all:
                idx_set = set(selected_indices)
                points = [p for i, p in enumerate(points) if i in idx_set]
                subset_note = f" (restricted to {len(points)} selected points)"

            plot_type = context.get('plot_type', 'scatter')
            x_name = context.get('x')
            y_name = context.get('y')
            color_by = context.get('color_by')

            if not x_name:
                return AgentResult(
                    success=False,
                    message="Please specify at least an 'x' variable to plot.",
                    error="Missing x"
                )

            x_key = self._resolve_key(x_name, points[0])
            x_data = [p.get(x_key) for p in points]

            traces = []
            layout = {
                "xaxis": {"title": self._display_name(x_name)},
                "margin": {"l": 60, "r": 30, "t": 40, "b": 60},
            }

            if plot_type in ("scatter", "line"):
                if not y_name:
                    return AgentResult(
                        success=False,
                        message="Scatter/line plots require both 'x' and 'y'.",
                        error="Missing y"
                    )
                y_key = self._resolve_key(y_name, points[0])
                y_data = [p.get(y_key) for p in points]

                trace = {
                    "x": x_data,
                    "y": y_data,
                    "mode": "markers" if plot_type == "scatter" else "lines+markers",
                    "type": "scatter",
                }
                if color_by:
                    c_key = self._resolve_key(color_by, points[0])
                    trace["marker"] = {
                        "color": [p.get(c_key) for p in points],
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": self._display_name(color_by)},
                    }
                traces.append(trace)
                layout["yaxis"] = {"title": self._display_name(y_name)}

            elif plot_type == "histogram":
                traces.append({"x": x_data, "type": "histogram"})
                layout["yaxis"] = {"title": "Count"}

            elif plot_type == "box":
                traces.append({"y": x_data, "type": "box",
                               "name": self._display_name(x_name)})

            else:
                return AgentResult(
                    success=False,
                    message=f"Unsupported plot_type '{plot_type}'.",
                    error=f"Unsupported plot_type: {plot_type}"
                )

            title = context.get('title') or self._auto_title(plot_type, x_name, y_name)
            layout["title"] = title

            print(f"Plotting Agent Returns: plot_type={plot_type}, title={title}, points={len(points)}, traces={len(traces)}, layout={layout}")

            return AgentResult(
                success=True,
                message=f"Created a {plot_type} plot: {title}{subset_note}.",
                data={
                    "plot_type": plot_type,
                    "traces": traces,
                    "layout": layout,
                    "title": title,
                    "point_count": len(points),
                }
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return AgentResult(
                success=False,
                message=f"Error creating plot: {str(e)}",
                error=str(e)
            )

    # --- Key resolution: reuse the same friendly-name maps as your other components ---
    def _resolve_key(self, name: str, sample_point: dict) -> str:
        available = sample_point.keys()

        # CASCADE design-variable aliases (matches DistanceCorrelation.vue) [3]
        cascade_map = {
            'GPU': 'gpu', 'Sparse': 'sparse',
            'Attention': 'attn', 'Convolution': 'conv',
            'gpu': 'gpu', 'sparse': 'sparse', 'attn': 'attn', 'conv': 'conv',
        }
        # Objective friendly-name → field (matches getObjectiveValueForPoint) [1]
        friendly_to_field = {
            'Energy': 'y', 'Runtime': 'x',
            'Latency per Token': 'latency_per_token_ms',
            'Energy per Inference': 'energy_per_inference_mJ',
            'Energy per Token': 'energy_per_token_mJ',
            'Average Power': 'average_power_W',
            'System Power': 'system_power_W',
            'System Cost': 'system_cost',
            'Avg Compute Util': 'avg_comp_util',
            'Avg Memory Util': 'avg_mem_util',
            'Prefill Tokens/sec': 'prefill_tokens_per_sec',
            'System Compute': 'system_compute_TOPS',
            'System Bandwidth': 'system_bandwidth_TBps',
            'System Capacity': 'system_capacity_GB',
        }

        if name in friendly_to_field and friendly_to_field[name] in available:
            return friendly_to_field[name]
        if name in cascade_map and cascade_map[name] in available:
            return cascade_map[name]
        if name in available:
            return name
        # snake_case fallback
        snake = name.lower().replace(' ', '_')
        if snake in available:
            return snake
        return name  # last resort — will produce None values, logged upstream

    def _display_name(self, name: str) -> str:
        return name

    def _auto_title(self, plot_type, x_name, y_name):
        if plot_type in ("scatter", "line") and y_name:
            return f"{y_name} vs {x_name}"
        return f"{plot_type.capitalize()} of {x_name}"

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": ["scatter", "histogram", "line", "box"],
                    "description": "Type of plot to generate. Default 'scatter'."
                },
                "x": {
                    "type": "string",
                    "description": (
                        "X-axis variable. Objective friendly names ('Energy', "
                        "'Runtime', 'Latency per Token', etc.) or design variables "
                        "(CASCADE: 'gpu','attn','sparse','conv'; PISTIL: 'num_cus', "
                        "'num_tmacs','batch_size', etc.)."
                    )
                },
                "y": {
                    "type": "string",
                    "description": "Y-axis variable (required for scatter/line)."
                },
                "color_by": {
                    "type": "string",
                    "description": "Optional third variable to color points by."
                },
                "title": {"type": "string", "description": "Optional plot title."},
                "use_all_points": {
                    "type": "boolean",
                    "description": "If false, restrict to highlighted selection.",
                    "default": True
                }
            },
            "required": ["x"]
        }

    def can_handle(self, query: str) -> bool:
        keywords = ['plot', 'graph', 'chart', 'visualize', 'scatter',
                    'histogram', 'distribution', 'show me a plot']
        return any(kw in query.lower() for kw in keywords)