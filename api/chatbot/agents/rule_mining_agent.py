"""
Rule Mining Agent for discovering association rules in Pareto-optimal designs.
"""
import numpy as np
from typing import Dict, Any, List

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.analysis.rule_mining import RuleMiner, RuleFormatter
from api.data.loaders import PointsLoader
from api.config.objectives import to_fields
from api.config.evaluators import get_evaluator_config


@AgentRegistry.register
class RuleMiningAgent(BaseAgent):

    @property
    def name(self) -> str:
        return "rule_mining_agent"

    @property
    def description(self) -> str:
        return "Performs rule mining to find patterns in Pareto-optimal designs."

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        try:
            loader = PointsLoader(self.evaluator, self.run_id)
            objectives = context.get('objectives')
            data = loader.load_deduplicated_data()
            max_pareto_rank = context.get('max_pareto_rank', 3) # this will be a fallback

            if len(data) == 0:
                return AgentResult(success=False, message="No data available.", error="No points data found")

            # NEW: restrict to highlighted/selected indices if provided
            selected_indices = context.get('selected_indices')
            point_selection = None
            if selected_indices:
                valid = [i for i in selected_indices if 0 <= i < len(data)]
                if len(valid) < 3:
                    return AgentResult(
                        success=False,
                        message=f"Only {len(valid)} highlighted points — need at least 3 for rule mining. "
                                f"Expand the selection or set use_all_points=true.",
                        error="Selection too small"
                    )
                point_selection = data[np.array(valid)]

            # Compute objective column indices into data
            cfg = get_evaluator_config(self.evaluator)
            all_obj_cols = cfg.objective_columns
            objective_col_indices = None
            if objectives:
                if cfg.objectives_first:
                    # CASCADE writes ONLY the selected objectives, in order, as the first columns.
                    # So the indices are simply 0..N-1 — NOT positions in the full config list.
                    objective_col_indices = list(range(len(objectives)))
                else:
                    # PISTIL: objectives live after decisions at fixed config positions.
                    wanted_fields = to_fields(objectives)
                    objective_col_indices = [cfg.get_objective_index(f) for f in wanted_fields if f in all_obj_cols]

            miner = RuleMiner(self.evaluator)
            rules = miner.mine_rules(
                data,                                    # ← full data, not pre-sliced
                point_selection=point_selection,          # ← optional highlighted selection
                max_pareto_rank=max_pareto_rank,
                objectives=objectives,
                objective_col_indices=objective_col_indices,
            )

            rules_string = miner.format_rules_string(rules)
            rules_dicts = miner.get_rules_as_dicts(rules)

            formatted_rules = []
            for rule_dict in rules_dicts:
                formatted = rule_dict.copy()
                formatted['formatted'] = RuleFormatter.format_rule_natural_language(
                    rule_dict.get('rule', '')
                )
                formatted_rules.append(formatted)

            return AgentResult(
                success=True,
                message=self._format_rules_message(formatted_rules),
                data={
                    'rules': formatted_rules,
                    'rules_string': rules_string,
                    'evaluator': self.evaluator,
                    'objectives': objectives,
                }
            )

        except Exception as e:
            return AgentResult(success=False, message=f"Error during rule mining: {e}", error=str(e))
    
    def _format_rules_message(self, rules: List[Dict[str, Any]]) -> str:
        """Format rules as a human-readable message."""
        if not rules:
            return "No significant association rules found in the current dataset."
        
        message = f"Rule Mining Analysis Results ({len(rules)} rules found):\n\n"
        
        for i, rule in enumerate(rules, 1):
            message += f"{i}. {rule.get('formatted', rule.get('rule', ''))}\n"
            message += f"   - Confidence (Feature→Pareto): {rule.get('conf_f_to_p', 0):.2%}\n"
            message += f"   - Confidence (Pareto→Feature): {rule.get('conf_p_to_f', 0):.2%}\n"
            message += f"   - Lift: {rule.get('lift', 0):.2f}\n\n"
        
        return message
    
    def can_handle(self, query: str) -> bool:
        """Check if query is about rule mining or patterns."""
        keywords = ['rule', 'pattern', 'mining', 'association', 'common', 'frequent']
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)
    
    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "max_pareto_rank": {
                    "type": "integer",
                    "description": (
                        "The maximum Pareto rank to include in rule mining. "
                        "For example, if the user says 'first 7 pareto ranks' this should be 7. "
                        "If the user says 'top 3' this should be 3. Defaults to 3 if not specified."
                    ),
                    "default": 3
                },
                "objectives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of objective names to use for rule mining."
                }
            },
            "required": []
        }