"""
Rule Mining Agent for discovering association rules in Pareto-optimal designs.
"""
import numpy as np
from typing import Dict, Any, List

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.analysis.rule_mining import RuleMiner, RuleFormatter
from api.data.loaders import PointsLoader


@AgentRegistry.register
class RuleMiningAgent(BaseAgent):
    """
    Agent that performs association rule mining on chiplet design data.
    Consolidates the rule_mining logic from model.py [2].
    """
    
    @property
    def name(self) -> str:
        return "rule_mining_agent"
    
    @property
    def description(self) -> str:
        return "Performs rule mining to find patterns in Pareto-optimal designs."
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute rule mining analysis.
        
        Args:
            context: Dictionary containing points data or analysis parameters.
            
        Returns:
            AgentResult with discovered rules.
        """
        try:
            # Load data
            loader = PointsLoader(self.evaluator, self.run_id)
            data = loader.load_deduplicated_data()
            
            if len(data) == 0:
                return AgentResult(
                    success=False,
                    message="No data available for rule mining analysis.",
                    error="No points data found"
                )
            
            # Get analysis parameters
            point_selection_params = context.get('point_selection_params')
            max_pareto_rank = context.get('max_pareto_rank', 3)
            
            # Perform rule mining
            miner = RuleMiner(self.evaluator)
            rules = miner.mine_rules(data, max_pareto_rank=max_pareto_rank)
            
            # Format response
            rules_string = miner.format_rules_string(rules)
            rules_dicts = miner.get_rules_as_dicts(rules)
            
            # Add natural language formatting
            formatted_rules = []
            for rule_dict in rules_dicts:
                formatted_rule = rule_dict.copy()
                formatted_rule['formatted'] = RuleFormatter.format_rule_natural_language(
                    rule_dict.get('rule', '')
                )
                formatted_rules.append(formatted_rule)
            
            message = self._format_rules_message(formatted_rules)
            
            return AgentResult(
                success=True,
                message=message,
                data={
                    'rules': formatted_rules,
                    'rules_string': rules_string,
                    'evaluator': self.evaluator,
                }
            )
            
        except Exception as e:
            print(f"Error during rule mining: {str(e)}")
            return AgentResult(
                success=False,
                message=f"Error during rule mining: {str(e)}",
                error=str(e)
            )
    
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