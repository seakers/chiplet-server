"""
Association rule mining for chiplet design optimization.
Consolidates the rule mining logic from model.py [2].
"""
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from copy import deepcopy
from dataclasses import dataclass, field

from api.config.evaluators import get_evaluator_config
from .pareto import ParetoCalculator
from api.config.objectives import to_fields


@dataclass
class MiningRule:
    """Represents a discovered association rule."""
    rule_set: Set[str]
    conf_f_to_p: float  # Confidence: feature implies Pareto
    conf_p_to_f: float  # Confidence: Pareto implies feature
    lift: float
    support: float = 0.0
    
    @property
    def rule_string(self) -> str:
        """Get rule as a formatted string."""
        return " AND ".join(sorted(self.rule_set))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'rule': self.rule_string,
            'conf_f_to_p': self.conf_f_to_p,
            'conf_p_to_f': self.conf_p_to_f,
            'lift': self.lift,
            'support': self.support,
        }


class RuleMiner:
    """
    Performs association rule mining on chiplet design data.
    Finds patterns that correlate with Pareto-optimal designs.
    """
    
    # Feature bins for categorizing design variable values
    FEATURE_BINS = ['none', 'low', 'medium', 'high']
    
    def __init__(self, evaluator: str = 'cascade', min_support: float = 0.01):
        self.evaluator = evaluator.lower()
        self.config = get_evaluator_config(evaluator)
        self.min_support = min_support
        self.pareto_calculator = ParetoCalculator()
    
    def mine_rules(self,
                   data: np.ndarray,
                   point_selection: np.ndarray = None,
                   max_pareto_rank: int = 3,
                   objectives: List[str] = None,
                   objective_col_indices: List[int] = None) -> List[MiningRule]:
        """
        Args:
            data: full dataset, columns = [all_objectives..., decisions...]
            objectives: friendly objective names selected by user.
            objective_col_indices: explicit column indices for the selected
                                   objectives within `data`. If provided, used
                                   directly. Otherwise we assume the first
                                   `len(objectives)` columns ARE the selected
                                   objectives (legacy behaviour).
        """
        if len(data) == 0:
            return []

        decision_cols = self.config.decision_columns
        all_obj_cols = self.config.objective_columns  # internal field names
        num_objectives = len(objective_col_indices) if objective_col_indices else len(objectives) if objectives else len(all_obj_cols)

        point_vals = data[:, -num_objectives:] if self.config.objectives_first else data[:, :num_objectives]
        ranks = ParetoCalculator.calculate_pareto_ranks(point_vals)
        
        # Add ranks to data
        data_with_ranks = np.hstack((data, ranks.reshape(-1, 1)))
        
        # Select points for analysis
        if point_selection is None:
            point_selection = data_with_ranks[data_with_ranks[:, -1] < max_pareto_rank]
        
        if len(point_selection) == 0:
            return []
        
        # Build rules dictionary
        rules_dict = self._build_rules_dict(data_with_ranks, decision_cols, num_objectives)

        print("Rules dict built! Keys:", list(rules_dict.keys()))
        
        # Find Pareto-optimal rule combinations
        pfront_rules, pfront_costs, pfront_lifts = self._find_pareto_rules(
            rules_dict, point_selection, data_with_ranks
        )
        
        # Convert to MiningRule objects
        rules = []
        for i, rule_set in enumerate(pfront_rules):
            rules.append(MiningRule(
                rule_set=rule_set,
                conf_f_to_p=float(pfront_costs[i, 1]),
                conf_p_to_f=float(pfront_costs[i, 0]),
                lift=float(pfront_lifts[i, 0]),
            ))
        
        return rules
    
    def _build_rules_dict(self, 
                          data: np.ndarray, 
                          decision_cols: List[str],
                          num_objectives: int) -> Dict[str, np.ndarray]:
        """
        Build dictionary mapping rule names to matching points.
        
        Rules are defined based on relative ranges of design variables [2].
        """
        rules_dict = {}
        n_cols = data.shape[1]
        available_decision_cols = n_cols - num_objectives

        if available_decision_cols <= 0:
            print(f"[RuleMiner] WARNING: data has {n_cols} cols but {num_objectives} objectives "
                f"— no decision columns available.")
            return rules_dict

        # Only iterate over decision columns that actually exist in data
        actual_decision_cols = decision_cols[:available_decision_cols]
        if len(actual_decision_cols) < len(decision_cols):
            print(f"[RuleMiner] WARNING: config has {len(decision_cols)} decision cols "
                f"but data only has {available_decision_cols}. "
                f"Using: {actual_decision_cols}")

        for chip_idx, col_name in enumerate(actual_decision_cols):
            col_idx = chip_idx + num_objectives
            col_data = data[:, col_idx]
            
            val_min = np.min(col_data)
            val_max = np.max(col_data)
            
            for feature in self.FEATURE_BINS:
                if feature == 'none':
                    mask = col_data == 0
                elif feature == 'low':
                    mask = (col_data > 0) & (col_data <= val_max * 0.33)
                elif feature == 'medium':
                    mask = (col_data > val_max * 0.33) & (col_data <= val_max * 0.66)
                elif feature == 'high':
                    mask = col_data > val_max * 0.66
                else:
                    continue
                
                rule_name = f"{col_name}_{feature}"
                rules_dict[rule_name] = data[mask]
        
        return rules_dict
    
    def _find_pareto_rules(self,
                           rules_dict: Dict[str, np.ndarray],
                           point_selection: np.ndarray,
                           full_data: np.ndarray) -> Tuple[List[Set[str]], np.ndarray, np.ndarray]:
        """
        Find rule combinations on the Pareto front of confidence values.
        
        This implements the iterative rule addition algorithm from model.py [2].
        """
        pfront_rules: List[Set[str]] = []
        pfront_costs = np.array([]).reshape(0, 2)
        pfront_lifts = np.array([]).reshape(0, 1)
        base_rule: Set[str] = set()
        
        pfront_rules, pfront_costs, pfront_lifts = self._add_rules(
            rules_dict, point_selection, pfront_rules, 
            pfront_costs, pfront_lifts, base_rule, full_data
        )
        
        return pfront_rules, pfront_costs, pfront_lifts
    
    def _add_rules(self,
                   rules_dict: Dict[str, np.ndarray],
                   point_selection: np.ndarray,
                   pfront_rules: List[Set[str]],
                   pfront_costs: np.ndarray,
                   pfront_lifts: np.ndarray,
                   base_rule: Set[str],
                   full_data: np.ndarray) -> Tuple[List[Set[str]], np.ndarray, np.ndarray]:
        """
        Iteratively add rules with high support on the Pareto front.
        
        This is the core recursive algorithm from model.py [2].
        """
        prev_pfront_rules = deepcopy(pfront_rules)
        
        if len(pfront_rules) == 0:
            _, pfront_rules, pfront_costs, pfront_lifts = self._find_confidences(
                rules_dict, point_selection, pfront_rules,
                pfront_costs, pfront_lifts, base_rule, full_data
            )
        else:
            for rule_set in pfront_rules:
                _, pfront_rules, pfront_costs, pfront_lifts = self._find_confidences(
                    rules_dict, point_selection, pfront_rules,
                    pfront_costs, pfront_lifts, rule_set, full_data
                )
        
        # Filter to Pareto front (maximize confidence values)
        if len(pfront_costs) > 0:
            pfront_mask = ParetoCalculator.is_pareto_efficient(-pfront_costs, return_mask=True)
            pfront_costs = pfront_costs[pfront_mask]
            pfront_lifts = pfront_lifts[pfront_mask]
            pfront_rules = [pfront_rules[i] for i in range(len(pfront_rules)) if pfront_mask[i]]
        
        print(f"New Rules Found: {len(pfront_rules)} (previously {len(prev_pfront_rules)})")
        # Recurse if we found new rules
        if pfront_rules != prev_pfront_rules:
            pfront_rules, pfront_costs, pfront_lifts = self._add_rules(
                rules_dict, point_selection, pfront_rules,
                pfront_costs, pfront_lifts, base_rule, full_data
            )
        
        return pfront_rules, pfront_costs, pfront_lifts
    
    def _find_confidences(self,
                          rules_dict: Dict[str, np.ndarray],
                          point_selection: np.ndarray,
                          pfront_rules: List[Set[str]],
                          pfront_costs: np.ndarray,
                          pfront_lifts: np.ndarray,
                          base_rule: Set[str],
                          full_data: np.ndarray) -> Tuple[int, List[Set[str]], np.ndarray, np.ndarray]:
        """
        Calculate confidence values for rule combinations.
        
        This implements the confidence calculation from model.py [2].
        """
        new_rules = 0
        
        for rule in rules_dict:
            new_rule = deepcopy(base_rule)
            new_rule.add(rule)
            
            if new_rule not in pfront_rules:  # Don't calculate for duplicate rule sets
                # Get intersection of all rule point sets
                rule_points_all = [set(map(tuple, rules_dict[r])) for r in new_rule]
                rule_point_set = np.array(list(set.intersection(*rule_points_all)))
                
                if len(rule_point_set) == 0:
                    continue
                
                # Find points that are both in rule set and in point selection (Pareto front)
                p_and_f = []
                for point in rule_point_set:
                    if any(np.equal(point, point_selection).all(axis=1)):
                        p_and_f.append(point)
                
                len_p_and_f = len(p_and_f)
                if len_p_and_f == 0:
                    continue
                
                # Check if support is high enough
                rule_support = len_p_and_f / len(full_data)
                if rule_support >= self.min_support:
                    # Calculate confidence values
                    conf_p_to_f = len_p_and_f / len(point_selection)  # P(F|Pareto)
                    conf_f_to_p = len_p_and_f / len(rule_point_set)   # P(Pareto|F)
                    
                    # Calculate lift
                    lift = (len_p_and_f * len(full_data)) / (len(point_selection) * len(rule_point_set))
                    
                    # Add to results
                    pfront_costs = np.vstack((pfront_costs, np.array([conf_p_to_f, conf_f_to_p])))
                    pfront_rules.append(new_rule)
                    pfront_lifts = np.vstack((pfront_lifts, np.array([lift])))
                    
                    new_rules += 1
        
        return new_rules, pfront_rules, pfront_costs, pfront_lifts
    
    def format_rules_string(self, rules: List[MiningRule], evaluator_name: str = None) -> str:
        """
        Format rules as a human-readable string.
        
        This matches the format expected by the chatbot [2].
        """
        if evaluator_name is None:
            evaluator_name = self.evaluator.upper()
        
        rule_str = f"Analysis for Evaluator: {evaluator_name}\n"
        rule_str += "Rules were defined based on relative ranges of the design variables "
        rule_str += "(none, low, medium, high).\n\n"
        
        if not rules:
            rule_str += "No significant rules found in the current dataset."
        else:
            for rule in rules:
                rule_str += f"Rule: {rule.rule_string}, "
                rule_str += f"conf(f->p): {rule.conf_f_to_p:.4f}, "
                rule_str += f"conf(p->f): {rule.conf_p_to_f:.4f}, "
                rule_str += f"lift: {rule.lift:.4f}\n\n"
        
        return rule_str
    
    def get_rules_as_dicts(self, rules: List[MiningRule]) -> List[Dict[str, Any]]:
        """Convert rules to list of dictionaries for JSON serialization."""
        return [rule.to_dict() for rule in rules]


class RuleFormatter:
    """
    Formats rules into human-readable natural language.
    Consolidates the repeated format_rule_natural_language function from views.py [1].
    """
    
    # Mapping of chiplet types to display names
    CHIPLET_DISPLAY_NAMES = {
        'GPU': 'GPU',
        'Attention': 'Attention',
        'Sparse': 'Sparse',
        'Convolution': 'Convolution',
        'num_cus': 'Compute Units',
        'num_tmacs': 'Tensor MACs',
        'mem_buf_cap': 'Memory Buffer Capacity',
        'net_buf_cap': 'Network Buffer Capacity',
        'mem_banks_per_group': 'Memory Banks per Group',
        'mem_ranks': 'Memory Ranks',
        'mem_frac_bank_cap': 'Fractional Bank Capacity',
        'batch_size': 'Batch Size',
        'kv_cache': 'KV Cache',
    }
    
    # Mapping of feature levels to descriptions
    FEATURE_DESCRIPTIONS = {
        'none': 'no',
        'low': 'a low number of',
        'medium': 'a moderate number of',
        'high': 'a high number of',
        'very high': 'a very high number of',
    }
    
    @classmethod
    def format_rule_natural_language(cls, rule_string: str) -> str:
        """
        Convert a rule string like "GPU_high AND Attention_low" to natural language.
        
        This consolidates the duplicated format_rule_natural_language function [1].
        """
        if not rule_string:
            return ""
        
        parts = rule_string.split(' AND ')
        descriptions = []
        
        for part in parts:
            part = part.strip()
            if '_' in part:
                # Split on last underscore to handle names with underscores
                last_underscore = part.rfind('_')
                chiplet_type = part[:last_underscore]
                level = part[last_underscore + 1:]
                
                # Get display name and description
                display_name = cls.CHIPLET_DISPLAY_NAMES.get(chiplet_type, chiplet_type)
                level_desc = cls.FEATURE_DESCRIPTIONS.get(level, level)
                
                descriptions.append(f"{level_desc} {display_name}")
        
        if not descriptions:
            return rule_string
        
        if len(descriptions) == 1:
            return f"Designs with {descriptions[0]}"
        elif len(descriptions) == 2:
            return f"Designs with {descriptions[0]} and {descriptions[1]}"
        else:
            return f"Designs with {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
    
    @classmethod
    def format_rules_for_report(cls, rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format a list of rules for HTML report display.
        
        Adds 'formatted_rule' key with natural language description.
        """
        formatted_rules = []
        for rule in rules:
            formatted_rule = rule.copy()
            formatted_rule['formatted_rule'] = cls.format_rule_natural_language(rule.get('rule', ''))
            formatted_rules.append(formatted_rule)
        return formatted_rules