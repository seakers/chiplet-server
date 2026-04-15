"""
Evaluator-specific configurations.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class EvaluatorConfig:
    """Configuration for a specific evaluator type."""
    name: str
    decision_columns: List[str]
    objective_columns: List[str]
    num_objectives: int = 2
    num_slots: int = 12
    csv_has_header: bool = False
    
    # Column indices in CSV (objectives first, then decisions for CASCADE)
    # For PISTIL: decisions first, then objectives
    objectives_first: bool = True
    
    # Constraint level thresholds (relative to max)
    constraint_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'none': 0.0,
        'low': 0.33,
        'medium': 0.66,
        'high': 1.0,
    })
    
    def get_decision_index(self, col_name: str) -> int:
        """Get the index of a decision column in the CSV."""
        try:
            idx = self.decision_columns.index(col_name)
            if self.objectives_first:
                return self.num_objectives + idx
            return idx
        except ValueError:
            raise ValueError(f"Unknown decision column: {col_name}")
    
    def get_objective_index(self, col_name: str) -> int:
        """Get the index of an objective column in the CSV."""
        try:
            idx = self.objective_columns.index(col_name)
            if self.objectives_first:
                return idx
            return len(self.decision_columns) + idx
        except ValueError:
            raise ValueError(f"Unknown objective column: {col_name}")


# Pre-configured evaluators
CASCADE_CONFIG = EvaluatorConfig(
    name='cascade',
    decision_columns=['GPU', 'Attention', 'Sparse', 'Convolution'],
    objective_columns=['exe_time', 'energy'],
    num_objectives=2,
    num_slots=12,
    csv_has_header=False,
    objectives_first=True,
)

PISTIL_CONFIG = EvaluatorConfig(
    name='pistil',
    decision_columns=[
        'num_cus', 'num_tmacs', 'mem_buf_cap', 'net_buf_cap',
        'mem_banks_per_group', 'mem_ranks', 'mem_frac_bank_cap',
        'batch_size', 'kv_cache'
    ],
    objective_columns=['latency_ms', 'energy_mJ'],
    num_objectives=2,
    csv_has_header=True,
    objectives_first=False,  # PISTIL has decisions first
)

EVALUATOR_CONFIGS: Dict[str, EvaluatorConfig] = {
    'cascade': CASCADE_CONFIG,
    'pistil': PISTIL_CONFIG,
}


def get_evaluator_config(evaluator_name: str) -> EvaluatorConfig:
    """Get the configuration for a specific evaluator."""
    config = EVALUATOR_CONFIGS.get(evaluator_name.lower())
    if not config:
        raise ValueError(f"Unknown evaluator: {evaluator_name}. Available: {list(EVALUATOR_CONFIGS.keys())}")
    return config