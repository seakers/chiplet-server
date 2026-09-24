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

    def minimize_for(self, objective_name: str) -> bool:
        """Return True if this objective should be minimized."""
        maximize = {'avg_comp_util', 'avg_mem_util', 'prefill_tokens_per_sec',
                    'system_compute_TOPS', 'system_bandwidth_TBps', 'system_capacity_GB'}
        return objective_name not in maximize


# Replace PISTIL_CONFIG objective_columns with full list
PISTIL_CONFIG = EvaluatorConfig(
    name='pistil',
    decision_columns=[
        'num_cus', 'num_tmacs', 'mem_buf_cap', 'net_buf_cap',
        'mem_banks_per_group', 'mem_ranks', 'mem_frac_bank_cap',
        'batch_size', 'kv_cache'
    ],
    objective_columns=[
        'latency_ms', 'energy_mJ', 'latency_per_token_ms', 'energy_per_inference_mJ', 
        'energy_per_token_mJ', 'prefill_tokens_per_sec', 'average_power_W', 'system_power_W', 
        'average_power_mem_W', 'average_power_comp_W', 'average_power_net_W', 'system_cost', 
        'chiplet_silicon_cost', 'memory_cost_2xHBLC', 'package_cost', 'package_silicon_cost', 
        'package_memory_cost', 'package_substrate_cost', 'system_silicon_cost', 'system_memory_cost', 
        'system_substrate_cost', 'system_pcb_cost', 'avg_comp_util', 'avg_mem_util', 
        'system_compute_TOPS', 'system_bandwidth_TBps', 'system_capacity_GB', 'num_packages', 
        'num_chiplets', 'hblc_capacity_GB', 'hblc_bandwidth_GBps', 'hblc_bw_per_capacity', 
        'peak_buffer_MB', 'avg_cache_util_MB', 'model_weight_capacity_GB', 
        'kv_cache_capacity_per_batch_GB', 'total_model_capacity_GB', 'cores_per_cu', 
        'mem_buffer_size', 'net_buffer_size', 'tmacs_per_core', 'hblc_bank_groups', 
        'hblc_ranks', 'hblc_frac_bank_cap', 'algorithm'
    ],
    num_objectives=12,  # total available; user picks ≤3
    csv_has_header=True,
    objectives_first=False,
)

CASCADE_CONFIG = EvaluatorConfig(
    name='cascade',
    decision_columns=['GPU', 'Attention', 'Sparse', 'Convolution'],
    objective_columns=['exe_time', 'energy', 'energy_dram', 'mem_accessed', 'flops'],  # extend as supported
    num_objectives=5,
    num_slots=12,
    csv_has_header=False,
    objectives_first=True,
)

EVALUATOR_CONFIGS: Dict[str, EvaluatorConfig] = {
    'cascade': CASCADE_CONFIG,
    'pistil': PISTIL_CONFIG,
}


def get_evaluator_config(evaluator_name: str, num_objs: int = None) -> EvaluatorConfig:
    """Get the configuration for a specific evaluator."""
    config = EVALUATOR_CONFIGS.get(evaluator_name.lower())
    if not config:
        raise ValueError(f"Unknown evaluator: {evaluator_name}. Available: {list(EVALUATOR_CONFIGS.keys())}")
    if num_objs is not None:
        config.num_objectives = num_objs
    return config