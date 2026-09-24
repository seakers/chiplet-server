"""Single source of truth for objective name ↔ data field mapping."""

# Friendly name (used in UI/selectedObjectives) → internal CSV/dict key
OBJECTIVE_FIELD_MAP = {
    # CASCADE
    'Energy':                'energy',
    'Runtime':               'exe_time',
    'DRAM':                  'energy_dram',
    'Memory':                'mem_accessed',
    'FLOPS':                 'flops',
    # PISTIL
    'Latency per Token':     'latency_per_token_ms',
    'Energy per Inference':  'energy_per_inference_mJ',
    'Energy per Token':      'energy_per_token_mJ',
    'Average Power':         'average_power_W',
    'System Power':          'system_power_W',
    'System Cost':           'system_cost',
    'Avg Compute Util':      'avg_comp_util',
    'Avg Memory Util':       'avg_mem_util',
    'Prefill Tokens/sec':    'prefill_tokens_per_sec',
    'System Compute':        'system_compute_TOPS',
    'System Bandwidth':      'system_bandwidth_TBps',
    'System Capacity':       'system_capacity_GB',
}

# Friendly name → axis label (with units) for plot
OBJECTIVE_AXIS_LABEL = {
    'Energy':                'Total Energy (mJ)',
    'Runtime':               'Total time (ms)',
    'DRAM':                  'DRAM Energy (mJ)',
    'Memory':                'Memory Accessed',
    'FLOPS':                 'FLOPS',
    'Latency per Token':     'Latency per Token (ms)',
    'Energy per Inference':  'Energy per Inference (mJ)',
    'Energy per Token':      'Energy per Token (mJ)',
    'Average Power':         'Average Power (W)',
    'System Power':          'System Power (W)',
    'System Cost':           'System Cost ($)',
    'Avg Compute Util':      'Avg Compute Util (%)',
    'Avg Memory Util':       'Avg Memory Util (%)',
    'Prefill Tokens/sec':    'Prefill Tokens/sec',
    'System Compute':        'System Compute (TOPS)',
    'System Bandwidth':      'System Bandwidth (TB/s)',
    'System Capacity':       'System Capacity (GB)',
}

def to_field(name: str) -> str:
    """Friendly objective name → CSV/dict field key."""
    return OBJECTIVE_FIELD_MAP.get(name, name)

def to_fields(names):
    """List of friendly names → list of field keys."""
    return [to_field(n) for n in (names or [])]

def to_axis_label(name: str) -> str:
    return OBJECTIVE_AXIS_LABEL.get(name, name)

# Default objectives per evaluator
DEFAULT_OBJECTIVES = {
    'cascade': ['Runtime', 'Energy'],
    'pistil':  ['Latency per Token', 'Energy per Inference'],
}