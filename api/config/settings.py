"""
Centralized configuration settings for the chiplet design tool.
"""
import os
import sys

# Base paths
BASE_DIR = sys.path[0]
API_DIR = os.path.join(BASE_DIR, 'api')

# Evaluator paths
EVALUATOR_BASE_PATHS = {
    'cascade': os.path.join(API_DIR, 'Evaluator/cascade/chiplet_model'),
    'pistil': os.path.join(API_DIR, 'Evaluator/sim-v2-4-pistil-sim-clean'),
}

# Results directories
def get_results_dir(evaluator: str, run_id: str = None) -> str:
    """Get the results directory for a given evaluator and optional run_id."""
    base = EVALUATOR_BASE_PATHS.get(evaluator.lower())
    if not base:
        raise ValueError(f"Unknown evaluator: {evaluator}")
    
    if evaluator.lower() == 'cascade':
        return os.path.join(base, 'dse/results')
    elif evaluator.lower() == 'pistil':
        if run_id:
            return os.path.join(base, f'dse/results/{run_id}')
        return os.path.join(base, 'dse/results')
    return os.path.join(base, 'dse/results')

def get_points_file(evaluator: str, run_id: str = None) -> str:
    """Get the points.csv file path for a given evaluator."""
    results_dir = get_results_dir(evaluator, run_id)
    return os.path.join(results_dir, 'points.csv')

# Default optimization parameters
DEFAULT_POPULATION_SIZE = 50
DEFAULT_GENERATIONS = 100
DEFAULT_EPISODES = 100
DEFAULT_MINI_BATCH_SIZE = 32

# Constraint thresholds (relative to max value)
CONSTRAINT_THRESHOLDS = {
    'none': 0.0,
    'low': 0.33,
    'medium': 0.66,
    'high': 1.0,
}

# Algorithm mappings
ALGORITHM_DISPLAY_NAMES = {
    'GA': 'Genetic Algorithm',
    'Genetic Algorithm': 'Genetic Algorithm',
    'Full-Factorial': 'Full-Factorial',
    'Deep RL': 'Deep RL',
    'Reinforcement Learning': 'Deep RL',
}

ALGORITHM_DB_CODES = {
    'Genetic Algorithm': 'GA',
    'GA': 'GA',
    'Full-Factorial': 'FF',
    'Deep RL': 'DRL',
    'Reinforcement Learning': 'DRL',
}