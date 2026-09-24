"""
Data models for chiplet design optimization.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class DesignPoint:
    objectives: Dict[str, float]      # NEW: e.g., {'energy_mj': 5.2, 'latency_ms': 12}
    chiplets: Dict[str, int]
    pareto_rank: Optional[int] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    context_file_path: str = ""
    algorithm: str = ""
    trace: str = ""

    # Backward-compat properties
    @property
    def execution_time_ms(self) -> float:
        return self.objectives.get('exe_time_ms', self.objectives.get('latency_per_token_ms', 0))

    @property
    def energy_mj(self) -> float:
        return self.objectives.get('energy_mj', self.objectives.get('energy_per_inference_mJ', 0))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'x': self.execution_time_ms,
            'y': self.energy_mj,
            'execution_time_ms': self.execution_time_ms,
            'energy_mj': self.energy_mj,
            'gpu': self.chiplets.get('GPU', 0),
            'attn': self.chiplets.get('Attention', 0),
            'sparse': self.chiplets.get('Sparse', 0),
            'conv': self.chiplets.get('Convolution', 0),
            'chiplets': self.chiplets,
            'pareto_rank': self.pareto_rank,
            'algorithm': self.algorithm,
            'trace': self.trace,
            **self.additional_metrics
        }
    
    def to_frontend_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by frontend."""
        return {
            'x': self.execution_time_ms,
            'y': self.energy_mj,
            'gpu': self.chiplets.get('GPU', 0),
            'attn': self.chiplets.get('Attention', 0),
            'sparse': self.chiplets.get('Sparse', 0),
            'conv': self.chiplets.get('Convolution', 0),
            'algorithm': self.algorithm,
            'trace': self.trace,
        }
    
    @classmethod
    def from_csv_row(cls, row: List[str], evaluator: str = 'cascade', 
                     algorithm: str = '', trace: str = '') -> 'DesignPoint':
        """Create a DesignPoint from a CSV row."""
        if evaluator.lower() == 'cascade':
            # CASCADE format: exe_time, energy, GPU, Attention, Sparse, Convolution
            return cls(
                execution_time_ms=float(row[0]),
                energy_mj=float(row[1]),
                chiplets={
                    'GPU': int(float(row[2])),
                    'Attention': int(float(row[3])),
                    'Sparse': int(float(row[4])),
                    'Convolution': int(float(row[5])),
                },
                algorithm=algorithm,
                trace=trace,
            )
        elif evaluator.lower() == 'pistil':
            # PISTIL format: decisions first, then objectives
            return cls(
                execution_time_ms=float(row[9]),  # latency_ms
                energy_mj=float(row[10]),  # energy_mJ
                chiplets={
                    'num_cus': int(float(row[0])),
                    'num_tmacs': int(float(row[1])),
                    'mem_buf_cap': int(float(row[2])),
                    'net_buf_cap': int(float(row[3])),
                    'mem_banks_per_group': int(float(row[4])),
                    'mem_ranks': int(float(row[5])),
                    'mem_frac_bank_cap': float(row[6]),
                    'batch_size': int(float(row[7])),
                    'kv_cache': int(float(row[8])),
                },
                algorithm=algorithm,
                trace=trace,
            )
        else:
            raise ValueError(f"Unknown evaluator: {evaluator}")


@dataclass
class ParetoFront:
    """Represents a Pareto front of design points."""
    points: List[DesignPoint]
    rank: int = 0
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __iter__(self):
        return iter(self.points)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dictionaries for JSON serialization."""
        return [p.to_dict() for p in self.points]


@dataclass
class OptimizationResult:
    """Results from an optimization run."""
    run_id: str
    algorithm: str
    model: str
    trace_name: str
    objectives: List[str]
    population_size: int
    generations: int
    design_points: List[DesignPoint]
    pareto_front: Optional[ParetoFront] = None
    execution_time_seconds: float = 0.0
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'run_id': self.run_id,
            'algorithm': self.algorithm,
            'model': self.model,
            'trace_name': self.trace_name,
            'objectives': self.objectives,
            'population_size': self.population_size,
            'generations': self.generations,
            'design_points': [p.to_dict() for p in self.design_points],
            'pareto_front': self.pareto_front.to_list() if self.pareto_front else [],
            'execution_time_seconds': self.execution_time_seconds,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class RunStatistics:
    """Statistical summary of an optimization run."""
    total_points: int
    pareto_count: int
    avg_time: float
    avg_energy: float
    min_time: float
    min_energy: float
    max_time: float
    max_energy: float
    std_time: float
    std_energy: float
    
    @classmethod
    def from_points(cls, points: List[Dict[str, Any]]) -> 'RunStatistics':
        """Calculate statistics from a list of points."""
        if not points:
            return cls(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        times = [p.get('x', p.get('execution_time_ms', 0)) for p in points]
        energies = [p.get('y', p.get('energy_mj', 0)) for p in points]
        
        import numpy as np
        return cls(
            total_points=len(points),
            pareto_count=0,  # Set separately after Pareto calculation
            avg_time=float(np.mean(times)),
            avg_energy=float(np.mean(energies)),
            min_time=float(np.min(times)),
            min_energy=float(np.min(energies)),
            max_time=float(np.max(times)),
            max_energy=float(np.max(energies)),
            std_time=float(np.std(times)),
            std_energy=float(np.std(energies)),
        )