"""
Pareto front calculation utilities.
Consolidates repeated Pareto logic from views.py and model.py [1][2].
"""
import numpy as np
from typing import List, Dict, Any, Tuple


class ParetoCalculator:
    """
    Calculates Pareto fronts and rankings for multi-objective optimization.
    """
    
    @staticmethod
    def is_dominated(point1: Dict[str, Any], point2: Dict[str, Any]) -> bool:
        """
        Check if point1 dominates point2 (all objectives minimized).
        Supports 2 or 3 objectives.
        
        Returns True if point1 dominates point2.
        """
        x1, y1 = point1.get('x', 0), point1.get('y', 0)
        x2, y2 = point2.get('x', 0), point2.get('y', 0)
        z1 = point1.get('z', None)
        z2 = point2.get('z', None)
        
        if z1 is not None and z2 is not None:
            # 3 objectives
            return (x1 <= x2 and y1 <= y2 and z1 <= z2) and (x1 < x2 or y1 < y2 or z1 < z2)
        else:
            # 2 objectives
            return (x1 <= x2 and y1 <= y2) and (x1 < x2 or y1 < y2)
    
    @staticmethod
    def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
        """
        Find the Pareto-efficient points.
        
        Args:
            costs: An (n_points, n_costs) array where lower is better.
            return_mask: If True, return a boolean mask. Otherwise return indices.
            
        Returns:
            Boolean mask or indices of Pareto-efficient points.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0
        
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
    
    @classmethod
    def get_pareto_front(cls, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find Pareto optimal points from a list of point dictionaries.
        
        This is the main method used by the frontend [1].
        """
        if not points:
            return []
        
        pareto_front = []
        for i, point in enumerate(points):
            is_dominated = False
            for j, other_point in enumerate(points):
                if i != j and cls.is_dominated(other_point, point):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(point)
        
        return pareto_front
    
    @classmethod
    def get_pareto_front_numpy(cls, objectives: np.ndarray) -> np.ndarray:
        """
        Find Pareto optimal points from a numpy array of objectives.
        
        Args:
            objectives: (n_points, n_objectives) array.
            
        Returns:
            Boolean mask of Pareto optimal points.
        """
        return cls.is_pareto_efficient(objectives, return_mask=True)
    
    @classmethod
    def calculate_pareto_ranks(cls, points: np.ndarray, max_ranks: int = None) -> np.ndarray:
        """
        Calculate Pareto ranks for all points.
        
        Rank 0 = Pareto front, Rank 1 = second front, etc.
        This consolidates the ranking logic from model.py [2].
        
        Args:
            points: (n_points, n_objectives) array of objective values.
            max_ranks: Maximum number of ranks to compute (None = all).
            
        Returns:
            Array of rank values for each point.
        """
        n_points = points.shape[0]
        ranks = np.full(n_points, -1, dtype=int)
        remaining_mask = np.ones(n_points, dtype=bool)
        
        current_rank = 0
        points_copy = points.copy()
        max_vals = np.max(points, axis=0) * 1.1 + 1e-6
        
        while np.any(remaining_mask):
            if max_ranks is not None and current_rank >= max_ranks:
                break
            
            # Find Pareto front of remaining points
            remaining_points = points_copy[remaining_mask]
            if len(remaining_points) == 0:
                break
                
            pareto_mask = cls.is_pareto_efficient(remaining_points, return_mask=True)
            
            # Map back to original indices
            remaining_indices = np.where(remaining_mask)[0]
            pareto_indices = remaining_indices[pareto_mask]
            
            # Assign ranks
            ranks[pareto_indices] = current_rank
            
            # Remove from consideration by setting to max values
            points_copy[pareto_indices] = max_vals
            remaining_mask[pareto_indices] = False
            
            current_rank += 1
        
        return ranks
    
    @staticmethod
    def count_dominated_points(pareto_a: List[Dict], pareto_b: List[Dict]) -> int:
        """
        Count how many points in pareto_b are dominated by points in pareto_a.
        Used for comparative analysis between runs [1].
        """
        dominated_count = 0
        for point_b in pareto_b:
            for point_a in pareto_a:
                x_a, y_a = point_a.get('x', 0), point_a.get('y', 0)
                x_b, y_b = point_b.get('x', 0), point_b.get('y', 0)
                
                # Check if point_a dominates point_b
                if (x_a <= x_b and y_a <= y_b) and (x_a < x_b or y_a < y_b):
                    dominated_count += 1
                    break
        return dominated_count