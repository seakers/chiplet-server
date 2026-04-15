"""
Point Info Agent for retrieving information about specific design points.
"""
import numpy as np
import re
from typing import Dict, Any, List, Optional

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry


@AgentRegistry.register
class PointInfoAgent(BaseAgent):
    """
    Agent that provides information about specific chiplet design points.
    Consolidates the retrieve_point_parser logic from model.py [2].
    """
    
    PARAM_INDICES = {
        "flops": 0,
        "mem_accessed": 1,
        "exe_time": 2,
        "energy": 3
    }
    
    @property
    def name(self) -> str:
        return "point_info_agent"
    
    @property
    def description(self) -> str:
        return "Provides information about specific chiplet design points."
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute point information retrieval.
        
        Args:
            context: Dictionary containing query and point context data.
            
        Returns:
            AgentResult with point information.
        """
        try:
            query = context.get('query', '')
            point_context = context.get('point_context')
            full_data = context.get('full_data', [])
            
            if point_context is None or len(full_data) == 0:
                return AgentResult(
                    success=False,
                    message="No design point context available. Please select a design point first.",
                    error="No point context"
                )
            
            # Parse the query to determine what data to retrieve
            parsed = self._parse_data_request(query)
            
            if parsed is None:
                return AgentResult(
                    success=False,
                    message="Could not understand the data request. Please specify parameter (flops, mem_accessed, exe_time, energy), min/max, and number of points.",
                    error="Parse error"
                )
            
            # Retrieve the requested data
            result = self._retrieve_point_data(
                point_context,
                full_data,
                parsed['min_max'],
                parsed['param'],
                parsed['num_points']
            )
            
            return AgentResult(
                success=True,
                message=result,
                data=parsed
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error retrieving point info: {str(e)}",
                error=str(e)
            )
    
    def _parse_data_request(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Parse a data request from query string.
        Consolidates the retrieve_point_parser logic from model.py [2].
        """
        query_lower = query.lower()
        
        # Determine min/max
        min_max = 'max'  # Default to max
        if 'min' in query_lower or 'lowest' in query_lower or 'smallest' in query_lower:
            min_max = 'min'
        elif 'max' in query_lower or 'highest' in query_lower or 'largest' in query_lower or 'top' in query_lower:
            min_max = 'max'
        
        # Determine parameter
        param = None
        if 'energy' in query_lower or 'power' in query_lower:
            param = 'energy'
        elif 'time' in query_lower or 'runtime' in query_lower or 'execution' in query_lower or 'exe_time' in query_lower:
            param = 'exe_time'
        elif 'flop' in query_lower or 'compute' in query_lower:
            param = 'flops'
        elif 'memory' in query_lower or 'mem' in query_lower:
            param = 'mem_accessed'
        
        if param is None:
            return None
        
        # Determine number of points
        num_points = 5  # Default
        num_match = re.search(r'(\d+)', query)
        if num_match:
            num_points = int(num_match.group(1))
        
        return {
            'min_max': min_max,
            'param': param,
            'num_points': num_points
        }
    
    def _retrieve_point_data(self, 
                              point_context: np.ndarray, 
                              full_data: List[Dict[str, Any]],
                              min_max: str, 
                              param: str, 
                              num_points: int) -> str:
        """
        Retrieve point data based on parameters.
        Consolidates the retrieve_point_data logic from model.py [2].
        """
        param_ind = self.PARAM_INDICES.get(param)
        if param_ind is None:
            return f"Unknown parameter: {param}"
        
        # Get values from the total column (last row of each kernel)
        vals = point_context[:, -1, param_ind]
        
        # Get indices of top/bottom points
        if min_max == 'min':
            ind = np.argpartition(vals, min(num_points, len(vals) - 1))[:num_points]
        elif min_max == 'max':
            ind = np.argpartition(vals, -min(num_points, len(vals)))[-num_points:]
        else:
            return f"Invalid min_max value: {min_max}"
        
        # Find which chiplet has the max/min value for each kernel
        chiplet = np.argmax(point_context[ind, :-1, param_ind], axis=1)
        
        # Build response string
        totals_string = ""
        for i in range(len(ind)):
            kernel_idx = ind[i]
            chiplet_idx = chiplet[i]
            
            kernel_data = full_data[kernel_idx]
            kernel_name = kernel_data.get('name', f'Kernel {kernel_idx}')
            kernel_number = kernel_data.get('kernal_number', kernel_idx)
            
            chiplet_data = kernel_data.get('chiplets', {}).get(str(chiplet_idx), {})
            chiplet_name = chiplet_data.get('name', 'unknown')
            chiplet_value = chiplet_data.get(param, 0)
            
            totals_string += (
                f"Kernel number {kernel_number}, a {kernel_name} type kernel, "
                f"has a total {param} of {vals[kernel_idx]:.4f}. "
                f"The {min_max} {param} chiplet is chiplet number {chiplet_idx} "
                f"which is a {chiplet_name} type chiplet and has a value of {chiplet_value:.4f}.\n"
            )
        
        message = f"The {num_points} {min_max} {param} chiplets are:\n{totals_string}"
        return message
    
    def can_handle(self, query: str) -> bool:
        """Check if query is about specific point information."""
        keywords = ['chiplet', 'kernel', 'point', 'design', 'flops', 'memory', 'specific']
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)