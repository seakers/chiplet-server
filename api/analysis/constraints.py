"""
Constraint matching utilities for chiplet design highlighting.
Consolidates the constraint mapping logic from views.py [1].
"""
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass

from api.config.evaluators import get_evaluator_config


@dataclass
class ConstraintLevel:
    """Defines a constraint level with its matching function."""
    name: str
    description: str
    matcher: Callable[[Dict[str, Any]], bool]


class ConstraintMatcher:
    """
    Matches design points against constraint criteria.
    Consolidates the constraint_mapping logic from views.py [1].
    """
    
    def __init__(self, evaluator: str = 'cascade'):
        self.evaluator = evaluator.lower()
        self.config = get_evaluator_config(evaluator)
        self._constraint_mapping = self._build_constraint_mapping()
    
    def _build_constraint_mapping(self) -> Dict[str, Dict[str, ConstraintLevel]]:
        """
        Build constraint mapping based on evaluator type.
        
        This consolidates the hardcoded constraint_mapping from views.py [1].
        """
        if self.evaluator == 'cascade':
            return {
                'gpu': {
                    'none': ConstraintLevel('none', 'No GPU chiplets', lambda p: p.get('gpu', 0) == 0),
                    'low': ConstraintLevel('low', '1-3 GPU chiplets', lambda p: 1 <= p.get('gpu', 0) <= 3),
                    'medium': ConstraintLevel('medium', '4-6 GPU chiplets', lambda p: 4 <= p.get('gpu', 0) <= 6),
                    'high': ConstraintLevel('high', '7+ GPU chiplets', lambda p: p.get('gpu', 0) >= 7),
                },
                'attention': {
                    'none': ConstraintLevel('none', 'No Attention chiplets', lambda p: p.get('attn', 0) == 0),
                    'low': ConstraintLevel('low', '1-2 Attention chiplets', lambda p: 1 <= p.get('attn', 0) <= 2),
                    'medium': ConstraintLevel('medium', '3-5 Attention chiplets', lambda p: 3 <= p.get('attn', 0) <= 5),
                    'high': ConstraintLevel('high', '6+ Attention chiplets', lambda p: p.get('attn', 0) >= 6),
                },
                'sparse': {
                    'none': ConstraintLevel('none', 'No Sparse chiplets', lambda p: p.get('sparse', 0) == 0),
                    'low': ConstraintLevel('low', '1-3 Sparse chiplets', lambda p: 1 <= p.get('sparse', 0) <= 3),
                    'medium': ConstraintLevel('medium', '4-7 Sparse chiplets', lambda p: 4 <= p.get('sparse', 0) <= 7),
                    'high': ConstraintLevel('high', '8+ Sparse chiplets', lambda p: p.get('sparse', 0) >= 8),
                },
                'convolution': {
                    'none': ConstraintLevel('none', 'No Convolution chiplets', lambda p: p.get('conv', 0) == 0),
                    'low': ConstraintLevel('low', '1-2 Convolution chiplets', lambda p: 1 <= p.get('conv', 0) <= 2),
                    'medium': ConstraintLevel('medium', '3-4 Convolution chiplets', lambda p: 3 <= p.get('conv', 0) <= 4),
                    'high': ConstraintLevel('high', '5+ Convolution chiplets', lambda p: p.get('conv', 0) >= 5),
                },
            }
        elif self.evaluator == 'pistil':
            # PISTIL uses relative thresholds based on data range
            return self._build_relative_constraints()
        else:
            return {}
    
    def _build_relative_constraints(self) -> Dict[str, Dict[str, ConstraintLevel]]:
        """Build relative constraints for PISTIL evaluator."""
        # These will be populated dynamically based on data range
        return {}
    
    def get_matching_designs(self, 
                             points: List[Dict[str, Any]], 
                             chiplet_type: str, 
                             constraint_level: str) -> List[Dict[str, Any]]:
        """
        Find designs matching the specified constraint.
        
        Args:
            points: List of design point dictionaries.
            chiplet_type: Type of chiplet to filter on (e.g., 'gpu', 'attention').
            constraint_level: Level of constraint ('none', 'low', 'medium', 'high').
            
        Returns:
            List of matching design points.
        """
        chiplet_type = chiplet_type.lower()
        constraint_level = constraint_level.lower()
        
        if chiplet_type not in self._constraint_mapping:
            raise ValueError(f"Unknown chiplet type: {chiplet_type}. "
                           f"Available: {list(self._constraint_mapping.keys())}")
        
        if constraint_level not in self._constraint_mapping[chiplet_type]:
            raise ValueError(f"Unknown constraint level: {constraint_level}. "
                           f"Available: {list(self._constraint_mapping[chiplet_type].keys())}")
        
        constraint = self._constraint_mapping[chiplet_type][constraint_level]
        return [p for p in points if constraint.matcher(p)]
    
    def get_highlighted_points(self,
                               points: List[Dict[str, Any]],
                               chiplet_type: str,
                               constraint_level: str) -> List[Dict[str, Any]]:
        """
        Return all points with highlighting information.
        
        This is used by the frontend for visualization [1].
        """
        matching_designs = self.get_matching_designs(points, chiplet_type, constraint_level)
        
        highlighted_points = []
        for i, point in enumerate(points):
            is_matching = any(
                point.get('x') == match.get('x') and point.get('y') == match.get('y')
                for match in matching_designs
            )
            highlighted_point = {
                'index': i,
                'highlighted': is_matching,
                **point
            }
            highlighted_points.append(highlighted_point)
        
        return highlighted_points
    
    def parse_constraint_from_text(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parse chiplet type and constraint level from natural language text.
        
        This consolidates the parsing logic from views.py [1].
        """
        text_lower = text.lower()
        
        # Detect chiplet type
        chiplet_type = None
        chiplet_keywords = {
            'gpu': ['gpu'],
            'attention': ['attention', 'attn'],
            'sparse': ['sparse'],
            'convolution': ['convolution', 'conv'],
        }
        
        for chip_type, keywords in chiplet_keywords.items():
            if any(kw in text_lower for kw in keywords):
                chiplet_type = chip_type
                break
        
        # Detect constraint level
        constraint_level = None
        level_keywords = {
            'none': ['none', 'zero', 'no '],
            'low': ['low', 'few', 'small'],
            'medium': ['medium', 'moderate', 'some'],
            'high': ['high', 'many', 'lots'],
        }
        
        for level, keywords in level_keywords.items():
            if any(kw in text_lower for kw in keywords):
                constraint_level = level
                break
        
        if chiplet_type and constraint_level:
            return {'chiplet_type': chiplet_type, 'constraint_level': constraint_level}
        return None