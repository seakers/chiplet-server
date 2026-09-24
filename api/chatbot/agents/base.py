"""
Base agent interface for ChatBot sub-agents.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import time
from functools import wraps


def retry_on_failure(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    if result.success:
                        return result
                    if attempt < max_retries - 1:
                        time.sleep(backoff_factor ** attempt)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(backoff_factor ** attempt)
            return result
        return wrapper
    return decorator


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'message': self.message,
            'data': self.data,
            'error': self.error,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all ChatBot sub-agents.
    Each agent handles a specific type of analysis or operation.
    """
    
    def __init__(self, evaluator: str = 'cascade', run_id: str = None):
        self.evaluator = evaluator.lower()
        self.run_id = run_id
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifier for this agent."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Description of what this agent does."""
        pass
    
    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute the agent's main functionality.
        
        Args:
            context: Dictionary containing relevant context data.
            
        Returns:
            AgentResult with success status and message.
        """
        pass
    
    def can_handle(self, query: str) -> bool:
        """
        Check if this agent can handle the given query.
        Default implementation checks for agent name in query.
        """
        return self.name.lower() in query.lower()
    
    def safe_execute(self, context: Dict[str, Any]) -> AgentResult:
        """Execute with automatic retry and exponential backoff."""
        return retry_on_failure(max_retries=3)(self.execute)(context)
    
    def get_parameters_schema(self) -> dict:
        """Override in subclasses to expose typed parameters to the LLM."""
        return {"type": "object", "properties": {}, "required": []}


class AgentContext:
    """
    Shared context passed between agents and the main chatbot.
    """
    
    def __init__(self, evaluator: str = 'cascade', run_id: str = None):
        self.evaluator = evaluator
        self.run_id = run_id
        self.points_data: List[Dict[str, Any]] = []
        self.pareto_front: List[Dict[str, Any]] = []
        self.full_data: List[Dict[str, Any]] = []
        self.point_context: Any = None
        self.point_in_active_context: bool = False
        self.correlations: Dict[str, float] = {}
        self.rules: List[Dict[str, Any]] = []
    
    def load_points_data(self):
        """Load points data from CSV file based on evaluator."""
        from api.data.loaders import PointsLoader
        loader = PointsLoader(self.evaluator, self.run_id)
        self.points_data = loader.load_points_as_dicts()
        return self.points_data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'evaluator': self.evaluator,
            'run_id': self.run_id,
            'points_count': len(self.points_data),
            'pareto_count': len(self.pareto_front),
            'has_active_context': self.point_in_active_context,
        }