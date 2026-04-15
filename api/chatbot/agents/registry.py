"""
Agent registry for managing and discovering chatbot agents.
"""
from typing import Dict, Type, Optional, List
from .base import BaseAgent, AgentResult


class AgentRegistry:
    """
    Registry for chatbot agents.
    Provides factory methods for creating and discovering agents.
    """
    
    _agents: Dict[str, Type[BaseAgent]] = {}
    
    @classmethod
    def register(cls, agent_class: Type[BaseAgent]) -> Type[BaseAgent]:
        """
        Register an agent class. Can be used as a decorator.
        
        Example:
            @AgentRegistry.register
            class MyAgent(BaseAgent):
                ...
        """
        # Create a temporary instance to get the name
        temp_instance = agent_class.__new__(agent_class)
        temp_instance.__init__()
        agent_name = temp_instance.name
        cls._agents[agent_name] = agent_class
        return agent_class
    
    @classmethod
    def get(cls, name: str, evaluator: str = 'cascade', run_id: str = None) -> Optional[BaseAgent]:
        """Get an agent instance by name."""
        agent_class = cls._agents.get(name)
        if agent_class:
            return agent_class(evaluator=evaluator, run_id=run_id)
        return None
    
    @classmethod
    def get_all(cls, evaluator: str = 'cascade', run_id: str = None) -> List[BaseAgent]:
        """Get instances of all registered agents."""
        return [
            agent_class(evaluator=evaluator, run_id=run_id)
            for agent_class in cls._agents.values()
        ]
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List all registered agent names."""
        return list(cls._agents.keys())
    
    @classmethod
    def find_agent_for_query(cls, query: str, evaluator: str = 'cascade', run_id: str = None) -> Optional[BaseAgent]:
        """Find the first agent that can handle the given query."""
        for agent in cls.get_all(evaluator, run_id):
            if agent.can_handle(query):
                return agent
        return None