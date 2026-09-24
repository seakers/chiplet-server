from dataclasses import dataclass, field
from datetime import datetime

from api.chatbot.agents.base import AgentResult

@dataclass
class ConversationMemory:
    """Structured memory for the chatbot."""
    short_term: list = field(default_factory=list)   # Recent messages
    analysis_cache: dict = field(default_factory=dict) # Agent results cache
    session_insights: list = field(default_factory=list) # Key findings
    max_short_term: int = 20
    
    def add_message(self, role: str, content: str):
        self.short_term.append({
            "role": role, 
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        # Trim to prevent context overflow
        if len(self.short_term) > self.max_short_term:
            self.short_term = self.short_term[-self.max_short_term:]
    
    def cache_agent_result(self, agent_name: str, result: AgentResult):
        """Cache agent results to avoid redundant calls."""
        self.analysis_cache[agent_name] = {
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_cached_result(self, agent_name: str, max_age_seconds: int = 300):
        """Get cached result if still fresh."""
        cached = self.analysis_cache.get(agent_name)
        if cached:
            age = (datetime.now() - datetime.fromisoformat(cached["timestamp"])).seconds
            if age < max_age_seconds:
                return cached["result"]
        return None
    

class AgentPipeline:
    def __init__(self, agents: list):
        self.agents = agents

    def execute(self, initial_context: dict) -> list:
        context = initial_context.copy()
        results = []
        for agent in self.agents:
            result = agent.safe_execute(context)
            results.append(result)
            if result.success and result.data:
                context.update(result.data)
            elif not self._should_continue_on_failure(agent, result):
                break
        return results

    def _should_continue_on_failure(self, agent, result) -> bool:
        return False