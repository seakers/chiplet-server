from .registry import AgentRegistry

class QueryPreprocessor:
    """Validate and enhance user queries before processing."""
    
    CHIPLET_KEYWORDS = {
        'energy', 'runtime', 'flops', 'memory', 'pareto', 
        'optimization', 'correlation', 'chiplet', 'kernel'
    }
    
    def preprocess(self, query: str) -> dict:
        return {
            "original": query,
            "cleaned": query.strip(),
            "detected_intent": self._detect_intent(query),
            "relevant_agents": self._suggest_agents(query),
            "requires_context": self._requires_point_context(query)
        }
    
    def _detect_intent(self, query: str) -> str:
        query_lower = query.lower()
        if any(w in query_lower for w in ['optimize', 'run', 'start']):
            return 'optimization'
        if any(w in query_lower for w in ['correlat', 'affect', 'impact']):
            return 'analysis'
        if any(w in query_lower for w in ['show', 'what', 'which', 'find']):
            return 'retrieval'
        return 'general'
    
    def _suggest_agents(self, query: str) -> list:
        """Use the registry's can_handle logic instead of hardcoded keywords."""
        return [
            agent.name 
            for agent in AgentRegistry.get_all()
            if agent.can_handle(query)
        ]
    
    def _requires_point_context(self, query: str) -> bool:
        context_keywords = ['energy', 'runtime', 'chiplet', 'kernel', 'this design']
        return any(kw in query.lower() for kw in context_keywords)