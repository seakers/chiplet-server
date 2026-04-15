"""
Optimization Agent for handling optimization run requests.
"""
import re
import threading
from typing import Dict, Any, Optional

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry
from api.config.prompts import SystemPrompts


@AgentRegistry.register
class OptimizationAgent(BaseAgent):
    """
    Agent that handles optimization run requests.
    Consolidates the optimization_manager logic from model.py [2].
    """
    
    @property
    def name(self) -> str:
        return "optimization_agent"
    
    @property
    def description(self) -> str:
        return "Starts optimization runs based on user specifications."
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute optimization based on parsed parameters.
        
        Args:
            context: Dictionary containing optimization parameters or raw query.
            
        Returns:
            AgentResult with optimization status.
        """
        try:
            # Get the raw query if provided
            query = context.get('query', '')
            
            # Parse optimization parameters
            params = self._parse_optimization_request(query)
            
            if params.get('status') == 'error':
                return AgentResult(
                    success=False,
                    message=params.get('message', 'Failed to parse optimization request'),
                    error=params.get('message')
                )
            
            # Start the optimization run
            result = self._start_optimization(params)
            
            return AgentResult(
                success=result.get('status') == 'success',
                message=result.get('message', 'Optimization started'),
                data=result
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error starting optimization: {str(e)}",
                error=str(e)
            )
    
    def _parse_optimization_request(self, query: str) -> Dict[str, Any]:
        """
        Parse optimization parameters from a query string.
        Consolidates parsing logic from model.py [2].
        """
        result = {
            "model": None,
            "algorithm": None,
            "population_size": 50,
            "generations": 100,
            "objectives": [],
            "traces": []
        }
        
        query_lower = query.lower()
        
        # Parse model
        if "cascade" in query_lower:
            result["model"] = "CASCADE"
        elif "pistil" in query_lower:
            result["model"] = "PISTIL"
        elif "hisim" in query_lower:
            result["model"] = "HISIM"
        
        # Parse algorithm
        if "genetic" in query_lower or " ga " in query_lower:
            result["algorithm"] = "Genetic Algorithm"
        elif "full-factorial" in query_lower or "full factorial" in query_lower:
            result["algorithm"] = "Full-Factorial"
        elif "reinforcement" in query_lower or "deep rl" in query_lower or " rl " in query_lower:
            result["algorithm"] = "Deep RL"
        
        # Parse population size
        pop_match = re.search(r'population[:\s]*(\d+)', query_lower)
        if pop_match:
            result["population_size"] = int(pop_match.group(1))
        
        # Parse generations
        gen_match = re.search(r'generation[s]?[:\s]*(\d+)', query_lower)
        if gen_match:
            result["generations"] = int(gen_match.group(1))
        
        # Parse objectives
        if "energy" in query_lower:
            result["objectives"].append("energy")
        if "time" in query_lower or "runtime" in query_lower or "latency" in query_lower:
            result["objectives"].append("time")
        
        # Parse traces
        trace_matches = re.findall(r'gpt-[a-z0-9-]+', query_lower)
        for trace in trace_matches:
            result["traces"].append({"name": trace})
        
        # Validation
        if not result["model"]:
            result["model"] = "CASCADE"  # Default
        if not result["algorithm"]:
            result["algorithm"] = "Genetic Algorithm"  # Default
        if not result["objectives"]:
            result["objectives"] = ["energy", "time"]  # Default
        if not result["traces"]:
            result["traces"] = [{"name": "gpt-j-65536-weighted"}]  # Default
        
        return result
    
    def _start_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start the optimization run based on parameters."""
        from api.Evaluator.gaCascade import runGACascade
        from api.Evaluator.generator import generate_weighted_trace
        
        # Determine trace
        traces = params.get("traces", [])
        if len(traces) == 1:
            trace_name = traces[0].get("name", "gpt-j-65536-weighted")
        else:
            trace_names = [t.get("name") for t in traces]
            weights = [t.get("weight", 1.0) for t in traces]
            trace_name = generate_weighted_trace(trace_names, weights, label='optimization')
        
        algorithm = params.get("algorithm", "Genetic Algorithm")
        
        if algorithm == "Genetic Algorithm":
            # Start GA in background thread
            def run_ga():
                try:
                    runGACascade(
                        pop_size=params.get("population_size", 50),
                        n_gen=params.get("generations", 100),
                        trace=trace_name
                    )
                except Exception as e:
                    print(f"GA optimization error: {e}")
            
            thread = threading.Thread(target=run_ga, daemon=True)
            thread.start()
            
            return {
                "status": "success",
                "message": f"Optimization started using {algorithm} on trace '{trace_name}'",
                "params": params
            }
        
        elif algorithm == "Full-Factorial":
            return {
                "status": "success",
                "message": f"Full-Factorial optimization would be started on trace '{trace_name}'",
                "params": params
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown algorithm: {algorithm}"
            }
    
    def can_handle(self, query: str) -> bool:
        """Check if query is about running optimization."""
        keywords = ['optimize', 'optimization', 'run ga', 'start', 'genetic algorithm', 'full-factorial']
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)