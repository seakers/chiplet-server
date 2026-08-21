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
        return (
            "Starts optimization runs. REQUIRED parameters: model (CASCADE or PISTIL), "
            "algorithm (Genetic Algorithm, Full-Factorial, or Deep RL), and at least one trace/model name. "
            "If the user has NOT provided all required parameters, DO NOT call this tool. "
            "Instead, respond with a message asking the user for the missing information. "
            "For CASCADE: need trace name(s) and weights. "
            "For PISTIL: need pistil model name (e.g., llama3-8b). "
            "Optional: population_size (default 50), generations (default 100)."
        )
    
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
            
            # Prefer structured params passed by the LLM tool call; fall back to text parsing.
            structured_keys = ('model', 'algorithm', 'pistil_model', 'traces',
                               'objectives', 'population_size', 'generations')
            has_structured = any(context.get(k) not in (None, [], "") for k in structured_keys)

            if has_structured:
                params = {
                    "model":           (context.get("model") or "CASCADE").upper(),
                    "algorithm":       context.get("algorithm") or "Genetic Algorithm",
                    "population_size": int(context.get("population_size", 50)),
                    "generations":     int(context.get("generations", 100)),
                    "objectives":      context.get("objectives") or ["Energy", "Runtime"],
                    "traces":          context.get("traces") or [{"name": "gpt-j-65536-weighted", "weight": 1.0}],
                }
                if context.get("pistil_model"):
                    params["pistil_model"] = context.get("pistil_model")
            else:
                # Legacy path: parse from raw query text
                query = context.get("query", "")
                params = self._parse_optimization_request(query)

            if params.get('status') == 'error':
                return AgentResult(
                    success=False,
                    message=params.get('message', 'Failed to parse optimization request'),
                    error=params.get('message')
                )
            
            # Start the optimization run
            result = self._start_optimization(params)
            run_id = result.get("run_id")

            return AgentResult(
                success=result.get('status') == 'success',
                message=result.get('message', 'Optimization started'),
                data={
                    'run_id': run_id,
                    'model':  params.get('model'),
                    'algorithm': params.get('algorithm'),
                    'payload': result.get('payload'),
                }
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
        
        # Parse objectives — use canonical friendly names that match name_to_index
        if "energy" in query_lower:
            result["objectives"].append("Energy")
        if "time" in query_lower or "runtime" in query_lower or "latency" in query_lower:
            result["objectives"].append("Runtime")
        
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
            result["objectives"] = ["Energy", "Runtime"]  # Default
        if not result["traces"]:
            result["traces"] = [{"name": "gpt-j-65536-weighted"}]  # Default
        
        return result
    
    def _start_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start an optimization run using the same backend path as the UI."""
        from rest_framework.test import APIRequestFactory
        from api.views.optimization import run_optimization  # the view used by /api/run-optimization/

        # Build a payload matching what ProblemFormulation.vue sends [8]
        payload = {
            "model":           params.get("model", "CASCADE"),
            "algorithm":       params.get("algorithm", "Genetic Algorithm"),
            "objectives":      params.get("objectives") or ["Energy", "Runtime"],
            "traces":          params.get("traces") or [{"name": "gpt-j-65536-weighted", "weight": 1.0}],
            "population_size": params.get("population_size", 50),
            "generations":     params.get("generations", 100),
        }
        if payload["model"] == "PISTIL":
            payload["pistil_model"] = params.get("pistil_model", "llama3-8b")

        factory = APIRequestFactory()
        drf_req = factory.post('/api/run-optimization/', payload, format='json')
        response = run_optimization(drf_req)

        # Robustly read the body for both DRF Response and Django JsonResponse
        if hasattr(response, 'data'):
            body = response.data
        else:
            import json as _json
            try:
                body = _json.loads(response.content.decode('utf-8'))
            except Exception:
                body = {}

        run_id = (
            body.get('pistil_run_id')
            or body.get('run_directory')
            or body.get('run_id')
            or body.get('deep_rl_run_id')
        )

        return {
            "status": "success" if run_id else "error",
            "message": (
                f"Started {payload['algorithm']} on {payload['model']} "
                f"(run_id={run_id}). The plot will populate as points arrive."
                if run_id else
                f"Failed to start optimization: {body.get('error') or body.get('message')}"
            ),
            "run_id": run_id,
            "payload": payload,
            "response": body,
        }
    
    def can_handle(self, query: str) -> bool:
        """Check if query is about running optimization."""
        keywords = ['optimize', 'optimization', 'run ga', 'start', 'genetic algorithm', 'full-factorial']
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)
    

    def get_parameters_schema(self) -> dict:
        """Expose structured optimization parameters to the LLM."""
        return {
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "enum": ["CASCADE", "PISTIL", "HISIM"],
                    "description": "Which evaluator to run the optimization on."
                },
                "algorithm": {
                    "type": "string",
                    "enum": ["Genetic Algorithm", "Full-Factorial", "Deep RL"],
                    "description": "Optimization algorithm to use."
                },
                "pistil_model": {
                    "type": "string",
                    "description": "For PISTIL runs, the model/workload name (e.g. 'llama3-8b')."
                },
                "traces": {
                    "type": "array",
                    "description": "For CASCADE runs, list of trace names with optional weights.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name":   {"type": "string"},
                            "weight": {"type": "number"}
                        },
                        "required": ["name"]
                    }
                },
                "objectives": {
                    "type": "array",
                    "description": "Objectives to optimize (friendly names, e.g. 'Energy', 'Runtime').",
                    "items": {"type": "string"}
                },
                "population_size": {
                    "type": "integer",
                    "description": "GA population size (default 50)."
                },
                "generations": {
                    "type": "integer",
                    "description": "Number of GA generations (default 100)."
                }
            },
            "required": ["model", "algorithm"]
        }