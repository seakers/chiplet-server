"""
Main ChatBot orchestrator.
Consolidates and slims down the ChatBotModel from model.py [2].
"""
import os
import json
import numpy as np
from typing import Dict, Any, List, Optional
from openai import OpenAI
import dotenv

from api.config.prompts import SystemPrompts, FollowUpSuggestions
from api.config.evaluators import get_evaluator_config
from api.data.loaders import PointsLoader
from .agents import (
    AgentRegistry, 
    BaseAgent, 
    AgentResult,
    DistanceCorrelationAgent,
    RuleMiningAgent,
    OptimizationAgent,
    EnergyAnalysisAgent,
    RuntimeAnalysisAgent,
    PointInfoAgent,
    EvaluationAgent,
    HighlightingAgent,
    ReportAgent,
    ComparativeAnalysisAgent
)
from .agents.memory import ConversationMemory
from .agents.preprocessor import QueryPreprocessor

dotenv.load_dotenv()


class ChatBot:
    """
    Main ChatBot class that orchestrates sub-agents for chiplet design assistance.
    
    This is a slimmed-down version of the original ChatBotModel [2],
    delegating specialized functionality to dedicated agents.
    """
    
    def __init__(self, 
                 evaluator: str = 'cascade', 
                 run_id: str = None,
                 model: str = "gpt-5.4-mini",
                 specs: List[Dict[str, str]] = None):
        """
        Initialize the ChatBot.
        
        Args:
            evaluator: Type of evaluator ('cascade' or 'pistil').
            run_id: Optional run ID for context.
            model: OpenAI model to use.
            specs: Optional custom system specs.
        """
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._model = model
        self.evaluator = evaluator.lower()
        self.run_id = run_id
        self.config = get_evaluator_config(evaluator)
        
        # Initialize message history with system specs
        self._specs = specs or self._get_default_specs()
        self.memory = ConversationMemory(max_short_term=20)
        self.messages = self._specs.copy()
        
        # Context tracking
        self.point_context: Optional[np.ndarray] = None
        self.point_in_active_context: bool = False
        self.full_data: List[Dict[str, Any]] = []
        
        # Initialize agents
        self._init_agents()
        self.preprocessor = QueryPreprocessor()
        self.last_agent_results = []
        self.active_objectives = []
        self.highlighted_indices = []
    
    def _get_default_specs(self) -> List[Dict[str, str]]:
        """Get default system specifications."""
        return [
            {
                "role": "developer",
                "content": SystemPrompts.MAIN_ASSISTANT
            }
        ]
    
    def _init_agents(self):
        """Initialize all sub-agents."""
        self.agents: Dict[str, BaseAgent] = {
            'dcorr_agent': DistanceCorrelationAgent(self.evaluator, self.run_id),
            'rule_mining_agent': RuleMiningAgent(self.evaluator, self.run_id),
            'optimization_agent': OptimizationAgent(self.evaluator, self.run_id),
            'energy_analysis_agent': EnergyAnalysisAgent(self.evaluator, self.run_id),
            'runtime_analysis_agent': RuntimeAnalysisAgent(self.evaluator, self.run_id),
            'point_info_agent': PointInfoAgent(self.evaluator, self.run_id),
            'evaluation_agent': EvaluationAgent(self.evaluator, self.run_id),
            'highlighting_agent': HighlightingAgent(self.evaluator, self.run_id),
            'report_agent': ReportAgent(self.evaluator, self.run_id),
            'comparative_analysis_agent': ComparativeAnalysisAgent(self.evaluator, self.run_id),
        }

    def _handle_tool_calls(self, message) -> str:
        """
        Handle OpenAI tool call responses by routing to the appropriate agent.
        
        Args:
            message: The OpenAI assistant message containing tool_calls.
            
        Returns:
            Final response string after all tool calls are resolved.
        """
        import json

        # Append the assistant message (with tool_calls) to history
        self.messages.append(message)

        # Build shared context for all agents
        agent_context = self._build_agent_context()

        # Agents that should NEVER be served from cache (results depend on query parameters)
        NO_CACHE_AGENTS = {
            'highlighting_agent',
            'evaluation_agent',
            'dcorr_agent',          # NEW — depends on selected_indices
            'rule_mining_agent',    # NEW — depends on selected_indices
            'optimization_agent',   # NEW — never makes sense to cache
            'report_agent',         # NEW — always generate fresh
            'comparative_analysis_agent',  # NEW
        }

        for tool_call in message.tool_calls:
            agent_name = tool_call.function.name

            # Skip cache for highlighting and other stateful agents
            if agent_name not in NO_CACHE_AGENTS:
                cached = self.memory.get_cached_result(agent_name)
                if cached:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": cached.message
                    })
                    continue
            
            # Safely parse arguments
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                arguments = {}

            # Merge tool-call arguments into agent context
            merged_context = {**agent_context, **arguments}

            # Route to the correct agent
            agent = self.agents.get(agent_name)
            if agent:
                try:
                    result = agent.safe_execute(merged_context)
                    if result.success:
                        tool_result = result.message
                    else:
                        tool_result = (
                            f"Agent '{agent_name}' failed: "
                            f"{result.error or result.message}"
                        )
                except Exception as e:
                    tool_result = f"Agent '{agent_name}' raised an exception: {str(e)}"
                    
                self.memory.cache_agent_result(agent_name, result)
                self.last_agent_results.append((agent_name, result))
            else:
                tool_result = f"Unknown agent: '{agent_name}'. Available agents: {list(self.agents.keys())}"

            # Append tool result in the format OpenAI expects
            self.messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result
            })

        # Get the final response from the model after all tool calls
        final_completion = self._client.chat.completions.create(
            model=self._model,
            messages=self.messages,
            tools=self._build_tools(),
            tool_choice="auto"   # Allow chained tool calls if needed
        )

        final_message = final_completion.choices[0].message

        # Handle recursive tool calls (e.g. model calls another tool)
        if final_message.tool_calls:
            return self._handle_tool_calls(final_message)

        # Append and return the final text response
        self.memory.add_message("assistant", final_message.content)
        self.messages.append({"role": "assistant", "content": final_message.content})
        return final_message.content


    def _build_agent_context(self) -> dict:
        """
        Build the shared context dictionary passed to all agents.
        Centralizes context construction used across tool calls.
        """
        return {
            'evaluator':      self.evaluator,
            'run_id':         self.run_id,
            'objectives':     getattr(self, 'active_objectives', None),
            'trace_or_model':   getattr(self, 'trace_or_model', None),
            'full_data':      self.full_data,
            'point_context':  self.point_context,
            'selected_indices': getattr(self, 'highlighted_indices', []),
        }
    

    def set_objectives(self, objectives: list):
        """Set the active objectives for this run (e.g. ['Runtime', 'Energy'])."""
        self.active_objectives = objectives
        print(f"[ChatBot] Active objectives set to: {objectives}")


    def _build_tools(self) -> list:
        """
        Convert registered agents into OpenAI tool definitions.
        Required by _handle_tool_calls for follow-up completions.
        """
        tools = []
        for agent_name, agent in self.agents.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": agent_name,
                    "description": agent.description,
                    "parameters": agent.get_parameters_schema()
                }
            })
        return tools
    
    def get_response(self, 
                     query: str, 
                     role: str = "user",
                     use_retrieval: bool = False,
                     filters: Dict = None,
                     top_k: int = 6) -> str:
        """
        Get a response to a user query.
        
        This method handles the main conversation flow, detecting and executing
        sub-agent calls as needed [2].
        
        Args:
            query: User's question or request.
            role: Message role ('user' or 'assistant').
            use_retrieval: Whether to use retrieval-augmented generation.
            filters: Optional filters for retrieval.
            top_k: Number of results for retrieval.
            
        Returns:
            Assistant's response string.
        """
        preprocessed = self.preprocessor.preprocess(query)
        query = preprocessed["cleaned"]

        # Add user message to history
        self.memory.add_message(role, query)
        self.messages.append({"role": role, "content": query})

        if use_retrieval:
            return self._handle_retrieval_response(query, role, filters, top_k)

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=self.messages,
            tools=self._build_tools(),
            tool_choice="auto"
        )

        message = completion.choices[0].message

        # Route to tool handler if tools were called
        if message.tool_calls:
            return self._handle_tool_calls(message)

        # Plain text response — no tools invoked
        self.memory.add_message("assistant", message.content)
        self.messages.append({"role": "assistant", "content": message.content})
        return message.content
    
    def _load_current_points(self) -> List[Dict[str, Any]]:
        """Load current points data."""
        try:
            loader = PointsLoader(self.evaluator, self.run_id)
            return loader.load_points_as_dicts()
        except Exception as e:
            print(f"Error loading points: {e}")
            return []
    
    def _handle_retrieval_response(self, 
                                    query: str, 
                                    role: str,
                                    filters: Dict,
                                    top_k: int) -> Dict[str, Any]:
        """Handle retrieval-augmented generation."""
        from api.retrieval import search
        
        # Retrieve relevant documents
        results = search(query, self.get_embedding, top_k=top_k, filters=filters)
        
        # Build context
        context_lines = []
        citations = []
        for r in results:
            ctag = f"C{r['rank']}"
            text = r.get("text", "")
            snippet = text if len(text) <= 300 else text[:297] + "..."
            context_lines.append(f"[{ctag}] {snippet}")
            citations.append({
                "tag": ctag,
                "score": r.get("score"),
                "file_path": r.get("metadata", {}).get("file_path"),
                "metadata": r.get("metadata", {})
            })
        
        context_block = "\n".join(context_lines)
        
        # Create retrieval-specific messages
        system_msg = {
            "role": "developer",
            "content": (
                "Answer using ONLY the provided context. Cite sources as [C#]. "
                "If context is insufficient, say what is missing. Be concise and technical."
            )
        }
        
        user_msg = {
            "role": role,
            "content": f"Question: {query}\n\nContext:\n{context_block}"
        }
        
        messages = [system_msg] + self._specs + [user_msg]
        
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages
        )
        answer = completion.choices[0].message.content
        
        # Update history
        self.messages.append({"role": role, "content": query})
        self.messages.append({"role": "assistant", "content": answer})
        
        return {
            "final_answer": answer,
            "citations": citations
        }
    
    def add_information(self, context_file_path: str):
        """
        Load point context from a JSON file.
        Consolidates add_information from model.py [2].
        """
        self.point_context = []
        self.point_in_active_context = True
        
        try:
            if "cascade" in context_file_path:
                with open(context_file_path, 'r') as file:
                    self.full_data = json.load(file)
                    
                    for kernel in self.full_data:
                        kernel_data = []
                        chiplets = kernel.get('chiplets', {})
                        
                        for chiplet_id, chiplet_info in chiplets.items():
                            chiplet_data = []
                            for key, value in chiplet_info.items():
                                if key != 'name':
                                    chiplet_data.append(value)
                            kernel_data.append(chiplet_data)
                        
                        # Add total data
                        total_data = []
                        for key, value in kernel.get('total', {}).items():
                            total_data.append(value)
                        kernel_data.append(total_data)
                        
                        self.point_context.append(kernel_data)
                    
                    self.point_context = np.array(self.point_context)
                    print(f"Point Context Shape: {self.point_context.shape}")
            else:
                # For pistil: load CSV or JSON based on file extension
                for path in context_file_path:
                    with open(path, 'r') as file:
                        self.point_context.append(file.read())
                    
        except FileNotFoundError:
            print(f"Error: File at {context_file_path} not found.")
            self.point_in_active_context = False

    def set_trace_or_model(self, trace_or_model: str):
        """
        Set the trace or model name for Pistil evaluations.
        Consolidates set_trace_or_model from model.py [2].
        """
        self.trace_or_model = trace_or_model
        print(f"[ChatBot] Trace/Model set to: {trace_or_model}")
    
    def add_run_context(self, 
                        summary_text: str, 
                        analytics_text: str = None, 
                        suggestions: str = None):
        """
        Add run context to the conversation history.
        Consolidates add_run_context from model.py [2].
        """
        self.messages.append({"role": "assistant", "content": summary_text})
        
        if analytics_text:
            self.messages.append({"role": "assistant", "content": analytics_text})
        
        if suggestions:
            self.messages.append({"role": "assistant", "content": suggestions})
        
        print(f"ChatBot: Added run context - Summary: {len(summary_text)} chars")
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API."""
        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    
    def get_distance_correlations(self, 
                                   objective_vals: List[List[float]],
                                   design_vals: List[List[float]],
                                   metric_names: List[str] = None,
                                   decision_names: List[str] = None) -> str:
        """
        Calculate distance correlations between objectives and design variables.
        
        This consolidates the get_distance_correlations logic from model.py [2].
        """
        from api.analysis.distance_correlation import DistanceCorrelationAnalyzer
        
        analyzer = DistanceCorrelationAnalyzer(self.evaluator)
        
        objective_array = np.array(objective_vals)
        design_array = np.array(design_vals)
        
        correlations = analyzer.calculate_correlations(
            objective_array, 
            design_array,
            metric_names,
            decision_names
        )
        
        return analyzer.get_correlation_string(correlations)
        
    def rule_mining(self, point_selection_params=None):
        # Extract max_pareto_rank from point_selection_params so the agent receives it
        max_pareto_rank = 3  # default
        if point_selection_params:
            max_pareto_rank = point_selection_params.get('pareto_end_rank', 3)

        context = {
            'point_selection_params': point_selection_params,
            'objectives': getattr(self, 'active_objectives', None),
            'max_pareto_rank': max_pareto_rank,
        }
        result = self.agents['rule_mining_agent'].execute(context)
        if result.success:
            return result.data.get('rules_string', result.message)
        return result.message

    def summarize(self, prompt: str) -> str:
        """Direct LLM completion with NO tool calling — for insight summaries."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "developer", "content": SystemPrompts.MAIN_ASSISTANT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content
    
    def optimization_manager(self, query: str) -> Dict[str, Any]:
        """
        Handle optimization requests.
        
        This delegates to the OptimizationAgent [2].
        """
        context = {'query': query}
        result = self.agents['optimization_agent'].execute(context)
        return result.to_dict()
    
    def add_enhanced_energy_analysis(self) -> str:
        """
        Add comprehensive energy analysis context.
        
        This delegates to the EnergyAnalysisAgent [2].
        """
        if not self.point_in_active_context or len(self.full_data) == 0:
            return "No active context or full data available for energy analysis."
        
        context = {
            'full_data': self.full_data,
            'point_context': self.point_context,
        }
        
        result = self.agents['energy_analysis_agent'].execute(context)
        return result.message
    
    def add_enhanced_runtime_analysis(self) -> str:
        """
        Add comprehensive runtime analysis context.
        
        This delegates to the RuntimeAnalysisAgent [2].
        """
        if not self.point_in_active_context or len(self.full_data) == 0:
            return "No active context or full data available for runtime analysis."
        
        context = {
            'full_data': self.full_data,
            'point_context': self.point_context,
        }
        
        result = self.agents['runtime_analysis_agent'].execute(context)
        return result.message
    
    def run_full_analysis(self) -> list:
        """Run all analysis agents in sequence, passing results forward."""
        from .agents.memory import AgentPipeline
        pipeline = AgentPipeline([
            self.agents['dcorr_agent'],
            self.agents['rule_mining_agent'],
            self.agents['energy_analysis_agent'],
            self.agents['runtime_analysis_agent'],
        ])
        return pipeline.execute(self._build_agent_context())

    def clear_history(self):
        self.messages = self._specs.copy()
        self.memory = ConversationMemory()   # Add this line
        # self.point_context = None
        self.point_in_active_context = False
        self.full_data = []
    
    def get_pareto_front_questions(self):
        """Initialize Pareto front context questions."""
        # Placeholder for compatibility with original implementation
        pass