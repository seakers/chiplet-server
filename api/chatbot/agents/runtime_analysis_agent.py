"""
Runtime Analysis Agent for detailed execution time analysis.
"""
import numpy as np
from typing import Dict, Any, List

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry


@AgentRegistry.register
class RuntimeAnalysisAgent(BaseAgent):
    """
    Agent that performs enhanced runtime analysis on chiplet designs.
    Consolidates the add_enhanced_runtime_analysis logic from model.py [2].
    """
    
    @property
    def name(self) -> str:
        return "runtime_analysis_agent"
    
    @property
    def description(self) -> str:
        return "Performs enhanced runtime analysis on the current design."
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute runtime analysis.
        
        Args:
            context: Dictionary containing full kernel data.
            
        Returns:
            AgentResult with runtime analysis findings.
        """
        try:
            full_data = context.get('full_data', [])
            
            if not full_data:
                return AgentResult(
                    success=False,
                    message="No kernel data available for runtime analysis. Please select a design point first.",
                    error="No full_data in context"
                )
            
            # Perform runtime analysis
            analysis_result = self._analyze_runtime_distribution(full_data)
            
            # Format the message
            message = self._format_runtime_analysis(analysis_result)
            
            return AgentResult(
                success=True,
                message=message,
                data=analysis_result
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error during runtime analysis: {str(e)}",
                error=str(e)
            )
    
    def _analyze_runtime_distribution(self, full_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze runtime distribution across all chiplets.
        This consolidates the runtime analysis logic from model.py [2].
        """
        runtime_analysis = []
        all_type_stats = {}
        
        for kernel_idx, kernel_data in enumerate(full_data):
            kernel_name = kernel_data.get('name', f'Kernel {kernel_idx}')
            chiplets = kernel_data.get('chiplets', {})
            
            # Sort chiplets by execution time
            sorted_chiplets = sorted(
                chiplets.items(),
                key=lambda x: x[1].get('exe_time', 0),
                reverse=True
            )
            
            # Get top runtime consumers
            top_consumers = sorted_chiplets[:3]
            
            # Calculate runtime statistics
            all_runtimes = [chiplet[1].get('exe_time', 0) for chiplet in sorted_chiplets]
            total_runtime = sum(all_runtimes)
            
            # Group by chiplet type
            type_runtime = {}
            for chiplet_id, chiplet_data in sorted_chiplets:
                chiplet_type = chiplet_data.get('name', 'unknown')
                if chiplet_type not in type_runtime:
                    type_runtime[chiplet_type] = []
                type_runtime[chiplet_type].append(chiplet_data.get('exe_time', 0))
            
            # Calculate type-specific statistics
            type_stats = {}
            for chiplet_type, runtimes in type_runtime.items():
                type_stats[chiplet_type] = {
                    'count': len(runtimes),
                    'total_runtime': sum(runtimes),
                    'avg_runtime': sum(runtimes) / len(runtimes) if runtimes else 0,
                    'percentage': (sum(runtimes) / total_runtime * 100) if total_runtime > 0 else 0
                }
                
                # Aggregate across all kernels
                if chiplet_type not in all_type_stats:
                    all_type_stats[chiplet_type] = {
                        'total_runtime': 0,
                        'count': 0,
                        'kernels': []
                    }
                all_type_stats[chiplet_type]['total_runtime'] += type_stats[chiplet_type]['total_runtime']
                all_type_stats[chiplet_type]['count'] += type_stats[chiplet_type]['count']
                all_type_stats[chiplet_type]['kernels'].append(kernel_name)
            
            runtime_analysis.append({
                'kernel_name': kernel_name,
                'kernel_number': kernel_data.get('kernal_number', kernel_idx),
                'total_runtime': total_runtime,
                'top_consumers': [(cid, cdata) for cid, cdata in top_consumers],
                'type_statistics': type_stats
            })
        
        # Calculate overall averages
        for chiplet_type in all_type_stats:
            count = all_type_stats[chiplet_type]['count']
            all_type_stats[chiplet_type]['avg_runtime'] = (
                all_type_stats[chiplet_type]['total_runtime'] / count if count > 0 else 0
            )
        
        # Find main bottleneck
        if all_type_stats:
            main_bottleneck = max(all_type_stats.items(), key=lambda x: x[1]['avg_runtime'])
        else:
            main_bottleneck = ('unknown', {'avg_runtime': 0, 'total_runtime': 0, 'count': 0, 'kernels': []})
        
        return {
            'kernel_analysis': runtime_analysis,
            'type_statistics': all_type_stats,
            'main_bottleneck': {
                'type': main_bottleneck[0],
                'stats': main_bottleneck[1]
            },
            'total_runtime': sum(k['total_runtime'] for k in runtime_analysis)
        }
    
    def _format_runtime_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format runtime analysis results as a human-readable message."""
        bottleneck = analysis['main_bottleneck']
        type_stats = analysis['type_statistics']
        
        message = "Runtime Bottleneck Analysis:\n\n"
        
        # Main bottleneck
        message += f"Main Runtime Bottleneck: {bottleneck['type'].upper()} chiplets\n"
        message += f"Average Runtime: {bottleneck['stats']['avg_runtime']:.3f} ms per chiplet\n"
        message += f"Total Runtime: {bottleneck['stats']['total_runtime']:.3f} ms\n"
        
        kernels = bottleneck['stats'].get('kernels', [])
        if kernels:
            message += f"Evidence: Observed in {len(kernels)} kernel(s): {', '.join(kernels[:5])}"
            if len(kernels) > 5:
                message += "..."
            message += "\n\n"
        
        # Top consumers from first kernel
        kernel_analysis = analysis.get('kernel_analysis', [])
        if kernel_analysis:
            first_kernel = kernel_analysis[0]
            message += f"Top Runtime Consumers ({first_kernel['kernel_name']} kernel):\n"
            for i, (chiplet_id, chiplet_data) in enumerate(first_kernel['top_consumers'], 1):
                exe_time = chiplet_data.get('exe_time', 0)
                message += f"{i}. Chiplet {chiplet_id} ({chiplet_data.get('name', 'unknown')}): {exe_time:.3f} ms\n"
            message += "\n"
        
        # Runtime breakdown by type
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['avg_runtime'], reverse=True)
        
        message += "Runtime Breakdown by Type:\n"
        for chiplet_type, stats in sorted_types:
            message += f"- {chiplet_type.upper()}: {stats['avg_runtime']:.3f} ms avg ({stats['count']} chiplets)\n"
        
        return message
    
    def can_handle(self, query: str) -> bool:
        """Check if query is about runtime analysis."""
        keywords = ['runtime', 'execution', 'time', 'performance', 'slow', 'fast', 'speed', 'latency', 'ms']
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)