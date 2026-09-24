"""
Energy Analysis Agent for detailed energy consumption analysis.
"""
import numpy as np
from typing import Dict, Any, List

from .base import BaseAgent, AgentResult
from .registry import AgentRegistry


@AgentRegistry.register
class EnergyAnalysisAgent(BaseAgent):
    """
    Agent that performs enhanced energy analysis on chiplet designs.
    Consolidates the add_enhanced_energy_analysis logic from model.py [2].
    """
    
    @property
    def name(self) -> str:
        return "energy_analysis_agent"
    
    @property
    def description(self) -> str:
        return "Performs enhanced energy analysis on the current design."
    
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        Execute energy analysis.
        
        Args:
            context: Dictionary containing full kernel data.
            
        Returns:
            AgentResult with energy analysis findings.
        """
        try:
            full_data = context.get('full_data', [])
            
            if not full_data:
                return AgentResult(
                    success=False,
                    message="No kernel data available for energy analysis. Please select a design point first.",
                    error="No full_data in context"
                )
            
            # Perform energy analysis
            analysis_result = self._analyze_energy_distribution(full_data)
            
            # Format the message
            message = self._format_energy_analysis(analysis_result)
            
            return AgentResult(
                success=True,
                message=message,
                data=analysis_result
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Error during energy analysis: {str(e)}",
                error=str(e)
            )
    
    def _analyze_energy_distribution(self, full_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze energy distribution across all chiplets.
        This consolidates the energy analysis logic from model.py [2].
        """
        energy_analysis = []
        all_type_stats = {}
        
        for kernel_idx, kernel_data in enumerate(full_data):
            kernel_name = kernel_data.get('name', f'Kernel {kernel_idx}')
            chiplets = kernel_data.get('chiplets', {})
            
            # Sort chiplets by energy consumption
            sorted_chiplets = sorted(
                chiplets.items(),
                key=lambda x: x[1].get('energy', 0),
                reverse=True
            )
            
            # Get top energy consumers
            top_consumers = sorted_chiplets[:3]
            
            # Calculate energy statistics
            all_energies = [chiplet[1].get('energy', 0) for chiplet in sorted_chiplets]
            total_energy = sum(all_energies)
            
            # Group by chiplet type
            type_energy = {}
            for chiplet_id, chiplet_data in sorted_chiplets:
                chiplet_type = chiplet_data.get('name', 'unknown')
                if chiplet_type not in type_energy:
                    type_energy[chiplet_type] = []
                type_energy[chiplet_type].append(chiplet_data.get('energy', 0))
            
            # Calculate type-specific statistics
            type_stats = {}
            for chiplet_type, energies in type_energy.items():
                type_stats[chiplet_type] = {
                    'count': len(energies),
                    'total_energy': sum(energies),
                    'avg_energy': sum(energies) / len(energies) if energies else 0,
                    'percentage': (sum(energies) / total_energy * 100) if total_energy > 0 else 0
                }
                
                # Aggregate across all kernels
                if chiplet_type not in all_type_stats:
                    all_type_stats[chiplet_type] = {
                        'total_energy': 0,
                        'count': 0,
                        'kernels': []
                    }
                all_type_stats[chiplet_type]['total_energy'] += type_stats[chiplet_type]['total_energy']
                all_type_stats[chiplet_type]['count'] += type_stats[chiplet_type]['count']
                all_type_stats[chiplet_type]['kernels'].append(kernel_name)
            
            energy_analysis.append({
                'kernel_name': kernel_name,
                'kernel_number': kernel_data.get('kernal_number', kernel_idx),
                'total_energy': total_energy,
                'top_consumers': [(cid, cdata) for cid, cdata in top_consumers],
                'type_statistics': type_stats
            })
        
        # Calculate overall averages
        for chiplet_type in all_type_stats:
            count = all_type_stats[chiplet_type]['count']
            all_type_stats[chiplet_type]['avg_energy'] = (
                all_type_stats[chiplet_type]['total_energy'] / count if count > 0 else 0
            )
        
        # Find main bottleneck
        if all_type_stats:
            main_bottleneck = max(all_type_stats.items(), key=lambda x: x[1]['avg_energy'])
        else:
            main_bottleneck = ('unknown', {'avg_energy': 0, 'total_energy': 0, 'count': 0, 'kernels': []})
        
        return {
            'kernel_analysis': energy_analysis,
            'type_statistics': all_type_stats,
            'main_bottleneck': {
                'type': main_bottleneck[0],
                'stats': main_bottleneck[1]
            },
            'total_energy': sum(k['total_energy'] for k in energy_analysis)
        }
    
    def _format_energy_analysis(self, analysis: Dict[str, Any]) -> str:
        """Format energy analysis results as a human-readable message."""
        bottleneck = analysis['main_bottleneck']
        type_stats = analysis['type_statistics']
        
        message = "Energy Bottleneck Analysis - High Energy Consumers:\n\n"
        
        # Main bottleneck
        message += f"Main Energy Bottleneck: {bottleneck['type'].upper()} chiplets\n"
        message += f"Average Energy: {bottleneck['stats']['avg_energy']:.3f} mJ per chiplet\n"
        message += f"Total Energy: {bottleneck['stats']['total_energy']:.3f} mJ\n"
        
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
            message += f"Top Energy Consumers ({first_kernel['kernel_name']} kernel):\n"
            for i, (chiplet_id, chiplet_data) in enumerate(first_kernel['top_consumers'], 1):
                chiplet_energy = chiplet_data.get('energy', 0)
                message += f"{i}. Chiplet {chiplet_id} ({chiplet_data.get('name', 'unknown')}): {chiplet_energy:.3f} mJ\n"
            message += "\n"
        
        # High energy consumers breakdown
        avg_threshold = sum(s['avg_energy'] for s in type_stats.values()) / len(type_stats) if type_stats else 0
        sorted_types = sorted(type_stats.items(), key=lambda x: x[1]['avg_energy'], reverse=True)
        
        message += "High Energy Consumers (Causing Higher Energy Consumption):\n"
        for chiplet_type, stats in sorted_types[:3]:
            message += f"- {chiplet_type.upper()}: {stats['avg_energy']:.3f} mJ avg ({stats['count']} chiplets)\n"
        
        return message
    
    def can_handle(self, query: str) -> bool:
        """Check if query is about energy analysis."""
        keywords = ['energy', 'power', 'consumption', 'bottleneck', 'efficient', 'mj', 'millijoule']
        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)