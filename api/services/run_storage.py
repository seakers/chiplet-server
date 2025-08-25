import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from django.utils import timezone
from ..models import OptimizationRun, DesignPoint, AnalyticsResult, ComparativeStudy

class RunStorageService:
    """Service for storing and retrieving optimization runs"""
    
    @staticmethod
    def generate_run_id() -> str:
        """Generate a unique run ID"""
        return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    @staticmethod
    def create_optimization_run(
        algorithm: str = 'GA',
        model: str = 'CASCADE',
        population_size: int = 50,
        generations: int = 100,
        objectives: List[str] = None,
        trace_name: str = '',
        trace_sets: Dict = None,
        name: str = '',
        description: str = ''
    ) -> OptimizationRun:
        """Create a new optimization run record"""
        if objectives is None:
            objectives = ['execution_time', 'energy']
        
        run = OptimizationRun.objects.create(
            run_id=RunStorageService.generate_run_id(),
            name=name,
            description=description,
            algorithm=algorithm,
            model=model,
            population_size=population_size,
            generations=generations,
            objectives=objectives,
            trace_name=trace_name,
            trace_sets=trace_sets or {},
            status='running'
        )
        return run
    
    @staticmethod
    def store_design_points(
        run: OptimizationRun,
        design_points: List[Dict[str, Any]],
        results_directory: str = '',
        update_pareto: bool = True
    ) -> List[DesignPoint]:
        """Store design points for a run"""
        stored_points = []
        
        for point_data in design_points:
            # Extract design configuration
            chiplets = point_data.get('chiplets', {})
            gpu_count = chiplets.get('GPU', 0)
            attention_count = chiplets.get('Attention', 0)
            sparse_count = chiplets.get('Sparse', 0)
            convolution_count = chiplets.get('Convolution', 0)
            
            # Extract objective values
            execution_time_ms = point_data.get('execution_time_ms', point_data.get('x', 0))
            energy_mj = point_data.get('energy_mj', point_data.get('y', 0))
            
            # Create or update design point
            design_point, created = DesignPoint.objects.get_or_create(
                run=run,
                gpu_count=gpu_count,
                attention_count=attention_count,
                sparse_count=sparse_count,
                convolution_count=convolution_count,
                defaults={
                    'execution_time_ms': execution_time_ms,
                    'energy_mj': energy_mj,
                    'additional_metrics': point_data.get('additional_metrics', {}),
                    'context_file_path': point_data.get('context_file_path', '')
                }
            )
            
            if not created:
                # Update existing point
                design_point.execution_time_ms = execution_time_ms
                design_point.energy_mj = energy_mj
                design_point.additional_metrics = point_data.get('additional_metrics', {})
                design_point.save()
            
            stored_points.append(design_point)
        
        # Update run statistics
        run.total_designs_evaluated = len(stored_points)
        run.results_directory = results_directory
        run.save()
        
        # Update Pareto front if requested
        if update_pareto:
            RunStorageService.update_pareto_front(run)
        
        return stored_points
    
    @staticmethod
    def update_pareto_front(run: OptimizationRun) -> None:
        """Update Pareto front information for a run"""
        design_points = run.design_points.all()
        
        # Simple Pareto dominance check
        pareto_optimal = []
        for point in design_points:
            is_dominated = False
            for other_point in design_points:
                if point != other_point:
                    # Check if other_point dominates point
                    if (other_point.execution_time_ms <= point.execution_time_ms and 
                        other_point.energy_mj <= point.energy_mj and
                        (other_point.execution_time_ms < point.execution_time_ms or 
                         other_point.energy_mj < point.energy_mj)):
                        is_dominated = True
                        break
            
            point.is_pareto_optimal = not is_dominated
            point.save()
            
            if not is_dominated:
                pareto_optimal.append(point)
        
        # Update run statistics
        run.pareto_front_size = len(pareto_optimal)
        run.save()
    
    @staticmethod
    def complete_run(
        run: OptimizationRun,
        execution_time_seconds: float = None,
        analytics_results: Dict[str, Any] = None
    ) -> OptimizationRun:
        """Mark a run as completed and store analytics"""
        run.status = 'completed'
        run.completed_at = timezone.now()
        if execution_time_seconds:
            run.execution_time_seconds = execution_time_seconds
        run.save()
        
        # Store analytics results
        if analytics_results:
            RunStorageService.store_analytics_results(run, analytics_results)
        
        return run
    
    @staticmethod
    def store_analytics_results(
        run: OptimizationRun,
        analytics_results: Dict[str, Any]
    ) -> List[AnalyticsResult]:
        """Store analytics results for a run"""
        stored_analytics = []
        
        for analytics_type, result_data in analytics_results.items():
            if isinstance(result_data, dict) and 'file_path' in result_data:
                file_path = result_data['file_path']
                results_data = result_data.get('data', {})
            else:
                file_path = ''
                results_data = result_data if isinstance(result_data, dict) else {'result': result_data}
            
            analytics_result, created = AnalyticsResult.objects.get_or_create(
                run=run,
                analytics_type=analytics_type,
                defaults={
                    'results_data': results_data,
                    'file_path': file_path
                }
            )
            
            if not created:
                analytics_result.results_data = results_data
                analytics_result.file_path = file_path
                analytics_result.save()
            
            stored_analytics.append(analytics_result)
        
        return stored_analytics
    
    @staticmethod
    def get_run_for_plotting(run_id: str) -> Dict[str, Any]:
        """Get run data formatted for plotting"""
        try:
            run = OptimizationRun.objects.get(run_id=run_id)
            design_points = run.design_points.all()
            pareto_front = design_points.filter(is_pareto_optimal=True)
            
            return {
                'run_id': run.run_id,
                'run_name': run.name or run.run_id,
                'design_points': [
                    {
                        'x': point.execution_time_ms,
                        'y': point.energy_mj,
                        'gpu': point.gpu_count,
                        'attn': point.attention_count,
                        'sparse': point.sparse_count,
                        'conv': point.convolution_count,
                        'is_pareto': point.is_pareto_optimal
                    }
                    for point in design_points
                ],
                'pareto_front': [
                    {
                        'x': point.execution_time_ms,
                        'y': point.energy_mj,
                        'gpu': point.gpu_count,
                        'attn': point.attention_count,
                        'sparse': point.sparse_count,
                        'conv': point.convolution_count
                    }
                    for point in pareto_front
                ],
                'metadata': {
                    'algorithm': run.algorithm,
                    'population_size': run.population_size,
                    'generations': run.generations,
                    'trace_name': run.trace_name,
                    'created_at': run.created_at.isoformat(),
                    'total_designs': run.total_designs_evaluated,
                    'pareto_size': run.pareto_front_size
                }
            }
        except OptimizationRun.DoesNotExist:
            return None
    
    @staticmethod
    def get_comparative_study_data(run_a_id: str, run_b_id: str) -> Dict[str, Any]:
        """Get data for comparing two runs"""
        run_a_data = RunStorageService.get_run_for_plotting(run_a_id)
        run_b_data = RunStorageService.get_run_for_plotting(run_b_id)
        
        if not run_a_data or not run_b_data:
            return None
        
        return {
            'run_a': run_a_data,
            'run_b': run_b_data,
            'comparison': {
                'run_a_pareto_size': len(run_a_data['pareto_front']),
                'run_b_pareto_size': len(run_b_data['pareto_front']),
                'run_a_total_designs': run_a_data['metadata']['total_designs'],
                'run_b_total_designs': run_b_data['metadata']['total_designs']
            }
        }
    
    @staticmethod
    def migrate_existing_data() -> Dict[str, int]:
        """Migrate existing CSV data to database"""
        import csv
        from pathlib import Path
        
        results_dir = Path("api/Evaluator/cascade/chiplet_model/dse/results")
        migration_stats = {'runs_created': 0, 'points_migrated': 0}
        
        # Check for main points.csv
        main_csv = results_dir / "points.csv"
        if main_csv.exists():
            # Create a run for existing data
            run = RunStorageService.create_optimization_run(
                name="Migrated Historical Data",
                description="Data migrated from existing CSV files",
                algorithm='GA',
                population_size=50,
                generations=100,
                trace_name='gpt-j-65536-weighted'
            )
            
            # Read and store design points
            design_points = []
            with open(main_csv, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 6:
                        design_points.append({
                            'execution_time_ms': float(row[0]),
                            'energy_mj': float(row[1]),
                            'chiplets': {
                                'GPU': int(row[2]),
                                'Attention': int(row[3]),
                                'Sparse': int(row[4]),
                                'Convolution': int(row[5])
                            }
                        })
            
            RunStorageService.store_design_points(run, design_points, str(results_dir))
            RunStorageService.complete_run(run)
            
            migration_stats['runs_created'] += 1
            migration_stats['points_migrated'] += len(design_points)
        
        return migration_stats 