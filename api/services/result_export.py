import os
import csv
import zipfile
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
from django.conf import settings
from ..models import OptimizationRun, DesignPoint

class ResultExportService:
    """Service for exporting optimization results as zip files"""
    
    @staticmethod
    def generate_timestamped_filename(base_name: str = "myrun") -> str:
        """Generate a time-tagged filename"""
        timestamp = datetime.now().strftime("%Y%b%d_%H%M%S")
        return f"{base_name}_{timestamp}"
    
    @staticmethod
    def export_run_to_zip(run: OptimizationRun, base_name: str = None) -> str:
        """Export a single run to a zip file with CSV and text report"""
        if base_name is None:
            base_name = run.name or run.run_id
        
        # Generate timestamped filename
        timestamped_name = ResultExportService.generate_timestamped_filename(base_name)
        
        # Create export directory
        export_dir = Path(settings.MEDIA_ROOT) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        zip_path = export_dir / f"{timestamped_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV file with all design points
            csv_filename = f"{timestamped_name}_designs.csv"
            csv_content = ResultExportService._generate_csv_content(run)
            zipf.writestr(csv_filename, csv_content)
            
            # Add text report (placeholder for now)
            report_filename = f"{timestamped_name}_report.txt"
            report_content = ResultExportService._generate_text_report(run)
            zipf.writestr(report_filename, report_content)
            
            # Add metadata file
            metadata_filename = f"{timestamped_name}_metadata.json"
            metadata_content = ResultExportService._generate_metadata(run)
            zipf.writestr(metadata_filename, metadata_content)
        
        return str(zip_path)
    
    @staticmethod
    def export_comparative_study_to_zip(run_a: OptimizationRun, run_b: OptimizationRun, study_name: str = None) -> str:
        """Export a comparative study to a zip file"""
        if study_name is None:
            study_name = f"comparative_{run_a.run_id}_{run_b.run_id}"
        
        # Generate timestamped filename
        timestamped_name = ResultExportService.generate_timestamped_filename(study_name)
        
        # Create export directory
        export_dir = Path(settings.MEDIA_ROOT) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        zip_path = export_dir / f"{timestamped_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add CSV files for each run
            for run, suffix in [(run_a, "A"), (run_b, "B")]:
                csv_filename = f"{timestamped_name}_run{suffix}_designs.csv"
                csv_content = ResultExportService._generate_csv_content(run)
                zipf.writestr(csv_filename, csv_content)
            
            # Add comparative report
            report_filename = f"{timestamped_name}_comparative_report.txt"
            report_content = ResultExportService._generate_comparative_report(run_a, run_b)
            zipf.writestr(report_filename, report_content)
            
            # Add metadata
            metadata_filename = f"{timestamped_name}_metadata.json"
            metadata_content = ResultExportService._generate_comparative_metadata(run_a, run_b)
            zipf.writestr(metadata_filename, metadata_content)
        
        return str(zip_path)
    
    @staticmethod
    def _generate_csv_content(run: OptimizationRun) -> str:
        """Generate CSV content for a run's design points"""
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            'execution_time_ms', 'energy_mj', 'gpu_count', 'attention_count', 
            'sparse_count', 'convolution_count', 'total_chiplets', 'is_pareto_optimal'
        ])
        
        # Write design points
        design_points = run.design_points.all().order_by('execution_time_ms')
        for point in design_points:
            writer.writerow([
                point.execution_time_ms,
                point.energy_mj,
                point.gpu_count,
                point.attention_count,
                point.sparse_count,
                point.convolution_count,
                point.total_chiplets,
                'Yes' if point.is_pareto_optimal else 'No'
            ])
        
        return output.getvalue()
    
    @staticmethod
    def _generate_text_report(run: OptimizationRun) -> str:
        """Generate text report for a run (placeholder implementation)"""
        report_lines = [
            f"Optimization Run Report",
            f"=====================",
            f"",
            f"Run ID: {run.run_id}",
            f"Name: {run.name or 'Unnamed Run'}",
            f"Description: {run.description or 'No description provided'}",
            f"",
            f"Parameters:",
            f"  Algorithm: {run.algorithm}",
            f"  Model: {run.model}",
            f"  Population Size: {run.population_size}",
            f"  Generations: {run.generations}",
            f"  Objectives: {', '.join(run.objectives)}",
            f"  Trace: {run.trace_name or 'Not specified'}",
            f"",
            f"Results:",
            f"  Total Designs Evaluated: {run.total_designs_evaluated}",
            f"  Pareto Front Size: {run.pareto_front_size}",
            f"  Status: {run.status}",
            f"  Created: {run.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Completed: {run.completed_at.strftime('%Y-%m-%d %H:%M:%S') if run.completed_at else 'Not completed'}",
            f"",
            f"Pareto Optimal Designs:",
            f"======================",
        ]
        
        # Add Pareto optimal designs
        pareto_points = run.design_points.filter(is_pareto_optimal=True).order_by('execution_time_ms')
        if pareto_points:
            report_lines.extend([
                f"{'Time (ms)':<12} {'Energy (mJ)':<12} {'GPU':<4} {'ATTN':<5} {'SPARSE':<7} {'CONV':<5}",
                f"{'-'*12} {'-'*12} {'-'*4} {'-'*5} {'-'*7} {'-'*5}"
            ])
            for point in pareto_points:
                report_lines.append(
                    f"{point.execution_time_ms:<12.2f} {point.energy_mj:<12.2f} "
                    f"{point.gpu_count:<4} {point.attention_count:<5} "
                    f"{point.sparse_count:<7} {point.convolution_count:<5}"
                )
        else:
            report_lines.append("No Pareto optimal designs found.")
        
        report_lines.extend([
            f"",
            f"Note: This is a placeholder report. Detailed analysis and insights",
            f"will be added in future versions.",
            f"",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        return '\n'.join(report_lines)
    
    @staticmethod
    def _generate_comparative_report(run_a: OptimizationRun, run_b: OptimizationRun) -> str:
        """Generate comparative report for two runs"""
        report_lines = [
            f"Comparative Study Report",
            f"=======================",
            f"",
            f"Run A: {run_a.name or run_a.run_id}",
            f"Run B: {run_b.name or run_b.run_id}",
            f"",
            f"Comparison Summary:",
            f"  Run A - Total Designs: {run_a.total_designs_evaluated}, Pareto Size: {run_a.pareto_front_size}",
            f"  Run B - Total Designs: {run_b.total_designs_evaluated}, Pareto Size: {run_b.pareto_front_size}",
            f"",
            f"Note: This is a placeholder comparative report. Detailed analysis",
            f"and insights will be added in future versions.",
            f"",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        return '\n'.join(report_lines)
    
    @staticmethod
    def _generate_metadata(run: OptimizationRun) -> str:
        """Generate JSON metadata for a run"""
        import json
        
        metadata = {
            'run_id': run.run_id,
            'name': run.name,
            'description': run.description,
            'algorithm': run.algorithm,
            'model': run.model,
            'population_size': run.population_size,
            'generations': run.generations,
            'objectives': run.objectives,
            'trace_name': run.trace_name,
            'trace_sets': run.trace_sets,
            'status': run.status,
            'created_at': run.created_at.isoformat(),
            'completed_at': run.completed_at.isoformat() if run.completed_at else None,
            'total_designs_evaluated': run.total_designs_evaluated,
            'pareto_front_size': run.pareto_front_size,
            'execution_time_seconds': run.execution_time_seconds,
            'results_directory': run.results_directory,
            'analytics_directory': run.analytics_directory
        }
        
        return json.dumps(metadata, indent=2)
    
    @staticmethod
    def _generate_comparative_metadata(run_a: OptimizationRun, run_b: OptimizationRun) -> str:
        """Generate JSON metadata for a comparative study"""
        import json
        
        metadata = {
            'study_type': 'comparative',
            'run_a': {
                'run_id': run_a.run_id,
                'name': run_a.name,
                'total_designs': run_a.total_designs_evaluated,
                'pareto_size': run_a.pareto_front_size
            },
            'run_b': {
                'run_id': run_b.run_id,
                'name': run_b.name,
                'total_designs': run_b.total_designs_evaluated,
                'pareto_size': run_b.pareto_front_size
            },
            'generated_at': datetime.now().isoformat()
        }
        
        return json.dumps(metadata, indent=2)
    
    @staticmethod
    def get_export_url(zip_path: str) -> str:
        """Convert file path to URL for download"""
        # Remove the media root from the path to get the relative URL
        relative_path = zip_path.replace(str(settings.MEDIA_ROOT), '')
        return f"/media/exports{relative_path}"
    
    @staticmethod
    def cleanup_old_exports(max_age_days: int = 30) -> int:
        """Clean up old export files"""
        import time
        from datetime import timedelta
        
        export_dir = Path(settings.MEDIA_ROOT) / "exports"
        if not export_dir.exists():
            return 0
        
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        deleted_count = 0
        
        for file_path in export_dir.glob("*.zip"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
        
        return deleted_count 