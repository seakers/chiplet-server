from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from django.db.models import Q
from .models import OptimizationRun, DesignPoint, ComparativeStudy
from .serializers import (
    OptimizationRunSerializer, 
    OptimizationRunListSerializer,
    ComparativeStudySerializer,
    PlotDataSerializer,
    RunComparisonSerializer
)
from .services.run_storage import RunStorageService
from .services.result_export import ResultExportService
import json
from datetime import datetime

@api_view(['GET'])
@permission_classes([AllowAny])
def list_runs(request):
    """List all optimization runs"""
    runs = OptimizationRun.objects.all()
    serializer = OptimizationRunListSerializer(runs, many=True)
    return Response({
        'status': 'success',
        'runs': serializer.data,
        'total_count': runs.count()
    })

@api_view(['GET'])
@permission_classes([AllowAny])
def get_run_details(request, run_id):
    """Get detailed information about a specific run"""
    run = get_object_or_404(OptimizationRun, run_id=run_id)
    serializer = OptimizationRunSerializer(run)
    return Response({
        'status': 'success',
        'run': serializer.data
    })

@api_view(['GET'])
@permission_classes([AllowAny])
def get_run_for_plotting(request, run_id):
    """Get run data formatted specifically for plotting"""
    plot_data = RunStorageService.get_run_for_plotting(run_id)
    if plot_data:
        return Response({
            'status': 'success',
            'data': plot_data
        })
    else:
        return Response({
            'status': 'error',
            'message': f'Run {run_id} not found'
        }, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_multiple_runs_for_plotting(request):
    """Get multiple runs for comparison plotting"""
    run_ids = request.GET.getlist('run_ids')
    if not run_ids:
        return Response({
            'status': 'error',
            'message': 'No run IDs provided'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    runs_data = []
    for run_id in run_ids:
        run_data = RunStorageService.get_run_for_plotting(run_id)
        if run_data:
            runs_data.append(run_data)
    
    return Response({
        'status': 'success',
        'runs': runs_data,
        'total_runs': len(runs_data)
    })

@api_view(['GET'])
@permission_classes([AllowAny])
def compare_runs(request):
    """Compare two specific runs"""
    run_a_id = request.GET.get('run_a_id')
    run_b_id = request.GET.get('run_b_id')
    
    if not run_a_id or not run_b_id:
        return Response({
            'status': 'error',
            'message': 'Both run_a_id and run_b_id are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    comparison_data = RunStorageService.get_comparative_study_data(run_a_id, run_b_id)
    if comparison_data:
        return Response({
            'status': 'success',
            'comparison': comparison_data
        })
    else:
        return Response({
            'status': 'error',
            'message': 'One or both runs not found'
        }, status=status.HTTP_404_NOT_FOUND)

@api_view(['GET'])
@permission_classes([AllowAny])
def search_runs(request):
    """Search runs by various criteria"""
    query = request.GET.get('q', '')
    algorithm = request.GET.get('algorithm', '')
    status_filter = request.GET.get('status', '')
    trace_name = request.GET.get('trace_name', '')
    
    runs = OptimizationRun.objects.all()
    
    if query:
        runs = runs.filter(
            Q(name__icontains=query) | 
            Q(description__icontains=query) | 
            Q(run_id__icontains=query)
        )
    
    if algorithm:
        runs = runs.filter(algorithm=algorithm)
    
    if status_filter:
        runs = runs.filter(status=status_filter)
    
    if trace_name:
        runs = runs.filter(trace_name__icontains=trace_name)
    
    serializer = OptimizationRunListSerializer(runs, many=True)
    return Response({
        'status': 'success',
        'runs': serializer.data,
        'total_count': runs.count(),
        'filters_applied': {
            'query': query,
            'algorithm': algorithm,
            'status': status_filter,
            'trace_name': trace_name
        }
    })

@api_view(['GET'])
@permission_classes([AllowAny])
def get_pareto_fronts(request):
    """Get Pareto fronts for multiple runs"""
    run_ids = request.GET.getlist('run_ids')
    if not run_ids:
        return Response({
            'status': 'error',
            'message': 'No run IDs provided'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    pareto_fronts = {}
    for run_id in run_ids:
        run_data = RunStorageService.get_run_for_plotting(run_id)
        if run_data:
            pareto_fronts[run_id] = {
                'run_name': run_data['run_name'],
                'pareto_front': run_data['pareto_front'],
                'metadata': run_data['metadata']
            }
    
    return Response({
        'status': 'success',
        'pareto_fronts': pareto_fronts
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def create_comparative_study(request):
    """Create a new comparative study"""
    try:
        data = json.loads(request.body)
        run_a_id = data.get('run_a_id')
        run_b_id = data.get('run_b_id')
        name = data.get('name', '')
        description = data.get('description', '')
        shared_parameters = data.get('shared_parameters', {})
        
        if not run_a_id or not run_b_id:
            return Response({
                'status': 'error',
                'message': 'Both run_a_id and run_b_id are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        run_a = get_object_or_404(OptimizationRun, run_id=run_a_id)
        run_b = get_object_or_404(OptimizationRun, run_id=run_b_id)
        
        study = ComparativeStudy.objects.create(
            study_id=f"study_{run_a_id}_{run_b_id}",
            name=name,
            description=description,
            run_a=run_a,
            run_b=run_b,
            shared_parameters=shared_parameters
        )
        
        serializer = ComparativeStudySerializer(study)
        return Response({
            'status': 'success',
            'study': serializer.data
        })
        
    except json.JSONDecodeError:
        return Response({
            'status': 'error',
            'message': 'Invalid JSON data'
        }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            'status': 'error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def list_comparative_studies(request):
    """List all comparative studies"""
    studies = ComparativeStudy.objects.all()
    serializer = ComparativeStudySerializer(studies, many=True)
    return Response({
        'status': 'success',
        'studies': serializer.data,
        'total_count': studies.count()
    })

@api_view(['POST'])
@permission_classes([AllowAny])
def migrate_existing_data(request):
    """Migrate existing CSV data to database"""
    try:
        migration_stats = RunStorageService.migrate_existing_data()
        return Response({
            'status': 'success',
            'message': 'Data migration completed',
            'stats': migration_stats
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Migration failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['DELETE'])
@permission_classes([AllowAny])
def delete_run(request, run_id):
    """Delete a run and all associated data"""
    try:
        run = get_object_or_404(OptimizationRun, run_id=run_id)
        run_name = run.name or run.run_id
        run.delete()
        
        return Response({
            'status': 'success',
            'message': f'Run {run_name} deleted successfully'
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Failed to delete run: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
@permission_classes([AllowAny])
def get_run_statistics(request):
    """Get overall statistics about all runs"""
    total_runs = OptimizationRun.objects.count()
    completed_runs = OptimizationRun.objects.filter(status='completed').count()
    running_runs = OptimizationRun.objects.filter(status='running').count()
    failed_runs = OptimizationRun.objects.filter(status='failed').count()
    
    total_design_points = DesignPoint.objects.count()
    total_pareto_points = DesignPoint.objects.filter(is_pareto_optimal=True).count()
    
    # Algorithm distribution
    algorithm_stats = {}
    for run in OptimizationRun.objects.all():
        algorithm = run.algorithm
        algorithm_stats[algorithm] = algorithm_stats.get(algorithm, 0) + 1
    
    # Trace distribution
    trace_stats = {}
    for run in OptimizationRun.objects.all():
        trace = run.trace_name
        if trace:
            trace_stats[trace] = trace_stats.get(trace, 0) + 1
    
    return Response({
        'status': 'success',
        'statistics': {
            'total_runs': total_runs,
            'completed_runs': completed_runs,
            'running_runs': running_runs,
            'failed_runs': failed_runs,
            'total_design_points': total_design_points,
            'total_pareto_points': total_pareto_points,
            'algorithm_distribution': algorithm_stats,
            'trace_distribution': trace_stats
        }
    })

# New export endpoints
@api_view(['POST'])
@permission_classes([AllowAny])
def export_run_to_zip(request, run_id):
    """Export a single run to a zip file"""
    try:
        run = get_object_or_404(OptimizationRun, run_id=run_id)
        base_name = request.data.get('base_name') if request.data else None
        
        zip_path = ResultExportService.export_run_to_zip(run, base_name)
        download_url = ResultExportService.get_export_url(zip_path)
        
        return Response({
            'status': 'success',
            'message': f'Run {run.name or run.run_id} exported successfully',
            'download_url': download_url,
            'file_path': zip_path
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Export failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def export_comparative_study_to_zip(request):
    """Export a comparative study to a zip file"""
    try:
        data = request.data if request.data else {}
        run_a_id = data.get('run_a_id')
        run_b_id = data.get('run_b_id')
        study_name = data.get('study_name')
        
        if not run_a_id or not run_b_id:
            return Response({
                'status': 'error',
                'message': 'Both run_a_id and run_b_id are required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        run_a = get_object_or_404(OptimizationRun, run_id=run_a_id)
        run_b = get_object_or_404(OptimizationRun, run_id=run_b_id)
        
        zip_path = ResultExportService.export_comparative_study_to_zip(run_a, run_b, study_name)
        download_url = ResultExportService.get_export_url(zip_path)
        
        return Response({
            'status': 'success',
            'message': f'Comparative study exported successfully',
            'download_url': download_url,
            'file_path': zip_path
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Export failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def export_multiple_runs_to_zip(request):
    """Export multiple runs to a single zip file"""
    try:
        data = request.data if request.data else {}
        run_ids = data.get('run_ids', [])
        base_name = data.get('base_name', 'multi_run')
        
        if not run_ids:
            return Response({
                'status': 'error',
                'message': 'No run IDs provided'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        runs = []
        for run_id in run_ids:
            run = get_object_or_404(OptimizationRun, run_id=run_id)
            runs.append(run)
        
        # Generate timestamped filename
        timestamped_name = ResultExportService.generate_timestamped_filename(base_name)
        
        # Create export directory
        from pathlib import Path
        from django.conf import settings
        import zipfile
        
        export_dir = Path(settings.MEDIA_ROOT) / "exports"
        export_dir.mkdir(exist_ok=True)
        
        zip_path = export_dir / f"{timestamped_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add each run's data
            for i, run in enumerate(runs):
                # CSV file for this run
                csv_filename = f"{timestamped_name}_run{i+1}_{run.run_id}_designs.csv"
                csv_content = ResultExportService._generate_csv_content(run)
                zipf.writestr(csv_filename, csv_content)
                
                # Report for this run
                report_filename = f"{timestamped_name}_run{i+1}_{run.run_id}_report.txt"
                report_content = ResultExportService._generate_text_report(run)
                zipf.writestr(report_filename, report_content)
            
            # Add summary metadata
            metadata_filename = f"{timestamped_name}_summary.json"
            summary_data = {
                'export_type': 'multiple_runs',
                'base_name': base_name,
                'total_runs': len(runs),
                'runs': [
                    {
                        'run_id': run.run_id,
                        'name': run.name,
                        'total_designs': run.total_designs_evaluated,
                        'pareto_size': run.pareto_front_size
                    }
                    for run in runs
                ],
                'generated_at': datetime.now().isoformat()
            }
            import json
            zipf.writestr(metadata_filename, json.dumps(summary_data, indent=2))
        
        download_url = ResultExportService.get_export_url(str(zip_path))
        
        return Response({
            'status': 'success',
            'message': f'{len(runs)} runs exported successfully',
            'download_url': download_url,
            'file_path': str(zip_path)
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Export failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
@permission_classes([AllowAny])
def cleanup_exports(request):
    """Clean up old export files"""
    try:
        data = request.data if request.data else {}
        max_age_days = data.get('max_age_days', 30)
        
        deleted_count = ResultExportService.cleanup_old_exports(max_age_days)
        
        return Response({
            'status': 'success',
            'message': f'Cleaned up {deleted_count} old export files',
            'deleted_count': deleted_count,
            'max_age_days': max_age_days
        })
    except Exception as e:
        return Response({
            'status': 'error',
            'message': f'Cleanup failed: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR) 