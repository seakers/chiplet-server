"""
Report generation API endpoints.
Consolidates report endpoints from views.py [1].
"""
import os
import sys
import json
import csv
import re
from datetime import datetime

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.config.settings import EVALUATOR_BASE_PATHS
from api.data.loaders import PointsLoader, RunDataLoader
from api.analysis.pareto import ParetoCalculator
from api.analysis.rule_mining import RuleMiner, RuleFormatter
from api.analysis.distance_correlation import DistanceCorrelationAnalyzer
from api.reporting.generators import ReportGenerator
from api.chatbot.bot import ChatBot
from api.models import OptimizationRun


@api_view(["GET"])
def generate_report(request):
    """
    Generate an HTML report for the current optimization run.
    Refactored from views.py [1].
    """
    try:
        evaluator = request.GET.get("evaluator", "CASCADE")
        run_id = request.GET.get("run_id")
        
        # Get run parameters from database
        run_params = {
            'model': 'CASCADE',
            'algorithm': 'Genetic Algorithm',
            'objectives': ['Energy', 'Runtime'],
            'population_size': 50,
            'generations': 100,
            'trace_name': 'gpt-j-65536-weighted'
        }
        
        if run_id:
            try:
                optimization_run = OptimizationRun.objects.get(run_id=run_id)
                algorithm_display = optimization_run.get_algorithm_display()
                
                run_params = {
                    'model': optimization_run.model,
                    'algorithm': algorithm_display,
                    'objectives': optimization_run.objectives,
                    'population_size': optimization_run.population_size,
                    'generations': optimization_run.generations,
                    'trace_name': optimization_run.trace_name or 'gpt-j-65536-weighted'
                }
                print(f"[generate_report] Using params from DB: {run_params}")
            except OptimizationRun.DoesNotExist:
                print(f"[generate_report] Run {run_id} not found, using defaults")
        
        # Load points data
        loader = PointsLoader(evaluator, run_id)
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        points = loader.load_points_as_dicts(file_path)
        
        if not points:
            return JsonResponse({
                "status": "error",
                "message": "No data available for report"
            }, status=400)
        
        # Calculate Pareto front
        pareto_calculator = ParetoCalculator()
        pareto_points = pareto_calculator.get_pareto_front(points)
        
        # Get rule mining results
        chat_bot = ChatBot(evaluator=evaluator, run_id=run_id)
        rule_mining_str = chat_bot.rule_mining()
        
        # Parse rules
        rules = []
        rule_pattern = re.compile(
            r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), conf\(p->f\): ([0-9.eE+-]+), lift: \(?([0-9.eE+-]+)\)?"
        )
        for match in rule_pattern.finditer(rule_mining_str):
            rules.append({
                "rule": match.group(1),
                "conf_f_to_p": float(match.group(2)),
                "conf_p_to_f": float(match.group(3)),
                "lift": float(match.group(4)),
            })
        
        # Get distance correlations
        analyzer = DistanceCorrelationAnalyzer(evaluator)
        corr_results = analyzer.analyze_from_points(points)
        correlations = corr_results['correlations']
        
        # Generate report
        report_generator = ReportGenerator(evaluator)
        html_content = report_generator.generate_single_run_report(
            run_params=run_params,
            points=points,
            pareto_points=pareto_points,
            rules=rules,
            correlations=correlations,
            run_id=run_id
        )
        
        # Save report
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        reports_dir = os.path.join(WORKSPACE, 'dse/results/reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{run_id or 'current'}_{timestamp}.html"
        report_path = os.path.join(reports_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        web_link = f"/api/Evaluator/cascade/chiplet_model/dse/results/reports/{report_filename}"
        
        return JsonResponse({
            "status": "success",
            "report_content": html_content,
            "web_link": web_link,
            "download_link": web_link
        })
        
    except Exception as e:
        print(f"Error in generate_report: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@api_view(["GET"])
def get_previous_run_report(request):
    """
    Generate report for a previously loaded run.
    Refactored from views.py [1].
    """
    try:
        backup_filename = request.GET.get("backup_filename")
        evaluator = request.GET.get("evaluator", "CASCADE")
        
        if not backup_filename:
            return JsonResponse({
                "status": "error",
                "message": "backup_filename is required"
            }, status=400)
        
        # Load data from backup file
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        backup_path = os.path.join(results_dir, backup_filename)
        
        if not os.path.exists(backup_path):
            return JsonResponse({
                "status": "error",
                "message": f"Backup file not found: {backup_filename}"
            }, status=404)
        
        # Load points
        loader = PointsLoader(evaluator)
        points = loader.load_points_as_dicts(backup_path)
        
        # Get run metadata from zip if available
        run_data_loader = RunDataLoader(evaluator)
        run_params = run_data_loader.load_run_metadata(backup_filename)
        
        # Calculate Pareto front
        pareto_calculator = ParetoCalculator()
        pareto_points = pareto_calculator.get_pareto_front(points)
        
        # Get analytics
        chat_bot = ChatBot(evaluator=evaluator)
        rule_mining_str = chat_bot.rule_mining()
        
        rules = []
        rule_pattern = re.compile(
            r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), conf\(p->f\): ([0-9.eE+-]+), lift: \(?([0-9.eE+-]+)\)?"
        )
        for match in rule_pattern.finditer(rule_mining_str):
            rules.append({
                "rule": match.group(1),
                "conf_f_to_p": float(match.group(2)),
                "conf_p_to_f": float(match.group(3)),
                "lift": float(match.group(4)),
            })
        
        analyzer = DistanceCorrelationAnalyzer(evaluator)
        corr_results = analyzer.analyze_from_points(points)
        correlations = corr_results['correlations']
        
        # Generate report
        report_generator = ReportGenerator(evaluator)
        html_content = report_generator.generate_single_run_report(
            run_params=run_params,
            points=points,
            pareto_points=pareto_points,
            rules=rules,
            correlations=correlations
        )
        
        return JsonResponse({
            "status": "success",
            "report_content": html_content,
            "loaded_from_backup": True
        })
        
    except Exception as e:
        print(f"Error in get_previous_run_report: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
    
@api_view(["GET"])
def list_backup_files(request):
    """
    List all available backup CSV files and PISTIL run directories.
    Mirrors the original list_backup_files from views.py [1].
    """
    try:
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        
        files_cascade = []
        files_pistil = []
        
        # Scan CASCADE backup files
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.startswith('points_backup_') and filename.endswith('.csv'):
                    timestamp_str = filename.replace('points_backup_', '').replace('.csv', '')
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        display_name = timestamp.strftime("%B %d, %Y at %H:%M:%S")
                    except ValueError:
                        display_name = filename
                        timestamp = datetime.min
                    
                    file_path = os.path.join(results_dir, filename)
                    file_size = os.path.getsize(file_path)
                    
                    # Count points
                    point_count = 0
                    try:
                        with open(file_path, 'r') as f:
                            point_count = sum(1 for row in csv.reader(f) if row)
                    except Exception:
                        pass
                    
                    files_cascade.append({
                        'filename': filename,
                        'display_name': display_name,
                        'timestamp': timestamp.isoformat(),
                        'file_size': file_size,
                        'point_count': point_count,
                        'type': 'cascade',
                    })
        
        # Scan PISTIL run directories
        pistil_root = Path(sys.path[0]) / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean" / "dse" / "results"
        if pistil_root.exists():
            for run_dir in pistil_root.iterdir():
                if run_dir.is_dir() and run_dir.name.startswith('pistil_run_'):
                    points_file = run_dir / 'points.csv'
                    if points_file.exists():
                        timestamp_str = run_dir.name.replace('pistil_run_', '')
                        try:
                            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                            display_name = f"PISTIL - {timestamp.strftime('%B %d, %Y at %H:%M:%S')}"
                        except ValueError:
                            display_name = run_dir.name
                            timestamp = datetime.min
                        
                        point_count = 0
                        try:
                            with open(points_file, 'r') as f:
                                reader = csv.reader(f)
                                next(reader, None)  # skip header
                                point_count = sum(1 for row in reader if row)
                        except Exception:
                            pass
                        
                        files_pistil.append({
                            'filename': run_dir.name,
                            'display_name': display_name,
                            'timestamp': timestamp.isoformat(),
                            'point_count': point_count,
                            'type': 'pistil',
                        })
        
        # Merge and sort newest first
        file_list = files_cascade + files_pistil
        file_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return JsonResponse({'status': 'success', 'backup_files': file_list})
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)


@api_view(["POST"])
def load_previous_run(request):
    """
    Load a previous run from a backup file or database.
    Mirrors the original load_previous_run from views.py [1].
    """
    try:
        data = json.loads(request.body)
        backup_filename = data.get('backup_filename')
        
        if not backup_filename:
            return JsonResponse({"status": "error", "message": "backup_filename is required"}, status=400)
        
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        
        # Check database first
        try:
            optimization_run = OptimizationRun.objects.filter(run_id=backup_filename).first()
            if optimization_run:
                points = []
                for dp in optimization_run.design_points.all():
                    points.append({
                        'x': dp.execution_time_ms,
                        'y': dp.energy_mj,
                        'gpu': dp.gpu_count,
                        'attn': dp.attention_count,
                        'sparse': dp.sparse_count,
                        'conv': dp.convolution_count,
                        'algorithm': optimization_run.get_algorithm_display(),
                    })
                
                run_id = f"loaded_run_{optimization_run.run_id}"
                
                # Write temp CSV for chart polling compatibility
                temp_path = os.path.join(results_dir, f"temp_points_{run_id}.csv")
                with open(temp_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    for p in points:
                        writer.writerow([p['x'], p['y'], p['gpu'], p['attn'], p['sparse'], p['conv']])
                
                return JsonResponse({
                    "status": "success",
                    "data": points,
                    "metadata": {
                        'algorithm': optimization_run.get_algorithm_display(),
                        'model': optimization_run.model,
                        'objectives': optimization_run.objectives,
                        'population_size': optimization_run.population_size,
                        'generations': optimization_run.generations,
                        'trace_name': optimization_run.trace_name,
                    },
                    "run_id": run_id,
                    "source": "database"
                })
        except Exception as e:
            print(f"DB lookup failed: {e}")
        
        # Fall back to backup file
        # Handle both CASCADE and PISTIL
        if backup_filename.startswith('pistil_run_'):
            pistil_root = Path(sys.path[0]) / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean" / "dse" / "results"
            backup_path = pistil_root / backup_filename / "points.csv"
        else:
            backup_path = os.path.join(results_dir, backup_filename)
        
        if not os.path.exists(str(backup_path)):
            return JsonResponse({"status": "error", "message": f"File not found: {backup_filename}"}, status=404)
        
        timestamp_str = str(backup_filename).replace("points_backup_", "").replace(".csv", "").replace("pistil_run_", "")
        run_id = f"loaded_run_{timestamp_str}"
        
        loader = PointsLoader('pistil' if backup_filename.startswith('pistil_run_') else 'cascade')
        points = loader.load_points_as_dicts(str(backup_path))
        
        # Write temp CSV
        temp_path = os.path.join(results_dir, f"temp_points_{run_id}.csv")
        with open(temp_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for p in points:
                writer.writerow([p['x'], p['y'], p.get('gpu', 0), p.get('attn', 0), p.get('sparse', 0), p.get('conv', 0)])
        
        # Load zip metadata if available
        run_data_loader = RunDataLoader('cascade')
        metadata = run_data_loader.load_run_metadata(str(backup_filename))
        
        return JsonResponse({
            "status": "success",
            "data": points,
            "metadata": metadata,
            "run_id": run_id,
            "source": "file"
        })
        
    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)