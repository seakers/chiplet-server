"""
Optimization-related API endpoints.
Consolidates optimization endpoints from views.py [1].
"""
import os
import sys
import json
import csv
import numpy as np
from datetime import datetime

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.config.settings import EVALUATOR_BASE_PATHS
from api.data.loaders import PointsLoader
from api.Evaluator.gaCascade import runGACascade, runSingleCascade


def convert_ndarrays(obj):
    """
    Recursively convert numpy arrays to lists for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj


@api_view(["POST"])
def run_optimization(request):
    """
    Run an optimization based on provided parameters.
    """
    try:
        data = json.loads(request.body)
        
        algorithm = data.get('algorithm', 'Genetic Algorithm')
        model = data.get('model', 'CASCADE')
        trace = data.get('trace', 'gpt-j-65536-weighted')
        population_size = data.get('population_size', 50)
        generations = data.get('generations', 100)
        objectives = data.get('objectives', ['energy', 'time'])
        
        print(f"[run_optimization] Starting {algorithm} on {model} with trace {trace}")
        
        if algorithm == 'Genetic Algorithm':
            result = runGACascade(
                pop_size=population_size,
                n_gen=generations,
                trace=trace
            )
            result = convert_ndarrays(result)
            
            return JsonResponse({
                "status": "success",
                "message": f"Optimization completed using {algorithm}",
                "data": result
            })
        else:
            return JsonResponse({
                "status": "error",
                "message": f"Unknown algorithm: {algorithm}"
            }, status=400)
            
    except Exception as e:
        print(f"[run_optimization] Error: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@api_view(["GET"])
def evaluate_point_inputs(request):
    """
    Evaluate a point using the DataGenerator.
    Refactored from views.py [1].
    """
    print("Made it to the evaluate_point_inputs function")
    try:
        trace = request.GET.get("trace", "")
        
        # Provide a default trace if none is provided
        if not trace:
            trace = "gpt-j-65536-weighted"
            print(f"No trace provided, using default: {trace}")

        chiplets = {
            "Attention": int(request.GET.get("Attention", 0)),
            "GPU": int(request.GET.get("GPU", 0)),
            "Sparse": int(request.GET.get("Sparse", 0)),
            "Convolution": int(request.GET.get("Convolution", 0)),
        }

        print("Chiplets received:", chiplets)
        print("Trace:", trace)
        print("Total chiplets:", sum(chiplets.values()))

        # Evaluate the point and get the objectives (x, y coordinates)
        # Save custom points to CSV so they persist like GA points
        objectives = runSingleCascade(chiplets, trace, save_to_csv=True)
        
        print("Objectives returned from runSingleCascade:", objectives)
        
        # Return just the newly evaluated point
        evaluated_point = {
            "x": objectives[0],
            "y": objectives[1],
            "gpu": chiplets["GPU"],
            "attn": chiplets["Attention"],
            "sparse": chiplets["Sparse"],
            "conv": chiplets["Convolution"],
            "algorithm": "Custom",
            "trace": trace,
        }
        
        return Response({
            "evaluated_point": evaluated_point,
            "message": "Point evaluated successfully"
        })
        
    except Exception as e:
        print(f"Error in evaluate_point_inputs: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def get_chart_data(request):
    """
    Get chart data for visualization.
    Refactored from views.py [1].
    """
    model = request.GET.get("model", "CASCADE").upper()
    run_id = request.GET.get("run_id")
    file_path = request.GET.get("file_path")
    comparative = request.GET.get("comparative", "false").lower() == "true"
    requested_algorithm = request.GET.get("algorithm")
    
    # Determine algorithm to use
    algorithm_to_use = "Genetic Algorithm"  # Default
    
    if run_id:
        try:
            from api.models import OptimizationRun
            optimization_run = OptimizationRun.objects.get(run_id=run_id)
            algorithm_to_use = optimization_run.get_algorithm_display()
            print(f"[get_chart_data] Using algorithm from DB: {algorithm_to_use}")
        except Exception as e:
            print(f"[get_chart_data] Could not find run {run_id}: {e}")
            if requested_algorithm:
                algorithm_to_use = requested_algorithm
    
    if model == "CASCADE":
        loader = PointsLoader('cascade', run_id)
        
        if file_path:
            points_csv_path = file_path
        else:
            WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
            points_csv_path = os.path.join(WORKSPACE, 'dse/results/points.csv')
        
        if not os.path.exists(points_csv_path):
            return Response({"data": []})
        
        points = loader.load_points_as_dicts(points_csv_path)
        
        # Add algorithm and trace info to each point
        for point in points:
            point['algorithm'] = algorithm_to_use
            point['trace'] = request.GET.get("trace", "gpt-j-65536-weighted")
        
        return Response({"data": points})
    
    elif model == "PISTIL":
        # Handle PISTIL model
        loader = PointsLoader('pistil', run_id)
        
        if file_path:
            points_csv_path = file_path
        elif run_id and run_id.startswith("pistil_run_"):
            from pathlib import Path
            pistil_root = Path(__file__).parent / "Evaluator" / "sim-v2-4-pistil-sim-clean"
            output_dir = pistil_root / "dse" / "results" / run_id
            points_csv_path = os.path.join(str(output_dir), "points.csv")
        else:
            return Response({"data": []})
        
        if not os.path.exists(points_csv_path):
            return Response({"data": []})
        
        points = loader.load_points_as_dicts(points_csv_path)
        
        for point in points:
            point['algorithm'] = algorithm_to_use
            point['trace'] = request.GET.get("trace", "pistil-default")
        
        return Response({"data": points})
    
    return Response({"data": []})