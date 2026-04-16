"""
Analysis-related API endpoints.
Consolidates rule_mining and distance_correlation endpoints from views.py [1].
"""
import os
import re
import time
import json
import csv
import numpy as np
import dcor

from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.config.settings import get_points_file, EVALUATOR_BASE_PATHS
from api.config.evaluators import get_evaluator_config
from api.data.loaders import PointsLoader
from api.analysis.pareto import ParetoCalculator
from api.analysis.rule_mining import RuleMiner, RuleFormatter
from api.analysis.distance_correlation import DistanceCorrelationAnalyzer
from api.chatbot.bot import ChatBot


@api_view(["GET"])
def rule_mining(request):
    """
    Run rule mining and return the results in a structured format for the frontend table.
    Refactored to use centralized analysis classes [1].
    """
    print("[rule_mining] Called rule_mining endpoint.")
    start_time = time.time()
    
    # Get parameters from request
    region = request.GET.get("region", "pareto")
    pareto_start_rank = int(request.GET.get("paretoStartRank", 1))
    pareto_end_rank = int(request.GET.get("paretoEndRank", 3))
    energy_min = request.GET.get("energyMin")
    energy_max = request.GET.get("energyMax")
    time_min = request.GET.get("timeMin")
    time_max = request.GET.get("timeMax")
    evaluator = request.GET.get("evaluator", "cascade")
    run_id = request.GET.get("run_id")
    
    print(f"[rule_mining] Parameters: region={region}, pareto_ranks={pareto_start_rank}-{pareto_end_rank}")
    
    try:
        # Create point selection parameters
        point_selection_params = {
            "region": region,
            "pareto_start_rank": pareto_start_rank,
            "pareto_end_rank": pareto_end_rank,
            "energy_min": float(energy_min) if energy_min else None,
            "energy_max": float(energy_max) if energy_max else None,
            "time_min": float(time_min) if time_min else None,
            "time_max": float(time_max) if time_max else None,
        }
        
        # Use ChatBot for rule mining (maintains compatibility)
        chat_bot = ChatBot(evaluator=evaluator, run_id=run_id)
        rule_mining_str = chat_bot.rule_mining(point_selection_params)
        
        # Parse the rule_mining_str into a list of dicts for the frontend
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
        
        elapsed = time.time() - start_time
        print(f"[rule_mining] Returning {len(rules)} rules. Time taken: {elapsed:.2f} seconds.")
        return Response({"rules": rules, "elapsed": elapsed})
        
    except Exception as e:
        print(f"[rule_mining] Exception: {e}")
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def distance_correlation(request):
    """
    Compute distance correlation between each chiplet/design variable and objectives.
    Refactored to use centralized analysis classes [1].
    """
    print("Running Distance Correlation Views")
    evaluator = request.GET.get("evaluator", "CASCADE")
    run_id = request.GET.get("run_id", None)
    
    try:
        # Get evaluator config
        config = get_evaluator_config(evaluator)
        
        # Construct file path based on evaluator
        if evaluator.lower() == 'pistil':
            if not run_id:
                return Response({
                    "error": "run_id is required for PISTIL distance correlation"
                }, status=400)
            file_path = f'api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/{run_id}/points.csv'
        elif evaluator.lower() == 'cascade':
            file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        else:
            return Response({"error": f"Unknown evaluator: {evaluator}"}, status=400)
        
        if not os.path.exists(file_path):
            return Response({"error": f"Points file not found: {file_path}"}, status=404)
        
        print(f"[distance_correlation] Evaluator: {evaluator}, Using file: {file_path}")
        
        # Load data using PointsLoader
        loader = PointsLoader(evaluator, run_id)
        points = loader.load_points_as_dicts(file_path)
        
        if not points:
            return Response({"error": "No valid data available for analysis"}, status=400)
        
        # Use DistanceCorrelationAnalyzer
        analyzer = DistanceCorrelationAnalyzer(evaluator)
        results = analyzer.analyze_from_points(points)
        
        print(f"[distance_correlation] Computed {len(results['correlations'])} correlations")
        return Response(results['correlations'])
        
    except Exception as e:
        import traceback
        print(f"[distance_correlation] Exception: {traceback.format_exc()}")
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def distance_correlation_insights(request):
    """
    Compute distance correlation and send to LLM for meaningful insights analysis.
    Refactored to use centralized analysis classes [1].
    """
    try:
        evaluator = request.GET.get("evaluator", "CASCADE")
        run_id = request.GET.get("run_id", None)
        objective = request.GET.get("objective", "both")
        trace_name = request.GET.get("trace_name", "Unknown")
        
        # Get evaluator config
        config = get_evaluator_config(evaluator)
        
        # Construct file path
        if evaluator.lower() == 'pistil':
            if not run_id:
                return Response({"error": "run_id is required for PISTIL"}, status=400)
            file_path = f'api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/{run_id}/points.csv'
        else:
            file_path = request.GET.get(
                "file_path", 
                "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
            )
        
        if not os.path.exists(file_path):
            return Response({"error": f"Points file not found: {file_path}"}, status=404)
        
        # Load and analyze data
        loader = PointsLoader(evaluator, run_id)
        points = loader.load_points_as_dicts(file_path)
        
        if not points:
            return Response({"error": "No valid data available for analysis"}, status=400)
        
        analyzer = DistanceCorrelationAnalyzer(evaluator)
        results = analyzer.analyze_from_points(points)
        correlations = results['correlations']
        
        # Determine objective labels
        if evaluator.lower() == "cascade":
            obj0_label = "Time"
            obj1_label = "Energy"
            design_element = "chiplet types"
        else:
            obj0_label = "Latency"
            obj1_label = "Energy"
            design_element = "design parameters"
        
        # Create structured JSON data for UI display
        energy_correlations = {k: v for k, v in correlations.items() if obj1_label in k}
        time_correlations = {k: v for k, v in correlations.items() if obj0_label in k}
        
        # Sort by correlation value (descending)
        energy_sorted = sorted(
            energy_correlations.items(),
            key=lambda x: (-float('inf') if x[1] is None else x[1]),
            reverse=True
        )
        time_sorted = sorted(
            time_correlations.items(),
            key=lambda x: (-float('inf') if x[1] is None else x[1]),
            reverse=True
        )
        
        structured_data = {
            "high_impact_on_energy": [
                {
                    "variable": var_metric.split('_vs_')[0],
                    "correlation": (round(value, 3) if value is not None else None)
                }
                for var_metric, value in energy_sorted
            ],
            "high_impact_on_time": [
                {
                    "variable": var_metric.split('_vs_')[0],
                    "correlation": (round(value, 3) if value is not None else None)
                }
                for var_metric, value in time_sorted
            ],
            "trace_name": trace_name,
            "objective": objective,
            "evaluator": evaluator,
            "run_id": run_id
        }
        
        # Create goal-aware prompt for LLM
        if objective == "energy":
            goal_text = "minimize energy consumption"
            focus_metric = "energy"
        elif objective == "time":
            goal_text = f"minimize {obj0_label.lower()}"
            focus_metric = obj0_label.lower()
        else:
            goal_text = f"optimize both energy and {obj0_label.lower()}"
            focus_metric = "both metrics"
        
        # Get LLM insights
        chat_bot = ChatBot(evaluator=evaluator, run_id=run_id)
        
        prompt = (
            f"You are an expert in hardware design using the {evaluator} evaluator. "
            f"The current design goal is to {goal_text}.\n\n"
            f"Trace used: {trace_name}\n\n"
            f"Here are distance correlation results showing the relationship between {design_element} and performance metrics.\n\n"
            f"JSON Data:\n{json.dumps(structured_data, indent=2)}\n\n"
            f"Provide a concise, actionable summary (2-3 sentences) covering:\n"
            f"1. Which {design_element} have the strongest impact on {focus_metric} and why\n"
            f"2. One practical design recommendation for improving performance\n"
            f"3. Any surprising findings (if any)\n\n"
            f"Keep your response focused and to the point."
        )
        
        ai_insights = chat_bot.get_response(prompt)
        
        return Response({
            "correlations": correlations,
            "structured_data": structured_data,
            "ai_insights": ai_insights,
            "raw_summary": results['summary']
        })
        
    except Exception as e:
        import traceback
        print(f"[distance_correlation_insights] Exception: {traceback.format_exc()}")
        return Response({"error": str(e)}, status=500)