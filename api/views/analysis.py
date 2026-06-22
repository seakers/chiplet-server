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
from api.config.objectives import DEFAULT_OBJECTIVES, to_fields
from django.views.decorators.clickjacking import xframe_options_exempt
from django.utils.decorators import method_decorator


@api_view(["GET"])
def rule_mining(request):
    try:
        region = request.GET.get("region", "pareto")
        pareto_start_rank = int(request.GET.get("paretoStartRank", 1))
        pareto_end_rank   = int(request.GET.get("paretoEndRank", 3))
        evaluator = request.GET.get("evaluator", "cascade")
        run_id    = request.GET.get("run_id")

        # New: explicit objective list from frontend
        objectives_param = request.GET.get("objectives")
        requested_objectives = (
            [o.strip() for o in objectives_param.split(",")]
            if objectives_param else None
        )
        # Fallback: read obj0_name / obj1_name (legacy)
        if not requested_objectives:
            legacy = []
            for i in range(3):
                name = request.GET.get(f"obj{i}_name")
                if name:
                    legacy.append(name)
            if legacy:
                requested_objectives = legacy

        # Build obj_ranges aligned with requested_objectives
        obj_ranges = []
        names_for_ranges = requested_objectives or DEFAULT_OBJECTIVES.get(evaluator.lower(), [])
        for i, name in enumerate(names_for_ranges):
            min_val = request.GET.get(f"obj{i}_min")
            max_val = request.GET.get(f"obj{i}_max")
            obj_ranges.append({
                "name": name,
                "min": float(min_val) if min_val not in (None, "") else None,
                "max": float(max_val) if max_val not in (None, "") else None,
            })

        # File path
        if evaluator.lower() == "pistil":
            if not run_id:
                return Response({"error": "run_id is required for PISTIL"}, status=400)
            file_path = f"api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/{run_id}/points.csv"
        else:
            file_path = request.GET.get(
                "file_path",
                "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
            )

        if not os.path.exists(file_path):
            return Response({"error": f"Points file not found: {file_path}"}, status=404)

        # Load + run miner
        loader = PointsLoader(evaluator, run_id)
        data = loader.load_deduplicated_data(file_path)
        if len(data) == 0:
            return Response({"error": "No valid data for analysis"}, status=400)

        # Compute objective column indices into data
        cfg = get_evaluator_config(evaluator)
        all_obj_cols = cfg.objective_columns
        objective_col_indices = None
        if requested_objectives:
            wanted_fields = to_fields(requested_objectives)
            objective_col_indices = [cfg.get_objective_index(f) for f in wanted_fields if f in all_obj_cols]

        from api.analysis.rule_mining import RuleMiner
        miner = RuleMiner(evaluator)
        rules = miner.mine_rules(
            data,
            max_pareto_rank=pareto_end_rank,
            objectives=requested_objectives,
            objective_col_indices=objective_col_indices,
        )

        return Response({
            "rules": miner.get_rules_as_dicts(rules),
            "objectives": requested_objectives or names_for_ranges,
            "region": region,
            "pareto_range": [pareto_start_rank, pareto_end_rank],
        })

    except Exception as e:
        import traceback; traceback.print_exc()
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
        objectives_param = request.GET.get("objectives")  # e.g. "Energy,Runtime"
        requested_objectives = [o.strip() for o in objectives_param.split(",")] if objectives_param else None
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
        results = analyzer.analyze_from_points(points, requested_objectives=requested_objectives)
        
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
        trace_name = request.GET.get("trace_name", "Unknown")
        objectives_param = request.GET.get("objectives")
        requested_objectives = [o.strip() for o in objectives_param.split(",")] if objectives_param else None
        
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
        results = analyzer.analyze_from_points(points, requested_objectives=requested_objectives)
        correlations = results['correlations']
        
        # Determine design element
        if evaluator.lower() == "cascade":
            design_element = "chiplet types"
        else:
            design_element = "design parameters"
        
        # Get objective labels
        if requested_objectives:
            objectives_to_use = requested_objectives
            goal_text = f"optimize {' and '.join(o.lower() for o in requested_objectives)}"
            focus_metric = " and ".join(o.lower() for o in requested_objectives)
        else:
            if evaluator.lower() == "cascade":
                objectives_to_use = ["Time", "Energy"]
            else:
                objectives_to_use = ["Latency", "Energy"]
            goal_text = f"optimize {' and '.join(o.lower() for o in objectives_to_use)}"
            focus_metric = " and ".join(o.lower() for o in objectives_to_use)
        
        # Create structured JSON data for UI display
        # Group correlations by each objective
        objective_correlations = {}
        for obj in objectives_to_use:
            objective_correlations[obj] = {k: v for k, v in correlations.items() if obj in k}
        
        # Sort each objective's correlations by value (descending)
        sorted_objectives = {}
        for obj, obj_corrs in objective_correlations.items():
            sorted_objectives[obj] = sorted(
                obj_corrs.items(),
                key=lambda x: (-float('inf') if x[1] is None else x[1]),
                reverse=True
            )
        
        # Build high_impact sections for each objective
        structured_data = {
            "trace_name": trace_name,
            "objectives": requested_objectives or DEFAULT_OBJECTIVES.get(evaluator.lower(), []),
            "evaluator": evaluator,
            "run_id": run_id,
            "high_impact": results['high_impact'],
            "correlations": correlations
        }
        
        
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
    

@api_view(["GET"])
def rule_mining_insights(request):
    try:
        evaluator = request.GET.get("evaluator", "cascade")
        run_id    = request.GET.get("run_id")
        region    = request.GET.get("region", "pareto")
        pareto_start_rank = int(request.GET.get("paretoStartRank", 1))
        pareto_end_rank   = int(request.GET.get("paretoEndRank", 3))
        trace_name = request.GET.get("trace_name", "Unknown")
        objective  = request.GET.get("objective", "both")

        objectives_param = request.GET.get("objectives")
        requested_objectives = (
            [o.strip() for o in objectives_param.split(",")]
            if objectives_param
            else DEFAULT_OBJECTIVES.get(evaluator.lower(), [])
        )

        # Build per-objective ranges aligned with requested_objectives
        obj_ranges = []
        for i, name in enumerate(requested_objectives):
            min_v = request.GET.get(f"obj{i}_min")
            max_v = request.GET.get(f"obj{i}_max")
            obj_ranges.append({
                "name": name,
                "min":  float(min_v) if min_v not in (None, "") else None,
                "max":  float(max_v) if max_v not in (None, "") else None,
            })

        point_selection_params = {
            "region": region,
            "pareto_start_rank": pareto_start_rank,
            "pareto_end_rank": pareto_end_rank,
            "obj_ranges": obj_ranges,
        }

        chat_bot = ChatBot(evaluator=evaluator, run_id=run_id)
        chat_bot.set_objectives(requested_objectives)
        rule_mining_str = chat_bot.rule_mining(point_selection_params)

        goal_text = f"optimize {' and '.join(o.lower() for o in requested_objectives)}"

        # Build structured data for LLM prompt
        structured_data = {
            "evaluator": evaluator,
            "objectives": requested_objectives,
            "region": region,
            "rules_text": rule_mining_str,
        }

        prompt = (
            f"You are a chiplet design analyst. A rule mining analysis was run on "
            f"Pareto-optimal points for the goal: {goal_text}.\n\n"
            f"Trace used: {trace_name}\n\n"
            f"Here are the rules extracted:\n"
            f"JSON Data:\n{json.dumps(structured_data, indent=2)}\n\n"
            f"Provide a concise, actionable summary (2-3 sentences) covering:\n"
            f"1. The most important recurring pattern in optimal designs\n"
            f"2. One specific recommendation for chiplet combination\n"
            f"3. Any rule conflicts or redundancies (if any)\n\n"
            f"Keep your response focused and to the point."
        )
        response = chat_bot.get_response(prompt)

        return Response({
            "insights": response,
            "structured_data": structured_data,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def serve_pdf(request):
    """
    Serve a PDF file from the PISTIL gen_configs directory.
    """
    base = os.path.abspath(
        "/home/snagg/chiplet-server/api/Evaluator/sim-v2-4-pistil-sim-clean/configs/gen_configs"
    )
    path = request.GET.get("path", "")

    if not path:
        return Response({"error": "No path provided"}, status=400)

    full_path = os.path.abspath(path)

    # print(f"Resolved PDF path: {full_path}")

    if not full_path.startswith(base):
        return Response({"error": "Access denied"}, status=403)

    if not os.path.exists(full_path):
        return Response({"error": f"File not found: {full_path}"}, status=404)

    from django.http import FileResponse
    response = FileResponse(open(full_path, "rb"), content_type="application/pdf")
    response["Content-Disposition"] = "inline"
    return response