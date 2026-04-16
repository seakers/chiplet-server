"""
Design highlighting API endpoints.
Consolidates highlighting/constraint endpoints from views.py <source_id data="1" title="views.py" />.
"""
import os
import csv

from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.config.settings import EVALUATOR_BASE_PATHS
from api.data.loaders import PointsLoader
from api.analysis.constraints import ConstraintMatcher


@api_view(["GET"])
def get_designs_by_constraint(request):
    """
    Get designs that match specified constraint criteria.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        chiplet_type = request.GET.get("chiplet_type", "gpu")
        constraint_level = request.GET.get("constraint_level", "none")
        evaluator = request.GET.get("evaluator", "CASCADE")
        run_id = request.GET.get("run_id")
        
        print(f"[get_designs_by_constraint] chiplet={chiplet_type}, level={constraint_level}")
        
        # Load points
        loader = PointsLoader(evaluator, run_id)
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        points = loader.load_points_as_dicts(file_path)
        
        if not points:
            return Response({"matching_designs": [], "total_count": 0})
        
        # Get matching designs using ConstraintMatcher
        matcher = ConstraintMatcher(evaluator)
        
        try:
            matching_designs = matcher.get_matching_designs(
                points, chiplet_type, constraint_level
            )
        except ValueError as e:
            return Response({"error": str(e)}, status=400)
        
        return Response({
            "matching_designs": matching_designs,
            "total_count": len(matching_designs),
            "chiplet_type": chiplet_type,
            "constraint_level": constraint_level
        })
        
    except Exception as e:
        print(f"Error in get_designs_by_constraint: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def get_highlighted_points(request):
    """
    Get all points with highlighting information based on constraint criteria.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        chiplet_type = request.GET.get("chiplet_type", "gpu")
        constraint_level = request.GET.get("constraint_level", "none")
        evaluator = request.GET.get("evaluator", "CASCADE")
        run_id = request.GET.get("run_id")
        
        # Load points
        loader = PointsLoader(evaluator, run_id)
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        points = loader.load_points_as_dicts(file_path)
        
        if not points:
            return Response({"points": [], "highlighted_count": 0})
        
        # Get highlighted points
        matcher = ConstraintMatcher(evaluator)
        
        try:
            highlighted_points = matcher.get_highlighted_points(
                points, chiplet_type, constraint_level
            )
        except ValueError as e:
            return Response({"error": str(e)}, status=400)
        
        highlighted_count = sum(1 for p in highlighted_points if p.get('highlighted'))
        
        return Response({
            "points": highlighted_points,
            "highlighted_count": highlighted_count,
            "total_count": len(highlighted_points)
        })
        
    except Exception as e:
        print(f"Error in get_highlighted_points: {e}")
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def get_point_context(request):
    """
    Retrieve detailed point context JSON for a specific design point.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        # Get parameters
        run_id = request.GET.get("run_id")
        design = request.GET.get("design")
        gpu = request.GET.get("gpu", "0")
        attn = request.GET.get("attn", "0")
        sparse = request.GET.get("sparse", "0")
        conv = request.GET.get("conv", "0")
        
        if not run_id:
            return Response({"error": "run_id is required"}, status=400)
        
        # Construct the file path
        base_path = "/chiplet-server/api/Evaluator/cascade/chiplet_model/dse/results"
        
        # Construct candidate run directories to handle different formats
        candidates = []
        if run_id.startswith("loaded_run_"):
            mapped = run_id.replace("loaded_run_", "myrun_")
            if "_" in mapped:
                parts = mapped.split("_")
                if len(parts) >= 3:
                    date_part = parts[1]
                    time_part = parts[2]
                    mapped = f"myrun_{date_part}_{time_part}"
            candidates.append(mapped)
            candidates.append(run_id)
            candidates.append(f"temp_points_{run_id}")
        else:
            candidates.append(f"myrun_{run_id}")
            candidates.append(run_id)
            if run_id.startswith("restarted_run_"):
                candidates.append(run_id)
        
        # Construct the context file name
        context_filename = (
            f"{design}.json" if design else f"{gpu}gpu{attn}attn{sparse}sparse{conv}conv.json"
        )
        
        found_path = None
        tried = []
        for run_dir in candidates:
            candidate_path = os.path.join(base_path, run_dir, "pointContext", context_filename)
            tried.append(candidate_path)
            if os.path.exists(candidate_path):
                found_path = candidate_path
                break
        
        print(f"Looking for point context file, tried: {tried}")
        
        if not found_path:
            available_designs = []
            for run_dir in candidates:
                pc_dir = os.path.join(base_path, run_dir, "pointContext")
                if os.path.isdir(pc_dir):
                    try:
                        for fname in os.listdir(pc_dir):
                            if fname.endswith('.json'):
                                available_designs.append(fname[:-5])
                    except Exception:
                        pass
            
            return Response({
                "error": "Point context file not found",
                "tried_paths": tried,
                "run_id": run_id,
                "chiplet_config": design or f"{gpu}gpu{attn}attn{sparse}sparse{conv}conv",
                "available_designs": sorted(list(set(available_designs)))
            }, status=404)
        
        # Read and return the JSON file
        import json
        with open(found_path, 'r') as f:
            context_data = json.load(f)
        
        return Response({
            "message": "Point context retrieved successfully",
            "context": context_data,
            "file_path": found_path
        })
        
    except Exception as e:
        print(f"Error in get_point_context: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def get_kernel_breakdown(request):
    """
    Extract kernel breakdown from point context file.
    Refactored from views.py <source_id data="1" title="views.py" />.
    """
    try:
        run_id = request.GET.get("run_id")
        design = request.GET.get("design")
        gpu = request.GET.get("gpu", "0")
        attn = request.GET.get("attn", "0")
        sparse = request.GET.get("sparse", "0")
        conv = request.GET.get("conv", "0")
        output_format = request.GET.get("format", "json")
        evaluator = request.GET.get("evaluator", "cascade")
        
        if not run_id:
            return Response({"error": "run_id is required"}, status=400)
        
        # Construct the file path
        base_path = "/chiplet-server/api/Evaluator/cascade/chiplet_model/dse/results"
        
        # Build candidate paths
        candidates = []
        if run_id.startswith("loaded_run_"):
            mapped = run_id.replace("loaded_run_", "myrun_")
            candidates.append(mapped)
            candidates.append(run_id)
        else:
            candidates.append(f"myrun_{run_id}")
            candidates.append(run_id)
        
        context_filename = (
            f"{design}.json" if design else f"{gpu}gpu{attn}attn{sparse}sparse{conv}conv.json"
        )
        
        found_path = None
        for run_dir in candidates:
            candidate_path = os.path.join(base_path, run_dir, "pointContext", context_filename)
            if os.path.exists(candidate_path):
                found_path = candidate_path
                break
        
        if not found_path:
            return Response({"error": "Point context file not found"}, status=404)
        
        # Use ChatBot to extract kernel breakdown
        from api.chatbot.bot import ChatBot
        chat_bot = ChatBot(evaluator=evaluator, run_id=run_id)
        breakdown = chat_bot.extract_kernel_breakdown(
            context_file_path=found_path, 
            output_format=output_format
        )
        
        if breakdown is None:
            return Response({"error": "Failed to extract kernel breakdown"}, status=500)
        
        if output_format == 'csv' and isinstance(breakdown, str):
            return Response({
                "message": "Kernel breakdown saved to CSV",
                "file_path": breakdown,
                "breakdown": None
            })
        
        return Response({
            "message": "Kernel breakdown extracted successfully",
            "breakdown": breakdown,
            "file_path": found_path
        })
        
    except Exception as e:
        print(f"Error in get_kernel_breakdown: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)