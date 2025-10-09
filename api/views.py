from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .services.run_storage import RunStorageService
from .models import Task
from .serializers import TaskSerializer

import random
from api.Evaluator.generator import DataGenerator
from api.ChatBot.model import ChatBotModel
import dcor
import numpy as np
import re
import pandas as pd
from datetime import datetime

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import sys
import os
import threading
import traceback

from api.Evaluator.gaCascade import runGACascade, runSingleCascade
from api.Evaluator.generator import generate_weighted_trace
from api.Evaluator.evaluator import evaluate_custom_design
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from api.Evaluator.cascade.chiplet_model.dse.lib.trace_parser import TraceParser
from api.Evaluator.cascade.chiplet_model.dse.lib.chiplet_system import ChipletSystem
from api.Evaluator.gaCascade import CascadeProblem
from api.services.run_storage import RunStorageService
from api.models import OptimizationRun

# ===================== RESTART RUN ENDPOINT =====================
@api_view(["POST"])
@csrf_exempt
def restart_run(request):
    try:
        data = json.loads(request.body)
        backup_filename = data.get('backup_filename')
        generations = int(data.get('generations', 10))
        traces = data.get('traces', [])

        if not backup_filename:
            return JsonResponse({"status": "error", "message": "backup_filename is required"}, status=400)

        # Resolve paths
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        source_file = os.path.join(results_dir, backup_filename)

        if not os.path.exists(source_file):
            return JsonResponse({"status": "error", "message": f"Backup file not found: {backup_filename}"}, status=404)

        # Create new run folder
        run_id = f"restarted_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(results_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'pointContext'), exist_ok=True)

        # Create temporary file for new GA points only
        temp_new_points = os.path.join(run_dir, 'new_points_temp.csv')
        with open(temp_new_points, 'w') as f:
            f.write('')  # Empty file for new points

        # Keep original points in a separate file for reference
        original_points = os.path.join(run_dir, 'original_points.csv')
        try:
            with open(source_file, 'r') as src, open(original_points, 'w') as dst:
                for line in src:
                    dst.write(line.strip())
                    if not line.endswith('\n'):
                        dst.write('\n')
        except Exception as e:
            return JsonResponse({"status": "error", "message": f"Failed to copy original points: {str(e)}"}, status=500)

        # Start GA in background, seeded from previous designs
        def run_seeded_ga():
            try:
                # Use weighted trace if available; default to gpt-j-65536-weighted
                trace_name = 'gpt-j-65536-weighted'
                if isinstance(traces, list) and len(traces) > 0 and isinstance(traces[0], dict) and traces[0].get('name'):
                    trace_name = traces[0]['name']

                # Build initial population from previous points' decisions (cols 2..5 -> 4 chiplet counts, repeat to 12 vars)
                import csv
                decisions = []
                try:
                    with open(source_file, 'r') as f:
                        reader = csv.reader(f)
                        for row in reader:
                            if len(row) >= 6:
                                gpu, attn, sparse, conv = int(float(row[2])), int(float(row[3])), int(float(row[4])), int(float(row[5]))
                                # Map 4 counts into 12 decision vars by repeating each count 3x (simple lift)
                                indiv = [gpu, attn, sparse, conv] * 3
                                decisions.append(indiv[:12])
                except Exception:
                    pass

                from pymoo.operators.sampling.rnd import FloatRandomSampling
                import numpy as np
                initial_sampling = None
                if decisions:
                    initial_sampling = np.array(decisions, dtype=int)

                # Use population size as size of initial sampling, fallback to 50
                pop_size = len(decisions) if decisions else 50

                # Modify GA to write to temp file instead of main points.csv
                import shutil
                original_ga_output = os.path.join(run_dir, 'points.csv')
                
                # Run GA with modified output directory (it will write to points.csv)
                runGACascade(
                    pop_size=pop_size,
                    n_gen=generations,
                    trace=trace_name,
                    initial_population=initial_sampling,
                    output_dir=run_dir
                )
                
                # After GA completes, merge original + new points into final file
                final_points = os.path.join(run_dir, f'points_merged_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
                with open(final_points, 'w') as merged:
                    # Write original points first
                    with open(original_points, 'r') as orig:
                        for line in orig:
                            merged.write(line)
                    
                    # Write new GA points
                    if os.path.exists(original_ga_output):
                        with open(original_ga_output, 'r') as new:
                            for line in new:
                                merged.write(line)
                
                print(f"Restart run completed. Merged file: {final_points}")
            except Exception as e:
                print(f"restart_run background error: {e}")

        t = threading.Thread(target=run_seeded_ga, daemon=True)
        t.start()

        # Read initial points from the copied original_points.csv to send to frontend
        initial_points_data = []
        try:
            import csv
            with open(original_points, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 6:
                        initial_points_data.append({
                            'x': float(row[0]), 'y': float(row[1]),
                            'gpu': float(row[2]), 'attn': float(row[3]),
                            'sparse': float(row[4]), 'conv': float(row[5]),
                            'type': 'optimization', 'algorithm': 'Genetic Algorithm',
                            'trace': 'gpt-j-65536-weighted'
                        })
        except Exception as e:
            print(f"Error reading initial points: {e}")

        return JsonResponse({
            "status": "success",
            "run_id": run_id,
            "initial_points": initial_points_data,
            "message": "Restarted run initialized and GA started"
        })
    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

################ Instantiate the necessary objects ################
dataGenerator = DataGenerator()
chat_bot = ChatBotModel()
################ Instantiate the necessary objects ################

def generate_run_id():
    """Generate a unique run ID with timestamp"""
    timestamp = datetime.now().strftime("%Y%b%d_%H%M%S")
    return f"myrun_{timestamp}"

def create_run_directory(run_id):
    """Create directory structure for a new run"""
    import os
    WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
    run_dir = os.path.join(WORKSPACE, 'dse/results', run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

# Helper to convert all ndarrays to lists
def convert_ndarrays(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarrays(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarrays(v) for v in obj]
    else:
        return obj

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer

@api_view(["GET"])
def compute_sum(request):
    try:
        num1 = int(request.GET.get("num1", 0))
        num2 = int(request.GET.get("num2", 0))
        result = num1 + num2
        return Response({"result": result})
    except (ValueError, TypeError):
        return Response({"error": "Invalid numbers provided."}, status=400)
    
@api_view(["GET"])
def get_chart_data(request):
    print("get_chart_data called")
    comparative = request.GET.get("comparative", "false").lower() == "true"
    
    # Get file path for loaded runs
    file_path = request.GET.get("file_path")
    
    if comparative:
        # Comparative mode: read from runA_results/points.csv and runB_results/points.csv
        import os
        import csv
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        runA_path = os.path.join(WORKSPACE, 'dse/results/runA_results/points.csv')
        runB_path = os.path.join(WORKSPACE, 'dse/results/runB_results/points.csv')
        points = []
        
        print(f"Checking comparative files: {runA_path}, {runB_path}")
        
        # Check if directories exist (analysis might be in progress)
        runA_dir = os.path.dirname(runA_path)
        runB_dir = os.path.dirname(runB_path)
        
        if os.path.exists(runA_path):
            try:
                with open(runA_path, mode='r') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        if len(row) >= 6:  # Ensure we have all required columns
                            points.append({
                                'x': float(row[0]),  # time
                                'y': float(row[1]),  # energy
                                'gpu': float(row[2]),
                                'attn': float(row[3]),
                                'sparse': float(row[4]),
                                'conv': float(row[5]),
                                'type': row[6] if len(row) > 6 else 'optimization',
                                'algorithm': 'Genetic Algorithm',
                                'trace': 'gpt-j-65536-weighted',
                                'run': 'A'
                            })
                print(f"Run A data loaded: {len([p for p in points if p['run'] == 'A'])} points")
            except Exception as e:
                print(f"Error reading {runA_path}: {e}")
        else:
            print(f"Run A file not found: {runA_path}")
            if os.path.exists(runA_dir):
                print(f"Run A directory exists, analysis may be in progress")
            
        if os.path.exists(runB_path):
            try:
                with open(runB_path, mode='r') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        if len(row) >= 6:  # Ensure we have all required columns
                            points.append({
                                'x': float(row[0]),  # time
                                'y': float(row[1]),  # energy
                                'gpu': float(row[2]),
                                'attn': float(row[3]),
                                'sparse': float(row[4]),
                                'conv': float(row[5]),
                                'type': row[6] if len(row) > 6 else 'optimization',
                                'algorithm': 'Genetic Algorithm',
                                'trace': 'gpt-j-65536-weighted',
                                'run': 'B'
                            })
                print(f"Run B data loaded: {len([p for p in points if p['run'] == 'B'])} points")
            except Exception as e:
                print(f"Error reading {runB_path}: {e}")
        else:
            print(f"Run B file not found: {runB_path}")
            if os.path.exists(runB_dir):
                print(f"Run B directory exists, analysis may be in progress")
        
        print(f"Comparative chart data: {len(points)} total points")
        return Response({"data": points})
    
    # Simple mode: read from specified file path or default to points.csv
    try:
        import os
        import csv
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        
        # Use provided file path or default to points.csv
        if file_path:
            points_csv_path = file_path
            print(f"Reading from provided file path: {points_csv_path}")
        else:
            points_csv_path = os.path.join(WORKSPACE, 'dse/results/points.csv')
            print(f"Reading from default points.csv: {points_csv_path}")
        
        if os.path.exists(points_csv_path):
            points = []
            with open(points_csv_path, mode='r') as file:
                csv_reader = csv.reader(file)
                for row in csv_reader:
                    if len(row) >= 6:  # Ensure we have all required columns
                        points.append({
                            'x': float(row[0]),  # time
                            'y': float(row[1]),  # energy
                            'gpu': float(row[2]),
                            'attn': float(row[3]),
                            'sparse': float(row[4]),
                            'conv': float(row[5]),
                            'type': row[6] if len(row) > 6 else 'optimization',  # point type
                            'algorithm': 'Genetic Algorithm',
                            'trace': 'gpt-j-65536-weighted'  # Default trace
                        })
            print(f"Read {len(points)} points from {points_csv_path}")
            return Response({"data": points})
        else:
            print(f"File not found: {points_csv_path}")
            return Response({"data": []})
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return Response({"data": []})

@api_view(["GET"])
def update_data(request):
    data = dataGenerator.get_data()
    print("Data Points: ", data)
    return Response({"data": data})

@api_view(["GET"])
def get_chat_response(request):
    """
    Get response from chatbot.
    """
    content = request.GET.get("content")
    role = request.GET.get("role")
    
    # Check for report generation request
    if content and any(phrase in content.lower() for phrase in ["generate report", "download report", "get report", "create report", "export report"]):
        try:
            # Generate the report
            from .views import generate_optimization_report
            report_response = generate_optimization_report(request)
            
            if report_response.status_code == 200:
                report_data = report_response.data
                chat_reply = f"Report generated successfully! You can download it here: {report_data['download_link']}"
                return JsonResponse({"message": chat_reply})
            else:
                return JsonResponse({"message": "Sorry, I encountered an error generating the report. Please try again."})
        except Exception as e:
            return JsonResponse({"message": "Sorry, I encountered an error generating the report. Please try again."})
    
    # Check for highlighting request
    if content and any(phrase in content.lower() for phrase in ["highlight", "show", "mark", "find"]) and any(chiplet in content.lower() for chiplet in ["gpu", "attention", "sparse", "convolution"]):
        try:
            print(f"[HIGHLIGHTING] Processing request: {content}")
            
            # Parse the highlighting request
            import re
            
            # Extract chiplet type
            chiplet_type = None
            if "gpu" in content.lower():
                chiplet_type = "gpu"
            elif "attention" in content.lower():
                chiplet_type = "attention"
            elif "sparse" in content.lower():
                chiplet_type = "sparse"
            elif "convolution" in content.lower():
                chiplet_type = "convolution"
            
            # Extract constraint level
            constraint_level = None
            if any(word in content.lower() for word in ["0", "zero", "none", "no"]):
                constraint_level = "none"
            elif any(word in content.lower() for word in ["low", "few", "small"]):
                constraint_level = "low"
            elif any(word in content.lower() for word in ["medium", "moderate"]):
                constraint_level = "medium"
            elif any(word in content.lower() for word in ["high", "many", "lots"]):
                constraint_level = "high"
            
            print(f"[HIGHLIGHTING] Parsed: chiplet_type={chiplet_type}, constraint_level={constraint_level}")
            
            if chiplet_type and constraint_level:
                # Call the highlighting endpoint directly with parameters
                from .views import get_designs_by_constraint
                
                # Instead of mocking the request, let's call the function logic directly
                try:
                    # Load data from CSV
                    file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
                    import csv
                    import os
                    
                    print(f"[HIGHLIGHTING] Checking if file exists: {file_path}")
                    if not os.path.exists(file_path):
                        print(f"[HIGHLIGHTING] File does not exist: {file_path}")
                        return JsonResponse({
                            "message": "No optimization data is available for highlighting. Please run an optimization first to generate design data, then try highlighting again."
                        })
                    
                    points = []
                    with open(file_path, mode='r') as file:
                        csv_reader = csv.reader(file)
                        for row in csv_reader:
                            points.append({
                                'x': float(row[0]),  # time
                                'y': float(row[1]),  # energy
                                'gpu': float(row[2]),
                                'attn': float(row[3]),
                                'sparse': float(row[4]),
                                'conv': float(row[5])
                            })
                    
                    print(f"[HIGHLIGHTING] Loaded {len(points)} points from CSV")
                    
                    # Define constraint mapping
                    constraint_mapping = {
                        'gpu': {
                            'none': lambda p: p['gpu'] == 0,
                            'low': lambda p: 1 <= p['gpu'] <= 3,
                            'medium': lambda p: 4 <= p['gpu'] <= 6,
                            'high': lambda p: p['gpu'] >= 7
                        },
                        'attention': {
                            'none': lambda p: p['attn'] == 0,
                            'low': lambda p: 1 <= p['attn'] <= 2,
                            'medium': lambda p: 3 <= p['attn'] <= 5,
                            'high': lambda p: p['attn'] >= 6
                        },
                        'sparse': {
                            'none': lambda p: p['sparse'] == 0,
                            'low': lambda p: 1 <= p['sparse'] <= 3,
                            'medium': lambda p: 4 <= p['sparse'] <= 7,
                            'high': lambda p: p['sparse'] >= 8
                        },
                        'convolution': {
                            'none': lambda p: p['conv'] == 0,
                            'low': lambda p: 1 <= p['conv'] <= 2,
                            'medium': lambda p: 3 <= p['conv'] <= 4,
                            'high': lambda p: p['conv'] >= 5
                        }
                    }
                    
                    # Find matching designs
                    matching_designs = []
                    if chiplet_type in constraint_mapping and constraint_level in constraint_mapping[chiplet_type]:
                        constraint_func = constraint_mapping[chiplet_type][constraint_level]
                        matching_designs = [point for point in points if constraint_func(point)]
                        print(f"[HIGHLIGHTING] Found {len(matching_designs)} matching designs")
                    else:
                        print(f"[HIGHLIGHTING] Invalid constraint: chiplet_type='{chiplet_type}' or constraint_level='{constraint_level}'")
                        return JsonResponse({"message": "I couldn't understand the highlighting request. Please specify a chiplet type (GPU, Attention, Sparse, Convolution) and a level (none, low, medium, high)."})
                    
                    # Return matching designs with indices for highlighting
                    highlighted_points = []
                    for i, point in enumerate(points):
                        is_matching = any(
                            point['x'] == match['x'] and point['y'] == match['y'] 
                            for match in matching_designs
                        )
                        highlighted_points.append({
                            'index': i,
                            'highlighted': is_matching,
                            'x': point['x'],
                            'y': point['y'],
                            'gpu': point['gpu'],
                            'attn': point['attn'],
                            'sparse': point['sparse'],
                            'conv': point['conv']
                        })
                    
                    highlight_data = {
                        "highlighted_points": highlighted_points,
                        "matching_count": len(matching_designs),
                        "total_count": len(points),
                        "constraint": f"{constraint_level} {chiplet_type} chiplets"
                    }
                    
                    chat_reply = f"The designs with {highlight_data['constraint']} are highlighted in yellow. Found {highlight_data['matching_count']} out of {highlight_data['total_count']} designs."
                    
                    print(f"[HIGHLIGHTING] Success: {chat_reply}")
                    
                    # Return both message and highlighting data
                    return JsonResponse({
                        "message": chat_reply,
                        "highlighting_data": highlight_data
                    })
                    
                except Exception as e:
                    print(f"[HIGHLIGHTING] Exception in direct call: {e}")
                    import traceback
                    traceback.print_exc()
                    return JsonResponse({"message": "Sorry, I encountered an error highlighting the designs. Please try again."})
            else:
                print(f"[HIGHLIGHTING] Failed to parse: chiplet_type={chiplet_type}, constraint_level={constraint_level}")
                return JsonResponse({"message": "I couldn't understand the highlighting request. Please specify a chiplet type (GPU, Attention, Sparse, Convolution) and a level (none, low, medium, high)."})
        except Exception as e:
            print(f"[HIGHLIGHTING] Exception: {e}")
            import traceback
            traceback.print_exc()
            return JsonResponse({"message": "Sorry, I encountered an error highlighting the designs. Please try again."})
    
    # Check for optimization run intent
    if content and ("run optimization" in content.lower() or "start ga" in content.lower()):
        # --- Extract parameters (use existing or dummy logic for now) ---
        # For demo, use dummy values or parse from content
        model = "CASCADE"
        algorithm = "Genetic Algorithm"
        trace = "gpt-j-65536-weighted"
        objectives = ["Energy", "Runtime"]
        pop_size = 50
        generations = 100
        # TODO: Replace above with actual parsing logic if available
        # Step 1: Immediately return a confirmation message
        chat_reply = f"Optimization run started using Genetic Algorithm on trace '{trace}' optimizing for {', '.join(objectives)}. I'll let you know when it's done!"
        # Step 2: Trigger GA run in background
        def run_async():
            # You may want to call run_optimization or runGACascade here
            # For demo, just call runGACascade
            runGACascade(pop_size=pop_size, n_gen=generations, trace=trace)
            # When done, add a notification to chat history
            chat_bot.messages.append({
                "role": "assistant",
                "content": f"Optimization run on trace '{trace}' is complete! Check the plot for results."
            })
        threading.Thread(target=run_async).start()
        # Expose a hint that a run has started; frontend can poll for latest
        return JsonResponse({"message": chat_reply, "run_started": True})
    if content and "run" in content.lower() and any(x in content.lower() for x in ["model", "trace", "population"]):
        parsed_result = chat_bot.handle_natural_language_optimization(content)
        # If we can attach a run id in future, include it here
        return Response({"response": parsed_result["message"], "run_started": parsed_result.get("status") == "success"})
    # Try RAG path first
    try:
        filters = {}
        # Optional: pass simple filters derived from query params
        for key in ["trace", "doc_type"]:
            val = request.GET.get(key)
            if val:
                filters[key] = val
        rag_result = chat_bot.get_response_with_retrieval(content, role, filters or None)
        if rag_result and isinstance(rag_result, dict) and rag_result.get("final_answer"):
            return Response({
                "response": rag_result["final_answer"],
                "citations": rag_result.get("citations", [])
            })
    except Exception as e:
        # Fallback to baseline chat
        pass

    response = chat_bot.get_response(content, role)
    return Response({"response": response})

@api_view(["GET"])
def get_latest_run_directory(request):
    """
    Return the most recent run directory (myrun_YYYYMonDD_HHMMSS) if exists on disk.
    Used by chat-triggered runs to discover the active run id.
    """
    try:
        base_path = "api/Evaluator/cascade/chiplet_model/dse/results"
        if not os.path.exists(base_path):
            return Response({"error": "results directory not found"}, status=404)
        candidates = [d for d in os.listdir(base_path) if d.startswith("myrun_")]
        if not candidates:
            return Response({"status": "empty"})
        latest = sorted(candidates)[-1]
        return Response({"status": "success", "run_directory": latest})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
def clear_chat(request):
    """
    Clear the chat history in the chatbot.
    """
    chat_bot.clear_history()
    return Response({"message": "Chat history cleared successfully."})

@api_view(["POST"])
def add_insights_context(request):
    """
    Add insights context to the AI's conversation history.
    """
    try:
        data = json.loads(request.body)
        insights = data.get("insights")
        if insights:
            # Add the insights as an assistant message to provide context
            chat_bot.messages.append({
                "role": "assistant",
                "content": f"Here are the insights from the analysis:\n\n{insights}\n\nI have this context and can answer follow-up questions about these insights."
            })
            return Response({"message": "Insights context added successfully"})
        else:
            return Response({"error": "No insights provided"}, status=400)
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["GET"])
def get_point_context(request):
    """
    Retrieve detailed point context JSON for a specific design point.
    """
    try:
        # Get parameters
        run_id = request.GET.get("run_id")
        # Optional: full design string like "0gpu6attn6sparse0conv"
        design = request.GET.get("design")
        gpu = request.GET.get("gpu", "0")
        attn = request.GET.get("attn", "0")
        sparse = request.GET.get("sparse", "0")
        conv = request.GET.get("conv", "0")
        
        if not run_id:
            return Response({"error": "run_id is required"}, status=400)
        
        # Construct the file path
        base_path = "/Users/ramyagotika/research-work/chiplet/chiplet-server/api/Evaluator/cascade/chiplet_model/dse/results"
        
        # Construct candidate run directories to handle different formats
        candidates = []
        if run_id.startswith("loaded_run_"):
            # Map loaded_run_YYYYMMDD_HHMMSS → myrun_YYYYMMDD_HHMMSS
            mapped = run_id.replace("loaded_run_", "myrun_")
            if "_" in mapped:
                parts = mapped.split("_")
                if len(parts) >= 3:
                    date_part = parts[1]
                    time_part = parts[2]
                    mapped = f"myrun_{date_part}_{time_part}"
            candidates.append(mapped)
            # Also try the original loaded_run_ directory name
            candidates.append(run_id)
            # Also try temp_points_<loaded_run_...> (based on CSV naming pattern)
            candidates.append(f"temp_points_{run_id}")
        else:
            candidates.append(f"myrun_{run_id}")
            candidates.append(run_id)
            # If this looks like restarted_run_YYYYMMDD_HHMMSS, try as-is
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
            # Gather available designs in the candidate run directories (if they exist)
            available_designs = []
            for run_dir in candidates:
                pc_dir = os.path.join(base_path, run_dir, "pointContext")
                if os.path.isdir(pc_dir):
                    try:
                        for fname in os.listdir(pc_dir):
                            if fname.endswith('.json'):
                                available_designs.append(fname[:-5])  # strip .json
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
def evaluate_point(request):
    """
    Evaluate a point using the DataGenerator.
    """
    print("Made it to the evaluate_point function")
    try:
        chipletKeys = request.GET.getlist("chiplets[]")
        trace = request.GET.get("trace")
        print("Chiplets:", chipletKeys)
        print("Trace:", trace)
        chiplets = {}
        for key in chipletKeys:
            chiplets[key] = chiplets.get(key, 0) + 1
                 
        dataGenerator.evaluate_point(chiplets, trace)
        data = dataGenerator.get_data()
        return Response({"data": data})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["GET"])
def evaluate_point_inputs(request):
    """
    Evaluate a point using the DataGenerator.
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
            "trace": trace,
        }
        
        print("Evaluated point being returned:", evaluated_point)
        return Response({"data": evaluated_point})
    except Exception as e:
        print(f"Error in evaluate_point_inputs: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
    
@api_view(["GET"])
def add_info(request):
    """
    Add information to the chatbot.
    """
    # print("Made it to the add_info function")
    exe = request.GET.get("exe")
    energy = request.GET.get("energy")
    gpu = request.GET.get("gpu")
    attn = request.GET.get("attn")
    sparse = request.GET.get("sparse")
    conv = request.GET.get("conv")
    content = f"The design is evaluated to have an energy of {energy} and an execution time of {exe}.\n"
    content += f"The design has {gpu} GPU chiplets, {attn} attention chiplets, {sparse} sparse chiplets, and {conv} convolution chiplets.\n"
    chiplet_file_path = f"api/Evaluator/cascade/chiplet_model/dse/results/pointContext/{gpu}gpu{attn}attn{sparse}sparse{conv}conv.json"
    # try:
    #     with open(chiplet_file_path, "r") as file:
    #         content += file.read()
    # except FileNotFoundError:
    #     return Response({"error": "Chiplet file not found."}, status=404)
    chat_bot.add_information(chiplet_file_path)
    # chat_bot.add_information("I have received context on this design! I am ready to answer questions about it.", "assistant")
    chat_bot.messages.append(
        {
            "role": "assistant",
            "content": "I have received context on this design! I am ready to answer questions about it."
        }
    )
    return Response({"message": "Information added successfully."})

@api_view(["GET"])
def rule_mining(request):
    """
    Run rule mining and return the results in a structured format for the frontend table.
    """
    import time
    print("[rule_mining] Called rule_mining endpoint.")
    start_time = time.time()
    
    # Get point selection parameters from request
    region = request.GET.get("region", "pareto")
    pareto_start_rank = int(request.GET.get("paretoStartRank", 1))
    pareto_end_rank = int(request.GET.get("paretoEndRank", 3))
    energy_min = request.GET.get("energyMin")
    energy_max = request.GET.get("energyMax")
    time_min = request.GET.get("timeMin")
    time_max = request.GET.get("timeMax")
    
    # Get file path for loaded runs
    file_path = request.GET.get("file_path", "api/Evaluator/cascade/chiplet_model/dse/results/points.csv")
    
    print(f"[rule_mining] Parameters: region={region}, pareto_ranks={pareto_start_rank}-{pareto_end_rank}")
    print(f"[rule_mining] Using file: {file_path}")
    if energy_min and energy_max:
        print(f"[rule_mining] Energy range: {energy_min}-{energy_max}")
    if time_min and time_max:
        print(f"[rule_mining] Time range: {time_min}-{time_max}")
    
    try:
        # Create point selection parameters for ChatBot model
        point_selection_params = {
            "region": region,
            "pareto_start_rank": pareto_start_rank,
            "pareto_end_rank": pareto_end_rank,
            "energy_min": float(energy_min) if energy_min else None,
            "energy_max": float(energy_max) if energy_max else None,
            "time_min": float(time_min) if time_min else None,
            "time_max": float(time_max) if time_max else None,
            "file_path": file_path  # Pass the file path to the ChatBot model
        }
        
        rule_mining_str = chat_bot.rule_mining(point_selection_params)
        print("[rule_mining] Finished rule_mining() call.")
        
        # Parse the rule_mining_str into a list of dicts for the frontend
        rules = []
        rule_pattern = re.compile(r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), conf\(p->f\): ([0-9.eE+-]+), lift: \[([0-9.eE+-]+)\]")
        for match in rule_pattern.finditer(rule_mining_str):
            rules.append({
                "rule": match.group(1),
                "conf_p_to_f": float(match.group(2)),
                "conf_f_to_p": float(match.group(3)),
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
    Compute distance correlation between each chiplet type and Energy (y column) and Time (x column).
    """
    try:
        # Get file path for loaded runs
        file_path = request.GET.get("file_path", "api/Evaluator/cascade/chiplet_model/dse/results/points.csv")
        print(f"[distance_correlation] Using file: {file_path}")
        
        # Load data from CSV
        import csv
        xs, ys, gpus, attns, sparses, convs = [], [], [], [], [], []
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
                gpus.append(float(row[2]))
                attns.append(float(row[3]))
                sparses.append(float(row[4]))
                convs.append(float(row[5]))
        
        # Check if we have data
        if len(xs) == 0:
            return Response({"error": "No data available for distance correlation analysis"}, status=400)
        
        # Compute distance correlation for each chiplet type vs Energy and vs Time
        def safe_dcor(x, y):
            """Safely compute distance correlation, handling edge cases"""
            try:
                result = float(dcor.distance_correlation(np.array(x), np.array(y)))
                # Check for NaN or infinite values
                if np.isnan(result) or np.isinf(result):
                    return 0.0
                return result
            except Exception as e:
                print(f"Error computing distance correlation: {e}")
                return 0.0
        
        result = {
            "GPU_vs_Energy": safe_dcor(gpus, ys),
            "Attention_vs_Energy": safe_dcor(attns, ys),
            "Sparse_vs_Energy": safe_dcor(sparses, ys),
            "Convolution_vs_Energy": safe_dcor(convs, ys),
            "GPU_vs_Time": safe_dcor(gpus, xs),
            "Attention_vs_Time": safe_dcor(attns, xs),
            "Sparse_vs_Time": safe_dcor(sparses, xs),
            "Convolution_vs_Time": safe_dcor(convs, xs),
        }
        
        print(f"[distance_correlation] Computed correlations: {result}")
        return Response(result)
    except Exception as e:
        print(f"[distance_correlation] Exception: {e}")
        return Response({"error": str(e)}, status=500)

@api_view(["GET"])
def distance_correlation_insights(request):
    """
    Compute distance correlation and send to LLM for meaningful insights analysis.
    Now supports goal-aware, design-specific insights with structured data.
    """
    try:
        from api.ChatBot.model import ChatBotModel
        chat_bot = ChatBotModel()
        
        # Get optimization context parameters
        objective = request.GET.get("objective", "both")  # "energy", "time", or "both"
        trace_name = request.GET.get("trace_name", "Unknown")
        run_id = request.GET.get("run_id", None)
        
        # Get file path for loaded runs
        file_path = request.GET.get("file_path", "api/Evaluator/cascade/chiplet_model/dse/results/points.csv")
        
        # Load data from CSV
        import csv
        xs, ys, gpus, attns, sparses, convs = [], [], [], [], [], []
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
                gpus.append(float(row[2]))
                attns.append(float(row[3]))
                sparses.append(float(row[4]))
                convs.append(float(row[5]))
        
        # Compute distance correlation safely (avoid NaN/Inf in JSON)
        def safe_dcor(a, b):
            try:
                val = float(dcor.distance_correlation(np.array(a), np.array(b)))
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            except Exception:
                return None

        correlations = {
            "GPU_vs_Energy": safe_dcor(gpus, ys),
            "Attention_vs_Energy": safe_dcor(attns, ys),
            "Sparse_vs_Energy": safe_dcor(sparses, ys),
            "Convolution_vs_Energy": safe_dcor(convs, ys),
            "GPU_vs_Time": safe_dcor(gpus, xs),
            "Attention_vs_Time": safe_dcor(attns, xs),
            "Sparse_vs_Time": safe_dcor(sparses, xs),
            "Convolution_vs_Time": safe_dcor(convs, xs),
        }
        
        # Create structured JSON data for UI display
        energy_correlations = {k: v for k, v in correlations.items() if 'Energy' in k}
        time_correlations = {k: v for k, v in correlations.items() if 'Time' in k}
        
        # Sort by correlation value (descending)
        # Sort, treating None as -inf so they sink to bottom
        energy_sorted = sorted(energy_correlations.items(), key=lambda x: (-np.inf if x[1] is None else x[1]), reverse=True)
        time_sorted = sorted(time_correlations.items(), key=lambda x: (-np.inf if x[1] is None else x[1]), reverse=True)
        
        # Create structured JSON data
        structured_data = {
            "high_impact_on_energy": [
                {"chiplet": chiplet_metric.split('_vs_')[0], "correlation": (round(value, 3) if (value is not None) else None)}
                for chiplet_metric, value in energy_sorted
            ],
            "high_impact_on_time": [
                {"chiplet": chiplet_metric.split('_vs_')[0], "correlation": (round(value, 3) if (value is not None) else None)}
                for chiplet_metric, value in time_sorted
            ],
            "trace_name": trace_name,
            "objective": objective,
            "run_id": run_id
        }
        
        # Create a formatted string for the AI
        correlation_data = "Distance Correlation Analysis Results:\n\n"
        correlation_data += "Energy Impact (Distance Correlation Values):\n"
        for chiplet_metric, value in energy_sorted:
            chiplet = chiplet_metric.split('_vs_')[0]
            val_str = f"{value:.3f}" if value is not None else "N/A"
            correlation_data += f"• {chiplet}: {val_str}\n"
        
        correlation_data += "\nTime Impact (Distance Correlation Values):\n"
        for chiplet_metric, value in time_sorted:
            chiplet = chiplet_metric.split('_vs_')[0]
            val_str = f"{value:.3f}" if value is not None else "N/A"
            correlation_data += f"• {chiplet}: {val_str}\n"
        
        # Create goal-aware prompt
        if objective == "energy":
            goal_text = "minimize energy consumption"
            focus_metric = "energy"
        elif objective == "time":
            goal_text = "minimize execution time"
            focus_metric = "execution time"
        else:
            goal_text = "optimize both energy and execution time"
            focus_metric = "both metrics"
        
        prompt = (
            f"You are an expert in chiplet design. The current design goal is to {goal_text}.\n\n"
            f"Trace used: {trace_name}\n\n"
            f"Here are distance correlation results showing the relationship between chiplet types and performance metrics. "
            f"Distance correlation ranges from 0 (no relationship) to 1 (perfect relationship).\n\n"
            f"JSON Data:\n{json.dumps(structured_data, indent=2)}\n\n"
            f"Provide a concise, actionable summary (2-3 sentences) covering:\n"
            f"1. Which chiplet types have the strongest impact on {focus_metric} and why\n"
            f"2. One practical design recommendation for improving performance\n"
            f"3. Any surprising findings (if any)\n\n"
            f"Keep your response focused and to the point. Users can ask follow-up questions for more details."
        )
        
        response = chat_bot.get_response(prompt, role="user")
        return Response({
            "insights": response,
            "structured_data": structured_data
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["GET"])
def rule_mining_insights(request):
    """
    Run rule mining, send the results to the LLM, and return a natural-language summary.
    Now supports goal-aware, design-specific insights with structured data.
    """
    try:
        from api.ChatBot.model import ChatBotModel
        chat_bot = ChatBotModel()
        
        # Get optimization context parameters
        objective = request.GET.get("objective", "both")  # "energy", "time", or "both"
        trace_name = request.GET.get("trace_name", "Unknown")
        run_id = request.GET.get("run_id", None)
        
        # Get point selection parameters from request (same as rule_mining endpoint)
        region = request.GET.get("region", "pareto")
        pareto_start_rank = int(request.GET.get("paretoStartRank", 1))
        pareto_end_rank = int(request.GET.get("paretoEndRank", 3))
        energy_min = request.GET.get("energyMin")
        energy_max = request.GET.get("energyMax")
        time_min = request.GET.get("timeMin")
        time_max = request.GET.get("timeMax")
        
        # Get file path for loaded runs
        file_path = request.GET.get("file_path", "api/Evaluator/cascade/chiplet_model/dse/results/points.csv")
        
        # Create point selection parameters for ChatBot model
        point_selection_params = {
            "region": region,
            "pareto_start_rank": pareto_start_rank,
            "pareto_end_rank": pareto_end_rank,
            "energy_min": float(energy_min) if energy_min else None,
            "energy_max": float(energy_max) if energy_max else None,
            "time_min": float(time_min) if time_min else None,
            "time_max": float(time_max) if time_max else None,
            "file_path": file_path  # Add file path for loaded runs
        }
        
        rule_mining_str = chat_bot.rule_mining(point_selection_params)
        
        # Format the rules into natural language before sending to AI
        def format_rule_natural_language(rule_str):
            """Convert underscore-separated rules to natural language"""
            if not rule_str:
                return rule_str
            
            # Split the rule into parts (usually separated by AND/OR)
            import re
            parts = re.split(r'\s+(AND|OR)\s+', rule_str)
            formatted_parts = []
            
            for i, part in enumerate(parts):
                # Skip AND/OR operators, they'll be added back
                if part == 'AND' or part == 'OR':
                    formatted_parts.append(part)
                    continue
                
                # Format individual conditions
                formatted_part = format_condition_natural_language(part)
                formatted_parts.append(formatted_part)
            
            return ' '.join(formatted_parts)
        
        def format_condition_natural_language(condition):
            """Format individual conditions like 'Attention_none' to 'None Attention Chiplets'"""
            if not condition:
                return condition
            
            # Remove extra whitespace
            condition = condition.strip()
            
            # Handle chiplet conditions like "Attention_none", "GPU_high", etc.
            chiplet_pattern = r'^(\w+)_(\w+)$'
            match = re.match(chiplet_pattern, condition)
            
            if match:
                chiplet_type, level = match.groups()
                
                # Map chiplet types to proper names
                chiplet_names = {
                    'GPU': 'GPU',
                    'Attention': 'Attention', 
                    'Sparse': 'Sparse',
                    'Convolution': 'Convolution'
                }
                
                # Map levels to natural language
                level_names = {
                    'none': 'None',
                    'low': 'Low',
                    'medium': 'Medium',
                    'high': 'High'
                }
                
                chiplet_name = chiplet_names.get(chiplet_type, chiplet_type)
                level_name = level_names.get(level, level)
                
                return f"{level_name} {chiplet_name} Chiplets"
            
            return condition
        
        # Parse rules into structured JSON format
        def parse_rules_to_json(rule_mining_str):
            """Parse rule mining string into structured JSON format"""
            import re
            rules = []
            
            # Pattern to match rule lines
            rule_pattern = re.compile(r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), conf\(p->f\): ([0-9.eE+-]+), lift: \[([0-9.eE+-]+)\]")
            
            for match in rule_pattern.finditer(rule_mining_str):
                rule_str = match.group(1)
                conf_f_to_p = float(match.group(2))
                conf_p_to_f = float(match.group(3))
                lift = float(match.group(4))
                
                # Parse conditions from rule string
                conditions = []
                if ' AND ' in rule_str:
                    conditions = [format_condition_natural_language(cond.strip()) for cond in rule_str.split(' AND ')]
                else:
                    conditions = [format_condition_natural_language(rule_str.strip())]
                
                rules.append({
                    "conditions": conditions,
                    "confidence_f_to_p": round(conf_f_to_p, 3),
                    "confidence_p_to_f": round(conf_p_to_f, 3),
                    "lift": round(lift, 3)
                })
            
            return rules
        
        # Create structured JSON data
        structured_rules = parse_rules_to_json(rule_mining_str)
        structured_data = {
            "rules": structured_rules,
            "trace_name": trace_name,
            "objective": objective,
            "run_id": run_id,
            "analysis_region": f"{region} (ranks {pareto_start_rank}-{pareto_end_rank})"
        }
        
        # Format the rule mining string for AI
        formatted_rule_mining_str = format_rule_natural_language(rule_mining_str)
        
        # Create goal-aware prompt
        if objective == "energy":
            goal_text = "minimize energy consumption"
        elif objective == "time":
            goal_text = "minimize execution time"
        else:
            goal_text = "optimize both energy and execution time"
        
        prompt = (
            f"You are a chiplet design analyst. A rule mining analysis was run on Pareto-optimal points for the goal: {goal_text}.\n\n"
            f"Trace used: {trace_name}\n\n"
            f"Here are the rules extracted:\n"
            f"JSON Data:\n{json.dumps(structured_data, indent=2)}\n\n"
            f"Provide a concise, actionable summary (2-3 sentences) covering:\n"
            f"1. The most important recurring pattern in optimal designs\n"
            f"2. One specific recommendation for chiplet combination\n"
            f"3. Any rule conflicts or redundancies (if any)\n\n"
            f"Keep your response focused and to the point. Users can ask follow-up questions for more details."
        )
        
        response = chat_bot.get_response(prompt, role="user")
        
        # Post-process to remove any sentence containing 'distance correlation'
        def clean_rule_mining_response(text):
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            cleaned_sentences = [s for s in sentences if 'distance correlation' not in s.lower()]
            result = ' '.join(cleaned_sentences).strip()
            return result

        cleaned_response = clean_rule_mining_response(response)
        return Response({
            "insights": cleaned_response,
            "structured_data": structured_data
        })
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
@csrf_exempt
def run_optimization(request):
    print("🔍 DEBUG: run_optimization function called!")
    print("🔍 DEBUG: Request method:", request.method)
    print("🔍 DEBUG: Request path:", request.path)
    print("🔍 DEBUG: Request headers:", dict(request.headers))
    
    print("=== Backend: run_optimization called ===")
    print(f"Backend: Request method: {request.method}")
    print(f"Backend: Request body: {request.body}")
    
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            print(f"Backend: Parsed JSON data: {data}")
            
            # Check if this is a comparative study
            comparative_study = data.get('comparative_study', False)
            print(f"Backend: Comparative study flag: {comparative_study}")
            
            if comparative_study:
                print("Backend: Processing as COMPARATIVE study")
                try:
                    # Check if both runs are previous runs (fast comparison)
                    run_a_config = data.get('run_a', {})
                    run_b_config = data.get('run_b', {})
                    
                    if run_a_config.get('type') == 'previous' and run_b_config.get('type') == 'previous':
                        print("Backend: FAST PATH - Both runs are previous, loading existing data")
                        
                        # Load previous run data directly from files
                        import pandas as pd
                        import os
                        
                        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
                        
                        def load_previous_run_data(backup_filename):
                            """Load data from a previous run backup file"""
                            file_path = os.path.join(WORKSPACE, 'dse/results', backup_filename)
                            
                            if not os.path.exists(file_path):
                                return {
                                    'points_count': 0,
                                    'pareto_count': 0,
                                    'best_energy': 0,
                                    'best_time': 0
                                }
                            
                            try:
                                # Read the CSV file without headers
                                df = pd.read_csv(file_path, header=None)
                                
                                print(f"Backend: Loaded {backup_filename} with shape {df.shape}")
                                print(f"Backend: First few rows: {df.head(3).values.tolist()}")
                                
                                # Calculate metrics
                                points_count = len(df)
                                
                                # First column is energy, second column is time
                                if len(df.columns) >= 2:
                                    best_energy = df[0].min()  # First column is energy
                                    best_time = df[1].min()   # Second column is time
                                    print(f"Backend: Energy range: {df[0].min()} to {df[0].max()}")
                                    print(f"Backend: Time range: {df[1].min()} to {df[1].max()}")
                                else:
                                    best_energy = 0
                                    best_time = 0
                                
                                # Simple Pareto calculation (points that are not dominated)
                                pareto_count = 0
                                if len(df) > 0 and len(df.columns) >= 2:
                                    for i, row in df.iterrows():
                                        dominated = False
                                        for j, other_row in df.iterrows():
                                            if i != j:
                                                if (other_row[0] <= row[0] and 
                                                    other_row[1] <= row[1] and
                                                    (other_row[0] < row[0] or 
                                                     other_row[1] < row[1])):
                                                    dominated = True
                                                    break
                                        if not dominated:
                                            pareto_count += 1
                                
                                return {
                                    'points_count': points_count,
                                    'pareto_count': pareto_count,
                                    'best_energy': best_energy,
                                    'best_time': best_time
                                }
                            except Exception as e:
                                print(f"Error loading {backup_filename}: {e}")
                                return {
                                    'points_count': 0,
                                    'pareto_count': 0,
                                    'best_energy': 0,
                                    'best_time': 0
                                }
                        
                        # Load both runs
                        run_a_filename = run_a_config.get('backup_filename')
                        run_b_filename = run_b_config.get('backup_filename')
                        
                        run_a_data = load_previous_run_data(run_a_filename)
                        run_b_data = load_previous_run_data(run_b_filename)
                        
                        print(f"Backend: Run A data: {run_a_data}")
                        print(f"Backend: Run B data: {run_b_data}")
                        
                        # Return immediate results
                        return JsonResponse({
                            "status": "success",
                            "message": "Previous runs compared successfully",
                            "run_a_id": f"previous_{run_a_filename}",
                            "run_b_id": f"previous_{run_b_filename}",
                            "run_a_points": run_a_data['points_count'],
                            "run_b_points": run_b_data['points_count'],
                            "run_a_pareto": run_a_data['pareto_count'],
                            "run_b_pareto": run_b_data['pareto_count'],
                            "run_a_best_energy": run_a_data['best_energy'],
                            "run_b_best_energy": run_b_data['best_energy'],
                            "run_a_best_time": run_a_data['best_time'],
                            "run_b_best_time": run_b_data['best_time'],
                            "comparison_type": "previous_runs",
                            "timestamp": "2025-08-03T20:48:00"
                        })
                    
                    # Regular comparative study path (for new runs or mixed)
                    print("Backend: REGULAR PATH - Running optimization for comparative study")
                    
                    # 1. Parse shared parameters
                    model = data.get('model')
                    algorithm = data.get('algorithm')
                    objectives = data.get('objectives')
                    population = int(data.get('population_size', data.get('population', 0)))
                    generations = int(data.get('generations', 0))
                    trace_sets = data['trace_sets']
                    
                    # Create optimization runs in database
                    runs = {}
                    for run_label in ['A', 'B']:
                        run_info = trace_sets.get(run_label)
                        if not run_info:
                            return JsonResponse({"status": "error", "message": f"Missing trace set for run {run_label}"}, status=400)
                        
                        traces = run_info.get('traces', [])
                        weights = run_info.get('weights', [])
                        if not traces or not weights or len(traces) != len(weights):
                            return JsonResponse({"status": "error", "message": f"Invalid traces/weights for run {run_label}"}, status=400)
                        total_weight = sum(weights)
                        if not np.isclose(total_weight, 1.0):
                            return JsonResponse({"status": "error", "message": f"Sum of weights for run {run_label} must be 1.0, got {total_weight}"}, status=400)
                        
                        # Create run record
                        run_name = f"Comparative Study {run_label}"
                        run_description = f"Run {run_label} with traces: {traces}"
                        
                        optimization_run = RunStorageService.create_optimization_run(
                            algorithm=algorithm,
                            model=model,
                            population_size=population,
                            generations=generations,
                            objectives=objectives,
                            trace_sets={run_label: run_info},
                            name=run_name,
                            description=run_description
                        )
                        runs[run_label] = optimization_run

                    results = {}
                    plot_data = {"A": [], "B": []}
                    analytics_files = {"runA": {}, "runB": {}}

                    # --- Analytics helper stubs ---
                    import os
                    def run_rule_mining(run_results, run_label):
                        # Stub: Save dummy rule mining results
                        analytics_dir = os.path.join("analytics")
                        os.makedirs(analytics_dir, exist_ok=True)
                        file_path = os.path.join(analytics_dir, f"run{run_label}_rule_mining.json")
                        result = {"rule_mining": f"Stub rule mining for run {run_label}"}
                        with open(file_path, "w") as f:
                            json.dump(result, f)
                        return file_path
                    def run_distance_correlation(run_results, run_label):
                        # Stub: Save dummy distance correlation results
                        analytics_dir = os.path.join("analytics")
                        os.makedirs(analytics_dir, exist_ok=True)
                        file_path = os.path.join(analytics_dir, f"run{run_label}_distance_correlation.json")
                        result = {"distance_correlation": f"Stub distance correlation for run {run_label}"}
                        with open(file_path, "w") as f:
                            json.dump(result, f)
                        return file_path

                    run_results = {}
                    def run_ga_for_label(run_label, traces, weights, optimization_run):
                        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
                        OUTPUT_DIR = f"{WORKSPACE}/dse/results/run{run_label}_results"
                        os.makedirs(OUTPUT_DIR, exist_ok=True)
                        
                        if len(traces) == 1:
                            trace_name = traces[0] if isinstance(traces[0], str) else traces[0].get('name', traces[0])
                            ga_result = runGACascade(pop_size=population, n_gen=generations, trace=trace_name, output_dir=OUTPUT_DIR)
                        else:
                            weighted_trace_name = generate_weighted_trace(traces, weights, label=run_label)
                            ga_result = runGACascade(pop_size=population, n_gen=generations, trace=weighted_trace_name, output_dir=OUTPUT_DIR)
                        
                        ga_result = convert_ndarrays(ga_result)
                        with open(f"{OUTPUT_DIR}/ga_result.json", "w") as f:
                            json.dump(ga_result, f)
                        
                        # Convert results to design points format
                        design_points = []
                        for pt in ga_result:
                            if isinstance(pt, dict):
                                # Extract chiplet configuration from the result
                                chiplets = pt.get('chiplets', {})
                                design_point = {
                                    'execution_time_ms': pt.get('execution_time_ms', pt.get('x', 0)),
                                    'energy_mj': pt.get('energy_mj', pt.get('y', 0)),
                                    'chiplets': chiplets,
                                    'additional_metrics': pt.get('additional_metrics', {}),
                                    'context_file_path': pt.get('context_file_path', '')
                                }
                                design_points.append(design_point)
                                pt['run'] = run_label
                            elif isinstance(pt, list) or isinstance(pt, tuple):
                                # Handle legacy format
                                if len(pt) >= 2:
                                    design_point = {
                                        'execution_time_ms': float(pt[0]),
                                        'energy_mj': float(pt[1]),
                                        'chiplets': {
                                            'GPU': int(pt[2]) if len(pt) > 2 else 0,
                                            'Attention': int(pt[3]) if len(pt) > 3 else 0,
                                            'Sparse': int(pt[4]) if len(pt) > 4 else 0,
                                            'Convolution': int(pt[5]) if len(pt) > 5 else 0
                                        }
                                    }
                                    design_points.append(design_point)
                                pt = list(pt) + [run_label]
                        
                        # Store design points in database
                        RunStorageService.store_design_points(
                            optimization_run, 
                            design_points, 
                            OUTPUT_DIR
                        )
                        
                        tagged_results = []
                        for pt in ga_result:
                            if isinstance(pt, dict):
                                pt['run'] = run_label
                            elif isinstance(pt, list) or isinstance(pt, tuple):
                                pt = list(pt) + [run_label]
                            tagged_results.append(pt)
                        run_results[run_label] = tagged_results

                    threads = []
                    for run_label in ['A', 'B']:
                        run_info = trace_sets.get(run_label)
                        traces = run_info.get('traces', [])
                        weights = run_info.get('weights', [])
                        optimization_run = runs[run_label]
                        
                        t = threading.Thread(target=run_ga_for_label, args=(run_label, traces, weights, optimization_run))
                        threads.append(t)
                        t.start()
                    
                    for t in threads:
                        t.join()

                    # After both threads finish, continue with analytics and response construction
                    for run_label in ['A', 'B']:
                        tagged_results = run_results.get(run_label, [])
                        results[f'run_{run_label.lower()}_results'] = tagged_results
                        plot_data[run_label] = tagged_results
                        
                        # Get run data for plotting
                        run_data = RunStorageService.get_run_for_plotting(runs[run_label].run_id)
                        if run_data:
                            plot_data[run_label] = run_data['design_points']
                        
                        # Real per-run analytics
                        try:
                            WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
                            OUTPUT_DIR = f"{WORKSPACE}/dse/results/run{run_label}_results"
                            points_csv_path = f"{OUTPUT_DIR}/points.csv"
                            analytics_dir = os.path.join("analytics")
                            os.makedirs(analytics_dir, exist_ok=True)
                            rule_mining_result = run_rule_mining_for_run(points_csv_path)
                            rule_mining_path = os.path.join(analytics_dir, f"run{run_label}_rule_mining.json")
                            with open(rule_mining_path, "w") as f:
                                json.dump({"rule_mining": rule_mining_result}, f)
                            analytics_files[f'run{run_label}']['rule_mining'] = rule_mining_path
                        except Exception as e:
                            analytics_files[f'run{run_label}']['rule_mining'] = f"error: {str(e)}"
                        try:
                            distance_corr_result = run_distance_correlation_for_run(points_csv_path)
                            dist_corr_path = os.path.join(analytics_dir, f"run{run_label}_distance_correlation.json")
                            with open(dist_corr_path, "w") as f:
                                json.dump({"distance_correlation": distance_corr_result}, f)
                            analytics_files[f'run{run_label}']['distance_correlation'] = dist_corr_path
                        except Exception as e:
                            analytics_files[f'run{run_label}']['distance_correlation'] = f"error: {str(e)}"
                    
                    # Complete runs in database
                    for run_label in ['A', 'B']:
                        analytics_results = {
                            'rule_mining': analytics_files[f'run{run_label}'].get('rule_mining', ''),
                            'distance_correlation': analytics_files[f'run{run_label}'].get('distance_correlation', '')
                        }
                        RunStorageService.complete_run(runs[run_label], analytics_results=analytics_results)
                    
                    # STEP 1: Automatically inject comparative study context into ChatBot
                    def generate_comparative_summary(run_a_id, run_b_id, trace_sets, config):
                        return f"""🔄 Comparative Study Complete!

📊 Study Summary:
• Run A: {run_a_id} ({trace_sets['A']['traces']})
• Run B: {run_b_id} ({trace_sets['B']['traces']})
• Model: {config['model']}
• Algorithm: {config['algorithm']}
• Objectives: {', '.join(config['objectives'])}

Both optimization runs have completed and I can help you compare the results!"""

                    def generate_comparative_insights(analytics_files):
                        insights = []
                        
                        # Check if analytics files exist and have content
                        for run_label in ['A', 'B']:
                            rule_file = analytics_files[f'run{run_label}'].get('rule_mining', '')
                            dist_file = analytics_files[f'run{run_label}'].get('distance_correlation', '')
                            
                            if rule_file and not rule_file.startswith('error'):
                                insights.append(f"🔍 Run {run_label}: Rule mining analysis completed")
                            if dist_file and not dist_file.startswith('error'):
                                insights.append(f"📈 Run {run_label}: Distance correlation analysis completed")
                        
                        if insights:
                            return "\n".join(insights)
                        else:
                            return "📊 Analytics completed for both runs - ready for comparison"
                    
                    def generate_comparative_suggestions():
                        return """💬 You can ask me to compare:

🎯 **Direct Comparisons:**
• "Which run performed better overall?"
• "Compare the Pareto fronts of both runs"
• "What are the key differences between Run A and Run B?"
• "Which run achieved better energy efficiency?"
• "Which run achieved faster execution times?"

🔍 **Detailed Analysis:**
• "Compare the rule mining results between runs"
• "How do the distance correlations differ?"
• "What patterns are unique to each run?"
• "Which run has more diverse design solutions?"

📊 **Specific Metrics:**
• "Compare the best energy consumption between runs"
• "Compare the fastest execution times between runs"
• "Which run has more Pareto-optimal designs?"
• "How do the design spaces differ?"

🔄 **Trace Impact:**
• "How do different traces affect the results?"
• "What's the impact of trace selection on optimization?"
• "Which trace combination works better?"

💡 **Recommendations:**
• "Based on the comparison, what should I optimize next?"
• "Which approach should I use for my next run?"
• "What insights can I apply to future optimizations?"

Just ask me to compare any aspect of the two runs!"""
                    
                    # Generate comparative context messages
                    comp_config = {
                        'model': model,
                        'algorithm': algorithm,
                        'objectives': objectives
                    }
                    
                    comp_summary_msg = generate_comparative_summary(
                        runs['A'].run_id, runs['B'].run_id, trace_sets, comp_config
                    )
                    comp_analytics_msg = generate_comparative_insights(analytics_files)
                    comp_suggestions_msg = generate_comparative_suggestions()
                    
                    # Add comparative context to ChatBot automatically
                    chat_bot.add_run_context(comp_summary_msg, comp_analytics_msg, comp_suggestions_msg)
                    
                    print("Backend: Comparative study completed successfully")
                    print("Backend: ChatBot comparative context automatically updated")
                    
                    # 7. Construct response
                    response = {
                        "status": "success",
                        "run_a_results": results.get('run_a_results', []),
                        "run_b_results": results.get('run_b_results', []),
                        "plot_data": plot_data,
                        "analytics_files": analytics_files,
                        "run_ids": {
                            "run_a": runs['A'].run_id,
                            "run_b": runs['B'].run_id
                        },
                        "metadata": {
                            "trace_sets": trace_sets,
                            "model": model,
                            "algorithm": algorithm,
                            "objectives": objectives,
                            "population": population,
                            "generations": generations
                        }
                    }
                    
                    # Step 1: Load analytics files
                    import json as pyjson
                    analytics_dir = os.path.join("analytics")
                    runA_rule_file = os.path.join(analytics_dir, "runA_rule_mining.json")
                    runA_dist_file = os.path.join(analytics_dir, "runA_distance_correlation.json")
                    runB_rule_file = os.path.join(analytics_dir, "runB_rule_mining.json")
                    runB_dist_file = os.path.join(analytics_dir, "runB_distance_correlation.json")
                    def safe_json_load(path):
                        try:
                            if os.path.exists(path):
                                with open(path) as f:
                                    return pyjson.load(f)
                        except Exception as e:
                            print(f"[Comparative Study] Failed to load {path}: {e}")
                        return None

                    # Step 2: Load analytics data
                    runA_rule_data = safe_json_load(runA_rule_file)
                    runA_dist_data = safe_json_load(runA_dist_file)
                    runB_rule_data = safe_json_load(runB_rule_file)
                    runB_dist_data = safe_json_load(runB_dist_file)

                    # Step 3: Generate comparative insights
                    comparative_insights = {
                        "run_a_rule_mining": runA_rule_data,
                        "run_a_distance_correlation": runA_dist_data,
                        "run_b_rule_mining": runB_rule_data,
                        "run_b_distance_correlation": runB_dist_data,
                    }

                    response["comparative_insights"] = comparative_insights

                    return JsonResponse(response)

                except Exception as e:
                    return JsonResponse({"status": "error", "message": str(e)}, status=500)
            else:
                print("Backend: Processing as REGULAR optimization")
                # Regular (non-comparative) optimization
                try:
                    print("Backend: Starting regular optimization")
                    import os
                    import numpy as np  # Move numpy import here for np.isclose() usage
                    # Generate unique run ID
                    run_id = generate_run_id()
                    print(f"Backend: Generated run ID: {run_id}")
                    run_dir = create_run_directory(run_id)
                    print(f"Backend: Created run directory: {run_dir}")
                    
                    # Parse parameters
                    model = data.get('model', 'CASCADE')
                    algorithm = data.get('algorithm', 'Genetic Algorithm')
                    objectives = data.get('objectives', [])
                    traces = data.get('traces', [])
                    population = int(data.get('population_size', data.get('population', 50)))
                    generations = int(data.get('generations', 100))

                    
                    print(f"Backend: Parsed parameters - model: {model}, algorithm: {algorithm}, objectives: {objectives}")
                    print(f"Backend: Traces: {traces}, population: {population}, generations: {generations}")

                    
                    # Validate required fields
                    if not objectives:
                        return JsonResponse({"status": "error", "message": "Objectives are required"}, status=400)
                    if not traces:
                        return JsonResponse({"status": "error", "message": "At least one trace is required"}, status=400)
                    
                    # Validate trace weights sum to 1.0
                    total_weight = sum(trace.get('weight', 1.0) for trace in traces)
                    if not np.isclose(total_weight, 1.0):
                        return JsonResponse({"status": "error", "message": f"Sum of trace weights must be 1.0, got {total_weight}"}, status=400)
                    

                    
                    # Run optimization with simple output directory
                    WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
                    SIMPLE_OUTPUT_DIR = os.path.join(WORKSPACE, 'dse/results')  # Use simple location
                    
                    # Clear the current points.csv for new optimization
                    current_points_file = os.path.join(SIMPLE_OUTPUT_DIR, "points.csv")
                    with open(current_points_file, 'w') as f:
                        f.write("")  # Clear the file
                    print(f"Cleared {current_points_file} for new optimization")
                    
                    # Determine trace name
                    if len(traces) == 1:
                        trace_name = traces[0].get('name', traces[0])
                    else:
                        trace_names = [trace.get('name', trace) for trace in traces]
                        weights = [trace.get('weight', 1.0) for trace in traces]
                        trace_name = generate_weighted_trace(trace_names, weights, label='main')
                    
                    # Create optimization run in database with correct trace name
                    optimization_run = RunStorageService.create_optimization_run(
                        algorithm=algorithm,
                        model=model,
                        population_size=population,
                        generations=generations,
                        objectives=objectives,
                        trace_name=trace_name,
                        trace_sets={'main': traces},
                        name=f"Optimization Run - {algorithm}",
                        description=f"Run with {len(traces)} traces"
                    )
                    
                    # Run optimization
                    ga_result = runGACascade(
                        pop_size=population, 
                        n_gen=generations, 
                        trace=trace_name, 
                        output_dir=SIMPLE_OUTPUT_DIR
                    )
                    
                    # Ensure GA results are JSON-serializable (convert numpy arrays to lists)
                    ga_result = convert_ndarrays(ga_result)
                    print(f"Backend: GA result type after conversion: {type(ga_result)}; length: {len(ga_result) if hasattr(ga_result, '__len__') else 'n/a'}")
                    
                    # Create a timestamped backup of the points.csv AFTER optimization completes
                    import shutil
                    from datetime import datetime
                    if os.path.exists(current_points_file) and os.path.getsize(current_points_file) > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup_file = os.path.join(SIMPLE_OUTPUT_DIR, f"points_backup_{timestamp}.csv")
                        shutil.copy2(current_points_file, backup_file)
                        print(f"Backed up optimization data to: {backup_file}")
                    else:
                        print(f"No data to backup - {current_points_file} is empty or doesn't exist")
                    
                    # Convert results to design points format
                    design_points = []
                    for pt in ga_result:
                        if isinstance(pt, dict):
                            chiplets = pt.get('chiplets', {})
                            design_point = {
                                'execution_time_ms': pt.get('execution_time_ms', pt.get('x', 0)),
                                'energy_mj': pt.get('energy_mj', pt.get('y', 0)),
                                'chiplets': chiplets,
                                'additional_metrics': pt.get('additional_metrics', {}),
                                'context_file_path': pt.get('context_file_path', '')
                            }
                            design_points.append(design_point)
                    
                    # Save results to database
                    RunStorageService.store_design_points(
                        optimization_run, 
                        design_points, 
                        run_dir
                    )
                    
                    # Run analytics
                    try:
                        points_csv_path = os.path.join(SIMPLE_OUTPUT_DIR, "points.csv")  # Use simple location
                        analytics_dir = os.path.join("analytics")
                        os.makedirs(analytics_dir, exist_ok=True)
                        
                        # Rule mining using ChatBot model
                        rule_mining_result = chat_bot.rule_mining()
                        rule_mining_path = os.path.join(analytics_dir, "optimization_rule_mining.json")
                        with open(rule_mining_path, "w") as f:
                            json.dump({"rule_mining": rule_mining_result}, f)
                        
                        # Distance correlation using ChatBot model
                        # First, load the data to get objective and design values
                        import csv
                        objective_vals = []
                        design_vals = []
                        with open(points_csv_path, mode='r') as file:
                            csv_reader = csv.reader(file)
                            for row in csv_reader:
                                objective_vals.append([float(row[0]), float(row[1])])  # time, energy
                                design_vals.append([int(row[2]), int(row[3]), int(row[4]), int(row[5])])  # GPU, Attention, Sparse, Convolution
                        
                        distance_corr_result = chat_bot.get_distance_correlations(objective_vals, design_vals)
                        dist_corr_path = os.path.join(analytics_dir, "optimization_distance_correlation.json")
                        with open(dist_corr_path, "w") as f:
                            json.dump({"distance_correlation": distance_corr_result}, f)
                        
                        analytics_results = {
                            'rule_mining': rule_mining_path,
                            'distance_correlation': dist_corr_path
                        }
                    except Exception as e:
                        analytics_results = {
                            'rule_mining': f"error: {str(e)}",
                            'distance_correlation': f"error: {str(e)}"
                        }
                    
                    # Complete run in database
                    RunStorageService.complete_run(optimization_run, analytics_results=analytics_results)
                    
                    # STEP 1: Automatically inject run context into ChatBot
                    def generate_run_summary(run_id, config, results, pareto_points):
                        return f"""✅ Optimization Run Complete: {run_id}

📊 Run Summary:
• Model: {config['model']}
• Algorithm: {config['algorithm']}
• Objectives: {', '.join(config['objectives'])}
• Trace: {config.get('trace_name', traces[0].get('name', 'Unknown') if traces else 'Unknown')}
• Population Size: {config['population']}
• Generations: {config['generations']}
• Design Points Generated: {len(results)}
• Pareto Front Size: {len(pareto_points)}

The optimization has finished and I now have context about this run. I can help you analyze the results!"""
                    
                    def generate_analytics_insights(rule_mining_result, distance_corr_result):
                        # Extract key insights from analytics results
                        insights = []
                        
                        # Rule mining insights
                        if rule_mining_result and "rule_mining" in rule_mining_result:
                            rule_str = rule_mining_result["rule_mining"]
                            # Count rules found
                            rule_count = len(re.findall(r"Rule:", rule_str))
                            if rule_count > 0:
                                insights.append(f"🔍 Rule Mining: Found {rule_count} significant design patterns in Pareto-optimal solutions")
                        
                        # Distance correlation insights
                        if distance_corr_result and "distance_correlation" in distance_corr_result:
                            corr_str = distance_corr_result["distance_correlation"]
                            # Extract highest correlations
                            corr_matches = re.findall(r"Distance correlation between objective (\w+) and chiplet type (\w+) is ([\d.]+)", corr_str)
                            if corr_matches:
                                highest_corr = max(corr_matches, key=lambda x: float(x[2]))
                                insights.append(f"📈 Distance Correlation: {highest_corr[1]} chiplets show strongest impact on {highest_corr[0]} ({float(highest_corr[2]):.3f})")
                        
                        if insights:
                            return "\n".join(insights)
                        else:
                            return "📊 Analytics completed - ready for detailed analysis"
                    
                    def generate_followup_suggestions():
                        return """💬 You can ask me about:

🎯 **Run Analysis:**
• "What are the key findings from this optimization run?"
• "Which designs are on the Pareto front?"
• "How does this run compare to previous ones?"

🔍 **Data Mining Insights:**
• "What patterns do the best designs share?"
• "Which chiplet type most affects energy consumption?"
• "Show me the rule mining results"

📊 **Specific Designs:**
• "Analyze design point [X,Y] on the plot"
• "What makes this design optimal?"
• "How can I improve this design?"

🔄 **Comparative Analysis:**
• "Compare this run with the previous one"
• "What changed when I modified the parameters?"
• "Which trace performs better?"

💡 **Design Recommendations:**
• "Suggest improvements for energy efficiency"
• "What's the optimal chiplet configuration?"
• "How should I adjust my design constraints?"

Just ask naturally - I have full context of your optimization run!"""
                    
                    # Calculate Pareto front
                    def get_pareto_front(points):
                        """Find Pareto optimal points"""
                        pareto_front = []
                        for i, point in enumerate(points):
                            is_dominated = False
                            for j, other_point in enumerate(points):
                                if i != j:
                                    # Check if other_point dominates point (both objectives minimized)
                                    if (other_point['x'] <= point['x'] and other_point['y'] <= point['y']) and \
                                       (other_point['x'] < point['x'] or other_point['y'] < point['y']):
                                        is_dominated = True
                                        break
                                if not is_dominated:
                                    pareto_front.append(point)
                        return pareto_front
                    
                    # --- FIX: Always assign frontend_results before use ---
                    frontend_results = []
                    for pt in ga_result:
                        if isinstance(pt, dict):
                            frontend_results.append({
                                'x': pt.get('execution_time_ms', pt.get('x', 0)),
                                'y': pt.get('energy_mj', pt.get('y', 0)),
                                'gpu': pt.get('chiplets', {}).get('GPU', 0),
                                'attn': pt.get('chiplets', {}).get('Attention', 0),
                                'sparse': pt.get('chiplets', {}).get('Sparse', 0),
                                'conv': pt.get('chiplets', {}).get('Convolution', 0),
                                'algorithm': algorithm,
                                'trace': traces[0].get('name', '') if len(traces) == 1 else 'weighted_trace'
                            })
                        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            frontend_results.append({
                                'x': float(pt[0]),
                                'y': float(pt[1]),
                                'algorithm': algorithm,
                                'trace': traces[0].get('name', '') if len(traces) == 1 else 'weighted_trace'
                            })
                    
                    print(f"Backend: Converted {len(frontend_results)} points for frontend")
                    
                    # Create zip file with complete run data
                    try:
                        import zipfile
                        from datetime import datetime
                        
                        # Create zip file name based on timestamp (same format as backup files)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        zip_filename = f"run_{timestamp}.zip"
                        zip_path = os.path.join(SIMPLE_OUTPUT_DIR, zip_filename)
                        
                        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                            # Add points.csv to zip
                            points_csv_path = os.path.join(SIMPLE_OUTPUT_DIR, "points.csv")
                            if os.path.exists(points_csv_path):
                                zipf.write(points_csv_path, "points.csv")
                                print(f"Added points.csv to zip: {zip_filename}")
                            
                            # Generate and add report.txt to zip
                            try:
                                # Generate report content
                                report_content = f"""Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Run ID: {run_id}

Configuration:
- Model: {model}
- Algorithm: {algorithm}
- Objectives: {', '.join(objectives)}
- Population Size: {population}
- Generations: {generations}
- Traces: {[trace.get('name', trace) for trace in traces]}

Results Summary:
- Total Design Points: {len(frontend_results)}
- Pareto Optimal Points: {len([p for p in frontend_results if p.get('pareto_optimal', False)])}

Design Points:
"""
                                for i, point in enumerate(frontend_results):
                                    report_content += f"""
Point {i+1}:
- Execution Time: {point.get('x', 0):.2f} ms
- Energy: {point.get('y', 0):.2f} mJ
- GPU Chiplets: {point.get('gpu', 0)}
- Attention Chiplets: {point.get('attn', 0)}
- Sparse Chiplets: {point.get('sparse', 0)}
- Convolution Chiplets: {point.get('conv', 0)}
"""
                                # Add report to zip
                                zipf.writestr("report.txt", report_content)
                                print(f"Added report.txt to zip: {zip_filename}")
                                
                            except Exception as e:
                                print(f"Error generating report for zip: {e}")
                                zipf.writestr("report.txt", f"Error generating report: {str(e)}")
                            
                            # Add metadata.json to zip
                            metadata = {
                                "run_id": run_id,
                                "timestamp": datetime.now().isoformat(),
                                "model": model,
                                "algorithm": algorithm,
                                "objectives": objectives,
                                "traces": traces,
                                "population_size": population,
                                "generations": generations,
                                "total_points": len(frontend_results),
                                "analytics_results": analytics_results
                            }
                            zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
                            print(f"Added metadata.json to zip: {zip_filename}")
                        
                        print(f"Successfully created zip file: {zip_path}")
                        
                    except Exception as e:
                        print(f"Error creating zip file: {e}")
                        # Don't fail the entire request if zip creation fails
                    
                    response = {
                        "status": "success",
                        "data": frontend_results,
                        "plot_data": frontend_results,
                        "run_id": optimization_run.run_id,
                        "run_directory": run_id,  # Add the run directory ID
                        "metadata": {
                            "model": model,
                            "algorithm": algorithm,
                            "objectives": objectives,
                            "traces": traces,
                            "population": population,
                            "generations": generations
                        }
                    }
                    
                    print(f"Backend: Sending response with run_directory: {run_id}")
                    print(f"Backend: Response structure: {response}")
                    
                    return JsonResponse(response)
                except Exception as e:
                    print(f"Backend: Error during optimization: {str(e)}")
                    print(f"Backend: Error type: {type(e)}")
                    import traceback
                    print(f"Backend: Traceback: {traceback.format_exc()}")
                    # --- FIX: Always assign frontend_results to avoid UnboundLocalError ---
                    frontend_results = []
                    return JsonResponse({"status": "error", "message": str(e)}, status=500)
        except json.JSONDecodeError:
            return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)
    else:
        return JsonResponse({"status": "error", "message": "Only POST method allowed"}, status=405)

@api_view(["GET"])
def generate_optimization_report(request):
    """
    Generate a downloadable report for a single optimization run.
    """
    try:
        import os
        import json as pyjson
        from datetime import datetime
        from api.models import OptimizationRun
        
        # Get current timestamp
        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H%M%S")
        
        # Get run_id from request parameters
        run_id = request.GET.get('run_id')
        
        # Fetch run parameters from database
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
                run_params = {
                    'model': optimization_run.model,
                    'algorithm': optimization_run.get_algorithm_display(),
                    'objectives': optimization_run.objectives,
                    'population_size': optimization_run.population_size,
                    'generations': optimization_run.generations,
                    'trace_name': optimization_run.trace_name or 'gpt-j-65536-weighted'
                }
                print(f"Fetched run parameters for {run_id}: {run_params}")
            except OptimizationRun.DoesNotExist:
                print(f"OptimizationRun with run_id {run_id} not found, trying to get most recent run")
                # Try to get the most recent completed run
                try:
                    most_recent_run = OptimizationRun.objects.filter(
                        status='completed'
                    ).order_by('-created_at').first()
                    
                    if most_recent_run:
                        run_params = {
                            'model': most_recent_run.model,
                            'algorithm': most_recent_run.get_algorithm_display(),
                            'objectives': most_recent_run.objectives,
                            'population_size': most_recent_run.population_size,
                            'generations': most_recent_run.generations,
                            'trace_name': most_recent_run.trace_name or 'gpt-j-65536-weighted'
                        }
                        print(f"Using parameters from most recent run {most_recent_run.run_id}: {run_params}")
                    else:
                        print("No completed runs found, using default parameters")
                except Exception as e:
                    print(f"Error fetching most recent run: {e}, using default parameters")
            except Exception as e:
                print(f"Error fetching run parameters: {e}, using default parameters")
        else:
            # No run_id provided, try to get the most recent completed run
            try:
                most_recent_run = OptimizationRun.objects.filter(
                    status='completed'
                ).order_by('-created_at').first()
                
                if most_recent_run:
                    run_params = {
                        'model': most_recent_run.model,
                        'algorithm': most_recent_run.get_algorithm_display(),
                        'objectives': most_recent_run.objectives,
                        'population_size': most_recent_run.population_size,
                        'generations': most_recent_run.generations,
                        'trace_name': most_recent_run.trace_name or 'gpt-j-65536-weighted'
                    }
                    print(f"Using parameters from most recent run {most_recent_run.run_id}: {run_params}")
                else:
                    print("No completed runs found, using default parameters")
            except Exception as e:
                print(f"Error fetching most recent run: {e}, using default parameters")
        
        # Load data from CSV
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        import csv
        points = []
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                points.append({
                    'x': float(row[0]),  # time
                    'y': float(row[1]),  # energy
                    'gpu': float(row[2]),
                    'attn': float(row[3]),
                    'sparse': float(row[4]),
                    'conv': float(row[5])
                })
        
        # Get Pareto front using proper Pareto dominance
        def is_pareto_dominant(point1, point2):
            """Check if point1 dominates point2 (both objectives should be minimized)"""
            return (point1['x'] <= point2['x'] and point1['y'] <= point2['y']) and \
                   (point1['x'] < point2['x'] or point1['y'] < point2['y'])
        
        def get_pareto_front(points):
            """Find Pareto optimal points"""
            pareto_front = []
            for i, point in enumerate(points):
                is_dominated = False
                for j, other_point in enumerate(points):
                    if i != j and is_pareto_dominant(other_point, point):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_front.append(point)
            return pareto_front
        
        pareto_points = get_pareto_front(points)
        
        # Get rule mining results
        rule_mining_str = chat_bot.rule_mining()
        
        # Parse rules for table format
        rules = []
        rule_pattern = re.compile(r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), conf\(p->f\): ([0-9.eE+-]+), lift: \[([0-9.eE+-]+)\]")
        for match in rule_pattern.finditer(rule_mining_str):
            rules.append({
                "rule": match.group(1),
                "conf_p_to_f": float(match.group(2)),
                "conf_f_to_p": float(match.group(3)),
                "lift": float(match.group(4)),
            })
        
        # Get distance correlation
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        gpus = [p['gpu'] for p in points]
        attns = [p['attn'] for p in points]
        sparses = [p['sparse'] for p in points]
        convs = [p['conv'] for p in points]
        
        correlations = {
            "GPU_vs_Energy": float(dcor.distance_correlation(np.array(gpus), np.array(ys))),
            "Attention_vs_Energy": float(dcor.distance_correlation(np.array(attns), np.array(ys))),
            "Sparse_vs_Energy": float(dcor.distance_correlation(np.array(sparses), np.array(ys))),
            "Convolution_vs_Energy": float(dcor.distance_correlation(np.array(convs), np.array(ys))),
            "GPU_vs_Time": float(dcor.distance_correlation(np.array(gpus), np.array(xs))),
            "Attention_vs_Time": float(dcor.distance_correlation(np.array(attns), np.array(xs))),
            "Sparse_vs_Time": float(dcor.distance_correlation(np.array(sparses), np.array(xs))),
            "Convolution_vs_Time": float(dcor.distance_correlation(np.array(convs), np.array(xs))),
        }
        
        # Format functions
        def format_rule_natural_language(rule_str):
            """Convert underscore-separated rules to natural language"""
            if not rule_str:
                return rule_str
            
            parts = re.split(r'\s+(AND|OR)\s+', rule_str)
            formatted_parts = []
            
            for i, part in enumerate(parts):
                if part == 'AND' or part == 'OR':
                    formatted_parts.append(part)
                    continue
                
                formatted_part = format_condition_natural_language(part)
                formatted_parts.append(formatted_part)
            
            return ' '.join(formatted_parts)
        
        def format_condition_natural_language(condition):
            """Format individual conditions like 'Attention_none' to 'None Attention Chiplets'"""
            if not condition:
                return condition
            
            condition = condition.strip()
            chiplet_pattern = r'^(\w+)_(\w+)$'
            match = re.match(chiplet_pattern, condition)
            
            if match:
                chiplet_type, level = match.groups()
                
                chiplet_names = {
                    'GPU': 'GPU',
                    'Attention': 'Attention', 
                    'Sparse': 'Sparse',
                    'Convolution': 'Convolution'
                }
                
                level_names = {
                    'none': 'None',
                    'low': 'Low',
                    'medium': 'Medium',
                    'high': 'High'
                }
                
                chiplet_name = chiplet_names.get(chiplet_type, chiplet_type)
                level_name = level_names.get(level, level)
                
                return f"{level_name} {chiplet_name} Chiplets"
            
            return condition
        
        # Generate HTML report content
        def format_rule_natural_language(rule_str):
            """Convert rule string to natural language description"""
            # Remove technical formatting and make it more readable
            rule = rule_str.replace("GPU=", "GPU count = ").replace("ATTN=", "Attention count = ").replace("SPARSE=", "Sparse count = ").replace("CONV=", "Convolution count = ")
            rule = rule.replace("AND", " and ").replace("OR", " or ")
            return rule
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Report - {date_str}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #e8f4fd;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .header-info {{
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .header-info p {{
            margin: 5px 0;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimization Report</h1>
        
        <div class="header-info">
            <p><strong>Date:</strong> {date_str}</p>
            <p><strong>Time:</strong> {time_str}</p>
            <p><strong>Report Type:</strong> Tradespace Exploration Run</p>
        </div>

        <div class="section">
            <h2>Problem Formulation</h2>
            <p><strong>Decisions:</strong></p>
            <ul>
                <li>Chiplet types: GPU, Attention, Convolution, Sparse</li>
                <li>Number of slots: 12</li>
            </ul>
            <p><strong>Objectives:</strong></p>
            <ul>
                <li>Optimization type: Multi-objective</li>
                {''.join([f'<li>Objective {i+1}: Minimize {objective.lower()}</li>' for i, objective in enumerate(run_params['objectives'])])}
            </ul>
            <p><strong>Models and parameters:</strong></p>
            <ul>
                <li>Model: {run_params['model']}</li>
                <li>Trace: {run_params['trace_name']}</li>
                <li>Search Algorithm: {run_params['algorithm']}</li>
                <li>Population Size: {run_params['population_size']}</li>
                <li>Generation Size: {run_params['generations']}</li>
            </ul>
        </div>

        <div class="section">
            <h2>Results Summary</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(points)}</div>
                    <div class="stat-label">Total Designs Evaluated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(pareto_points)}</div>
                    <div class="stat-label">Designs on Pareto Front</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Pareto Front Composition</h2>
            <p>X → time (ms), Y → Energy (mJ)</p>
            <table>
                <thead>
                    <tr>
                        <th>GPU</th>
                        <th>Attention</th>
                        <th>Sparse</th>
                        <th>Convolution</th>
                        <th>Time (ms)</th>
                        <th>Energy (mJ)</th>
                    </tr>
                </thead>
                <tbody>"""

        for point in pareto_points:
            html_content += f"""
                    <tr>
                        <td>{int(point['gpu'])}</td>
                        <td>{int(point['attn'])}</td>
                        <td>{int(point['sparse'])}</td>
                        <td>{int(point['conv'])}</td>
                        <td>{point['x']:.2f}</td>
                        <td>{point['y']:.2f}</td>
                    </tr>"""

        html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Association Rule Mining Results</h2>
            <p>Rules of the form F ⇒ Pareto (each row contains a text description of the feature F and importance measures)</p>
            <table>
                <thead>
                    <tr>
                        <th>Rule Description</th>
                        <th>Conf(F→P)</th>
                        <th>Conf(P→F)</th>
                        <th>Lift</th>
                    </tr>
                </thead>
                <tbody>"""

        for rule in rules:
            formatted_rule = format_rule_natural_language(rule['rule'])
            html_content += f"""
                    <tr>
                        <td>{formatted_rule}</td>
                        <td>{rule['conf_f_to_p']:.2f}</td>
                        <td>{rule['conf_p_to_f']:.2f}</td>
                        <td>{rule['lift']:.2f}</td>
                    </tr>"""

        html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Distance Correlation Analysis</h2>
            <p>Correlation between chiplet types and objectives</p>
            <table>
                <thead>
                    <tr>
                        <th>Chiplet Type</th>
                        <th>vs Energy</th>
                        <th>vs Time</th>
                    </tr>
                </thead>
                <tbody>"""

        # Add distance correlation results
        chiplet_types = ['GPU', 'Attention', 'Sparse', 'Convolution']
        for i, chiplet in enumerate(chiplet_types):
            html_content += f"""
                    <tr>
                        <td>{chiplet}</td>
                        <td>{correlations[f'{chiplet}_vs_Energy']:.3f}</td>
                        <td>{correlations[f'{chiplet}_vs_Time']:.3f}</td>
                    </tr>"""

        html_content += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

        # Save report to file system for future retrieval
        import os
        reports_dir = os.path.join("api/Evaluator/cascade/chiplet_model/dse/results/reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        # Generate report filename based on timestamp or run_id
        if run_id:
            report_filename = f"report_{run_id}_{time_str}.html"
        else:
            report_filename = f"report_{date_str}_{time_str}.html"
        
        report_path = os.path.join(reports_dir, report_filename)
        
        # Save the HTML report
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Also save metadata for future reference
        metadata = {
            'run_params': run_params,
            'timestamp': time_str,
            'date': date_str,
            'total_points': len(points),
            'pareto_points': len(pareto_points),
            'report_filename': report_filename
        }
        
        metadata_path = report_path.replace('.html', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Create download link
        web_link = f"/api/Evaluator/cascade/chiplet_model/dse/results/reports/{report_filename}"
        
        return Response({
            "status": "success",
            "message": "Report generated successfully",
            "web_link": web_link,
            "download_link": web_link,
            "report_filename": report_filename,
            "metadata": metadata
        })
        
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)

@api_view(["GET"])
def get_designs_by_constraint(request):
    """
    Get designs matching specific constraints for highlighting on the plot.
    """
    try:
        print(f"[get_designs_by_constraint] Called with request: {request}")
        print(f"[get_designs_by_constraint] Request.GET: {request.GET}")
        
        # Parse constraint parameters
        chiplet_type = request.GET.get("chiplet_type", "").lower()  # gpu, attention, sparse, convolution
        constraint_level = request.GET.get("constraint_level", "").lower()  # none, low, medium, high
        
        print(f"[get_designs_by_constraint] Parsed parameters: chiplet_type='{chiplet_type}', constraint_level='{constraint_level}'")
        
        # Load data from CSV
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        import csv
        import os
        
        print(f"[get_designs_by_constraint] Checking if file exists: {file_path}")
        if not os.path.exists(file_path):
            print(f"[get_designs_by_constraint] File does not exist: {file_path}")
            return Response({
                "error": "No optimization data available for highlighting. Please run an optimization first to generate design data.",
                "no_data": True
            }, status=404)
        
        points = []
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                points.append({
                    'x': float(row[0]),  # time
                    'y': float(row[1]),  # energy
                    'gpu': float(row[2]),
                    'attn': float(row[3]),
                    'sparse': float(row[4]),
                    'conv': float(row[5])
                })
        
        print(f"[get_designs_by_constraint] Loaded {len(points)} points from CSV")
        
        # Define constraint mapping
        constraint_mapping = {
            'gpu': {
                'none': lambda p: p['gpu'] == 0,
                'low': lambda p: 1 <= p['gpu'] <= 3,
                'medium': lambda p: 4 <= p['gpu'] <= 6,
                'high': lambda p: p['gpu'] >= 7
            },
            'attention': {
                'none': lambda p: p['attn'] == 0,
                'low': lambda p: 1 <= p['attn'] <= 2,
                'medium': lambda p: 3 <= p['attn'] <= 5,
                'high': lambda p: p['attn'] >= 6
            },
            'sparse': {
                'none': lambda p: p['sparse'] == 0,
                'low': lambda p: 1 <= p['sparse'] <= 3,
                'medium': lambda p: 4 <= p['sparse'] <= 7,
                'high': lambda p: p['sparse'] >= 8
            },
            'convolution': {
                'none': lambda p: p['conv'] == 0,
                'low': lambda p: 1 <= p['conv'] <= 2,
                'medium': lambda p: 3 <= p['conv'] <= 4,
                'high': lambda p: p['conv'] >= 5
            }
        }
        
        print(f"[get_designs_by_constraint] Available constraint types: {list(constraint_mapping.keys())}")
        print(f"[get_designs_by_constraint] Available levels for {chiplet_type}: {list(constraint_mapping.get(chiplet_type, {}).keys())}")
        
        # Find matching designs
        matching_designs = []
        if chiplet_type in constraint_mapping and constraint_level in constraint_mapping[chiplet_type]:
            constraint_func = constraint_mapping[chiplet_type][constraint_level]
            matching_designs = [point for point in points if constraint_func(point)]
            print(f"[get_designs_by_constraint] Found {len(matching_designs)} matching designs")
        else:
            print(f"[get_designs_by_constraint] Invalid constraint: chiplet_type='{chiplet_type}' not in {list(constraint_mapping.keys())} or constraint_level='{constraint_level}' not in {list(constraint_mapping.get(chiplet_type, {}).keys())}")
        
        # Return matching designs with indices for highlighting
        highlighted_points = []
        for i, point in enumerate(points):
            is_matching = any(
                point['x'] == match['x'] and point['y'] == match['y'] 
                for match in matching_designs
            )
            highlighted_points.append({
                'index': i,
                'highlighted': is_matching,
                'x': point['x'],
                'y': point['y'],
                'gpu': point['gpu'],
                'attn': point['attn'],
                'sparse': point['sparse'],
                'conv': point['conv']
            })
        
        result = {
            "highlighted_points": highlighted_points,
            "matching_count": len(matching_designs),
            "total_count": len(points),
            "constraint": f"{constraint_level} {chiplet_type} chiplets"
        }
        
        print(f"[get_designs_by_constraint] Returning result: {result}")
        return Response(result)
        
    except Exception as e:
        print(f"[get_designs_by_constraint] Error: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
def add_custom_point(request):
    """
    Add a custom point to a specific run's CSV file.
    """
    try:
        data = json.loads(request.body)
        run_id = data.get('run_id')
        point = data.get('point')
        
        if not run_id or not point:
            return JsonResponse({"status": "error", "message": "run_id and point are required"}, status=400)
        
        # Validate point data
        required_fields = ['x', 'y', 'gpu', 'attn', 'sparse', 'conv']
        for field in required_fields:
            if field not in point:
                return JsonResponse({"status": "error", "message": f"Missing required field: {field}"}, status=400)
        
        # Add point type if not provided
        if 'type' not in point:
            point['type'] = 'custom'
        
        # Write point to run-specific CSV file
        import os
        import csv
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        run_points_path = os.path.join(WORKSPACE, 'dse/results', run_id, 'points.csv')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(run_points_path), exist_ok=True)
        
        # Append point to CSV file
        with open(run_points_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([
                point['x'],
                point['y'],
                point['gpu'],
                point['attn'],
                point['sparse'],
                point['conv'],
                point['type']
            ])
        
        print(f"Added custom point to {run_id}: {point}")
        return JsonResponse({"status": "success", "message": "Custom point added successfully"})
        
    except Exception as e:
        print(f"Error adding custom point: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@api_view(["GET"])
def list_backup_files(request):
    """List available backup files for loading previous runs"""
    try:
        import os
        import glob
        from datetime import datetime
        
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        
        # Find all backup files
        backup_pattern = os.path.join(results_dir, "points_backup_*.csv")
        backup_files = glob.glob(backup_pattern)
        
        # Convert to list of file info
        file_list = []
        for file_path in backup_files:
            filename = os.path.basename(file_path)
            timestamp_str = filename.replace("points_backup_", "").replace(".csv", "")
            
            try:
                # Parse timestamp
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                file_info = {
                    'filename': filename,
                    'timestamp': timestamp.isoformat(),
                    'display_name': f"Run {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    'file_path': file_path
                }
                file_list.append(file_info)
            except ValueError:
                # Skip files with invalid timestamps
                continue
        
        # Sort by timestamp (newest first)
        file_list.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return JsonResponse({
            'status': 'success',
            'backup_files': file_list
        })
        
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': str(e)
        }, status=500)

@api_view(["GET"])
def test_endpoint(request):
    print("🔍 DEBUG: test_endpoint called!")
    return JsonResponse({"status": "success", "message": "Django is working!"})

@api_view(["POST"])
def load_previous_run(request):
    """
    Load a previous optimization run from backup file.
    """
    try:
        import os
        import csv
        import shutil
        import zipfile
        from datetime import datetime
        
        data = json.loads(request.body)
        backup_filename = data.get('backup_filename')
        
        if not backup_filename:
            return JsonResponse({"status": "error", "message": "backup_filename is required"}, status=400)
        
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        backup_file_path = os.path.join(results_dir, backup_filename)
        
        # Check if backup file exists
        if not os.path.exists(backup_file_path):
            return JsonResponse({"status": "error", "message": f"Backup file not found: {backup_filename}"}, status=404)
        
        # Extract timestamp from filename for run ID
        timestamp_str = backup_filename.replace("points_backup_", "").replace(".csv", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            run_id = f"loaded_run_{timestamp_str}"
        except ValueError:
            run_id = f"loaded_run_{backup_filename}"
        
        # Load points from backup file
        points = []
        with open(backup_file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) >= 6:  # Ensure we have at least the basic fields
                    point = {
                        'x': float(row[0]),  # time
                        'y': float(row[1]),  # energy
                        'gpu': float(row[2]),
                        'attn': float(row[3]),
                        'sparse': float(row[4]),
                        'conv': float(row[5]),
                        'type': row[6] if len(row) > 6 else 'optimization'  # Optional type field
                    }
                    points.append(point)
        
        print(f"Loaded {len(points)} points from backup file: {backup_filename}")
        
        # Create temporary points.csv for plotting (don't overwrite current)
        temp_points_path = os.path.join(results_dir, f"temp_points_{run_id}.csv")
        with open(temp_points_path, mode='w', newline='') as file:
            csv_writer = csv.writer(file)
            for point in points:
                csv_writer.writerow([
                    point['x'],
                    point['y'],
                    point['gpu'],
                    point['attn'],
                    point['sparse'],
                    point['conv'],
                    point.get('type', 'optimization')
                ])
        
        # Check if corresponding zip file exists
        zip_filename = f"run_{timestamp_str}.zip"
        zip_path = os.path.join(results_dir, zip_filename)
        has_zip = os.path.exists(zip_path)
        
        # Extract metadata from zip file if it exists
        metadata = None
        if has_zip:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    if 'metadata.json' in zipf.namelist():
                        metadata_content = zipf.read('metadata.json').decode('utf-8')
                        metadata = json.loads(metadata_content)
                        print(f"Extracted metadata from zip: {metadata}")
            except Exception as e:
                print(f"Error extracting metadata from zip: {e}")
                # Don't fail the entire request if metadata extraction fails
        
        # Ensure a DB OptimizationRun exists for this loaded backup, so UI can use /api/runs/compare/
        try:
            # Create the run if not already present
            existing = OptimizationRun.objects.filter(run_id=run_id).first()
            if not existing:
                algo = (metadata or {}).get('algorithm', 'Genetic Algorithm')
                model = (metadata or {}).get('model', 'CASCADE')
                objectives = (metadata or {}).get('objectives', ['Energy', 'Runtime'])
                pop_size = (metadata or {}).get('population_size', 0)
                gens = (metadata or {}).get('generations', 0)
                trace_sets = {
                    'loaded': {
                        'traces': [t.get('name') for t in (metadata or {}).get('traces', [])] if (metadata and metadata.get('traces')) else [],
                        'weights': [t.get('weight') for t in (metadata or {}).get('traces', [])] if (metadata and metadata.get('traces')) else []
                    }
                }
                run = RunStorageService.create_optimization_run(
                    algorithm=algo,
                    model=model,
                    population_size=pop_size or 0,
                    generations=gens or 0,
                    objectives=objectives,
                    trace_name=(trace_sets['loaded']['traces'][0] if trace_sets['loaded']['traces'] else ''),
                    trace_sets=trace_sets,
                    name=f"Loaded {backup_filename}",
                    description="Run loaded from backup CSV",
                    run_id=run_id
                )
                # Store points
                design_points = [
                    {
                        'execution_time_ms': p['x'],
                        'energy_mj': p['y'],
                        'chiplets': {
                            'GPU': int(p.get('gpu', 0)),
                            'Attention': int(p.get('attn', 0)),
                            'Sparse': int(p.get('sparse', 0)),
                            'Convolution': int(p.get('conv', 0))
                        }
                    } for p in points
                ]
                RunStorageService.store_design_points(run, design_points, results_dir)
                RunStorageService.complete_run(run)
        except Exception as e:
            # Log but do not fail the API response
            print(f"Warning: Failed to persist loaded run to DB: {e}")

        return JsonResponse({
            "status": "success",
            "run_id": run_id,
            "backup_filename": backup_filename,
            "timestamp": timestamp.isoformat() if 'timestamp' in locals() else None,
            "points": points,
            "total_points": len(points),
            "temp_points_path": temp_points_path,
            "has_zip_file": has_zip,
            "zip_filename": zip_filename if has_zip else None,
            "metadata": metadata
        })
        
    except Exception as e:
        print(f"Error loading previous run: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@api_view(["GET"])
def get_previous_run_report(request):
    """
    Get or generate a report for a previous run using the same logic as current runs.
    """
    try:
        import os
        import csv
        import re
        import numpy as np
        import dcor
        
        backup_filename = request.GET.get('backup_filename')
        
        if not backup_filename:
            return JsonResponse({"status": "error", "message": "backup_filename parameter is required"}, status=400)
        
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        
        # Extract timestamp from backup filename
        timestamp_str = backup_filename.replace("points_backup_", "").replace(".csv", "")
        run_id = f"loaded_run_{timestamp_str}"
        
        # Check if report already exists
        reports_dir = os.path.join(results_dir, "reports")
        existing_report_path = os.path.join(reports_dir, f"report_{run_id}_*.html")
        
        # Try to find existing report
        import glob
        existing_reports = glob.glob(existing_report_path)
        
        if existing_reports:
            # Use existing report
            report_path = existing_reports[0]
            with open(report_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Get metadata
            metadata_path = report_path.replace('.html', '_metadata.json')
            metadata = None
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Create web link for the existing report
            report_filename = os.path.basename(report_path)
            web_link = f"/api/Evaluator/cascade/chiplet_model/dse/results/reports/{report_filename}"
            
            return JsonResponse({
                "status": "success",
                "report_content": report_content,
                "metadata": metadata,
                "web_link": web_link,
                "download_link": web_link,
                "loaded_from_backup": True
            })
        
        # Generate new report using the same logic as current runs
        # Load data from backup file
        backup_file_path = os.path.join(results_dir, backup_filename)
        if not os.path.exists(backup_file_path):
            return JsonResponse({"status": "error", "message": f"Backup file not found: {backup_filename}"}, status=404)
        
        points = []
        with open(backup_file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) >= 6:
                    points.append({
                        'x': float(row[0]),  # time
                        'y': float(row[1]),  # energy
                        'gpu': float(row[2]),
                        'attn': float(row[3]),
                        'sparse': float(row[4]),
                        'conv': float(row[5])
                    })
        
        # Use default parameters for loaded runs (since we don't have the original parameters)
        run_params = {
            'model': 'CASCADE',
            'algorithm': 'Genetic Algorithm',
            'objectives': ['Energy', 'Runtime'],
            'population_size': 50,
            'generations': 100,
            'trace_name': 'gpt-j-65536-weighted'
        }
        
        # Try to get metadata from zip file if it exists
        zip_filename = f"run_{timestamp_str}.zip"
        zip_path = os.path.join(results_dir, zip_filename)
        if os.path.exists(zip_path):
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zipf:
                    if 'metadata.json' in zipf.namelist():
                        metadata_content = zipf.read('metadata.json').decode('utf-8')
                        metadata = json.loads(metadata_content)
                        
                        # Update run parameters with actual values
                        if metadata:
                            run_params.update({
                                'model': metadata.get('model', run_params['model']),
                                'algorithm': metadata.get('algorithm', run_params['algorithm']),
                                'objectives': metadata.get('objectives', run_params['objectives']),
                                'population_size': metadata.get('population_size', run_params['population_size']),
                                'generations': metadata.get('generations', run_params['generations']),
                                'trace_name': metadata.get('trace_name', run_params['trace_name'])
                            })
            except Exception as e:
                print(f"Error extracting metadata from zip: {e}")
        
        # Generate report using the same logic as generate_optimization_report
        # This will save the report for future use
        from datetime import datetime
        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H%M%S")
        
        # Get Pareto front
        def is_pareto_dominant(point1, point2):
            return (point1['x'] <= point2['x'] and point1['y'] <= point2['y']) and \
                   (point1['x'] < point2['x'] or point1['y'] < point2['y'])
        
        def get_pareto_front(points):
            pareto_front = []
            for i, point in enumerate(points):
                is_dominated = False
                for j, other_point in enumerate(points):
                    if i != j and is_pareto_dominant(other_point, point):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_front.append(point)
            return pareto_front
        
        pareto_points = get_pareto_front(points)
        
        # Get rule mining results
        from api.ChatBot.model import ChatBotModel
        chat_bot = ChatBotModel()
        
        # Use the backup file for rule mining
        point_selection_params = {
            "file_path": backup_file_path
        }
        rule_mining_str = chat_bot.rule_mining(point_selection_params)
        
        # Parse rules
        rules = []
        rule_pattern = re.compile(r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), conf\(p->f\): ([0-9.eE+-]+), lift: \[([0-9.eE+-]+)\]")
        for match in rule_pattern.finditer(rule_mining_str):
            rules.append({
                "rule": match.group(1),
                "conf_p_to_f": float(match.group(2)),
                "conf_f_to_p": float(match.group(3)),
                "lift": float(match.group(4)),
            })
        
        # Get distance correlation
        xs = [p['x'] for p in points]
        ys = [p['y'] for p in points]
        gpus = [p['gpu'] for p in points]
        attns = [p['attn'] for p in points]
        sparses = [p['sparse'] for p in points]
        convs = [p['conv'] for p in points]
        
        correlations = {
            "GPU_vs_Energy": float(dcor.distance_correlation(np.array(gpus), np.array(ys))),
            "Attention_vs_Energy": float(dcor.distance_correlation(np.array(attns), np.array(ys))),
            "Sparse_vs_Energy": float(dcor.distance_correlation(np.array(sparses), np.array(ys))),
            "Convolution_vs_Energy": float(dcor.distance_correlation(np.array(convs), np.array(ys))),
            "GPU_vs_Time": float(dcor.distance_correlation(np.array(gpus), np.array(xs))),
            "Attention_vs_Time": float(dcor.distance_correlation(np.array(attns), np.array(xs))),
            "Sparse_vs_Time": float(dcor.distance_correlation(np.array(sparses), np.array(xs))),
            "Convolution_vs_Time": float(dcor.distance_correlation(np.array(convs), np.array(xs))),
        }
        
        # Generate HTML report (same as generate_optimization_report)
        # ... (HTML generation code would be the same as in generate_optimization_report)
        # For brevity, I'll create a simplified version
        
        def format_rule_natural_language(rule_str):
            """Convert rule string to natural language description"""
            # Remove technical formatting and make it more readable
            rule = rule_str.replace("GPU=", "GPU count = ").replace("ATTN=", "Attention count = ").replace("SPARSE=", "Sparse count = ").replace("CONV=", "Convolution count = ")
            rule = rule.replace("AND", " and ").replace("OR", " or ")
            return rule
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Previous Run Report - {timestamp_str}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }}
        h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; }}
        .section {{ margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #e8f4fd; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
        .header-info {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .header-info p {{ margin: 5px 0; color: #2c3e50; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Previous Run Report</h1>
        
        <div class="header-info">
            <p><strong>Original Date:</strong> {timestamp_str}</p>
            <p><strong>Report Generated:</strong> {date_str} {time_str}</p>
            <p><strong>Report Type:</strong> Loaded Previous Run</p>
        </div>

        <div class="section">
            <h2>Problem Formulation</h2>
            <p><strong>Decisions:</strong></p>
            <ul>
                <li>Chiplet types: GPU, Attention, Convolution, Sparse</li>
                <li>Number of slots: 12</li>
            </ul>
            <p><strong>Objectives:</strong></p>
            <ul>
                <li>Optimization type: Multi-objective</li>
                {''.join([f'<li>Objective {i+1}: Minimize {objective.lower()}</li>' for i, objective in enumerate(run_params['objectives'])])}
            </ul>
            <p><strong>Models and parameters:</strong></p>
            <ul>
                <li>Model: {run_params['model']}</li>
                <li>Trace: {run_params['trace_name']}</li>
                <li>Search Algorithm: {run_params['algorithm']}</li>
                <li>Population Size: {run_params['population_size']}</li>
                <li>Generation Size: {run_params['generations']}</li>
            </ul>
        </div>

        <div class="section">
            <h2>Results Summary</h2>
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{len(points)}</div>
                    <div class="stat-label">Total Designs Evaluated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(pareto_points)}</div>
                    <div class="stat-label">Designs on Pareto Front</div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Pareto Front Composition</h2>
            <p>X → time (ms), Y → Energy (mJ)</p>
            <table>
                <thead>
                    <tr>
                        <th>GPU</th>
                        <th>Attention</th>
                        <th>Sparse</th>
                        <th>Convolution</th>
                        <th>Time (ms)</th>
                        <th>Energy (mJ)</th>
                    </tr>
                </thead>
                <tbody>"""

        for point in pareto_points:
            html_content += f"""
                    <tr>
                        <td>{int(point['gpu'])}</td>
                        <td>{int(point['attn'])}</td>
                        <td>{int(point['sparse'])}</td>
                        <td>{int(point['conv'])}</td>
                        <td>{point['x']:.2f}</td>
                        <td>{point['y']:.2f}</td>
                    </tr>"""

        html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Association Rule Mining Results</h2>
            <p>Rules of the form F ⇒ Pareto (each row contains a text description of the feature F and importance measures)</p>
            <table>
                <thead>
                    <tr>
                        <th>Rule Description</th>
                        <th>Conf(F→P)</th>
                        <th>Conf(P→F)</th>
                        <th>Lift</th>
                    </tr>
                </thead>
                <tbody>"""

        for rule in rules:
            formatted_rule = format_rule_natural_language(rule['rule'])
            html_content += f"""
                    <tr>
                        <td>{formatted_rule}</td>
                        <td>{rule['conf_f_to_p']:.2f}</td>
                        <td>{rule['conf_p_to_f']:.2f}</td>
                        <td>{rule['lift']:.2f}</td>
                    </tr>"""

        html_content += """
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Distance Correlation Analysis</h2>
            <p>Correlation between chiplet types and objectives</p>
            <table>
                <thead>
                    <tr>
                        <th>Chiplet Type</th>
                        <th>vs Energy</th>
                        <th>vs Time</th>
                    </tr>
                </thead>
                <tbody>"""

        chiplet_types = ['GPU', 'Attention', 'Sparse', 'Convolution']
        for chiplet in chiplet_types:
            html_content += f"""
                    <tr>
                        <td>{chiplet}</td>
                        <td>{correlations[f'{chiplet}_vs_Energy']:.3f}</td>
                        <td>{correlations[f'{chiplet}_vs_Time']:.3f}</td>
                    </tr>"""

        html_content += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>"""

        # Save the generated report for future use
        os.makedirs(reports_dir, exist_ok=True)
        report_filename = f"report_{run_id}_{time_str}.html"
        report_path = os.path.join(reports_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save metadata
        metadata = {
            'run_params': run_params,
            'timestamp': time_str,
            'date': date_str,
            'total_points': len(points),
            'pareto_points': len(pareto_points),
            'report_filename': report_filename,
            'backup_filename': backup_filename
        }
        
        metadata_path = report_path.replace('.html', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Create web link for the report
        web_link = f"/api/Evaluator/cascade/chiplet_model/dse/results/reports/{report_filename}"
        
        return JsonResponse({
            "status": "success",
            "report_content": html_content,
            "metadata": metadata,
            "web_link": web_link,
            "download_link": web_link,
            "loaded_from_backup": True
        })
        
    except Exception as e:
        print(f"Error in get_previous_run_report: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@api_view(["POST"])
def data_mining_followup(request):
    """
    Handle follow-up questions about data mining results with context-aware responses.
    """
    try:
        from api.ChatBot.model import ChatBotModel
        chat_bot = ChatBotModel()
        
        data = json.loads(request.body)
        question = data.get("question")
        data_mining_type = data.get("data_mining_type")  # "rule_mining" or "distance_correlation"
        structured_data = data.get("structured_data", {})
        
        if not question or not data_mining_type:
            return Response({
                "error": "question and data_mining_type are required"
            }, status=400)
        
        response = chat_bot.get_data_mining_followup_response(
            question, data_mining_type, structured_data
        )
        
        return Response({"response": response})
    except Exception as e:
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
def add_enhanced_insights_context(request):
    """
    Add both summary and detailed context to the AI's conversation history.
    """
    try:
        data = json.loads(request.body)
        summary_insights = data.get("summary_insights")
        detailed_context = data.get("detailed_context")
        
        if not summary_insights:
            return Response({"error": "summary_insights is required"}, status=400)
        
        # Create enhanced context message
        context_message = f"Here are the insights from the analysis:\n\n{summary_insights}\n\n"
        
        if detailed_context:
            context_message += f"I have detailed analysis available for this design point including:\n"
            context_message += f"- Per-chiplet energy breakdown and execution time\n"
            context_message += f"- Memory access patterns and bottlenecks\n"
            context_message += f"- Work distribution across chiplets\n"
            context_message += f"- Energy efficiency metrics\n\n"
            context_message += f"You can ask detailed questions like:\n"
            context_message += f"- 'What is the energy bottleneck for this design?'\n"
            context_message += f"- 'Which chiplet is consuming the most memory?'\n"
            context_message += f"- 'How is the work distributed across chiplets?'\n"
            context_message += f"- 'What's the energy efficiency of each component?'\n\n"
            context_message += f"Detailed context data is available for analysis."
        else:
            context_message += f"I have this context and can answer follow-up questions about these insights."
        
        # Add the enhanced context to AI memory
        chat_bot.messages.append({
            "role": "assistant",
            "content": context_message
        })
        
        return Response({"message": "Enhanced insights context added successfully"})
        
    except Exception as e:
        print(f"Error in add_enhanced_insights_context: {e}")
        import traceback
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)

@api_view(["POST"])
def integrate_custom_point_to_ga(request):
    """
    Integrate a custom point into the current GA generation.
    This will add the point to the current population and continue the GA.
    """
    try:
        data = json.loads(request.body)
        run_id = data.get('run_id')
        custom_point = data.get('custom_point')
        current_generation = data.get('current_generation', 0)
        
        if not run_id or not custom_point:
            return JsonResponse({"status": "error", "message": "run_id and custom_point are required"}, status=400)
        
        # Validate custom point data
        required_fields = ['gpu', 'attn', 'sparse', 'conv']
        for field in required_fields:
            if field not in custom_point:
                return JsonResponse({"status": "error", "message": f"Missing required field: {field}"}, status=400)
        
        # Convert chiplet configuration to GA format (12-element array)
        ga_point = []
        for i in range(12):
            if i < custom_point['gpu']:
                ga_point.append(0)  # GPU
            elif i < custom_point['gpu'] + custom_point['attn']:
                ga_point.append(1)  # Attention
            elif i < custom_point['gpu'] + custom_point['attn'] + custom_point['sparse']:
                ga_point.append(2)  # Sparse
            elif i < custom_point['gpu'] + custom_point['attn'] + custom_point['sparse'] + custom_point['conv']:
                ga_point.append(3)  # Convolution
            else:
                ga_point.append(0)  # Default to GPU for remaining slots
        
        # Get current population from the run
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        run_dir = os.path.join(WORKSPACE, 'dse/results', run_id)
        population_file = os.path.join(run_dir, 'current_population.json')
        
        current_population = []
        if os.path.exists(population_file):
            with open(population_file, 'r') as f:
                current_population = json.load(f)
        
        # Add custom point to current population
        current_population.append(ga_point)
        
        # Save updated population
        with open(population_file, 'w') as f:
            json.dump(current_population, f)
        
        # Continue GA with updated population
        # Get run configuration
        config_file = os.path.join(run_dir, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Continue GA with custom point in population
            trace = config.get('trace', 'gpt-j-65536-weighted')
            pop_size = config.get('population_size', 10)
            generations = config.get('generations', 5)
            
            # Run GA with updated population
            ga_result = runGACascade(
                pop_size=pop_size, 
                n_gen=generations - current_generation,  # Continue for remaining generations
                trace=trace, 
                initial_population=np.array(current_population),
                output_dir=run_dir
            )
            
            # Update results
            results_file = os.path.join(run_dir, 'ga_result.json')
            with open(results_file, 'w') as f:
                json.dump(convert_ndarrays(ga_result), f)
            
            return JsonResponse({
                "status": "success", 
                "message": "Custom point integrated into GA population",
                "remaining_generations": generations - current_generation,
                "updated_population_size": len(current_population)
            })
        else:
            return JsonResponse({"status": "error", "message": "Run configuration not found"}, status=404)
        
    except Exception as e:
        print(f"Error integrating custom point to GA: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@api_view(["POST"])
def save_custom_point_to_dataset(request):
    """
    Permanently save a custom point to the main points.csv dataset.
    """
    try:
        data = json.loads(request.body)
        point = data.get('point')
        
        if not point:
            return JsonResponse({"status": "error", "message": "point data is required"}, status=400)
        
        # Validate point data
        required_fields = ['x', 'y', 'gpu', 'attn', 'sparse', 'conv']
        for field in required_fields:
            if field not in point:
                return JsonResponse({"status": "error", "message": f"Missing required field: {field}"}, status=400)
        
        # Add point type if not provided
        if 'type' not in point:
            point['type'] = 'custom'
        
        # Write point to main points.csv file
        import os
        import csv
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        points_csv_path = os.path.join(WORKSPACE, 'dse/results/points.csv')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(points_csv_path), exist_ok=True)
        
        # Append point to CSV file
        with open(points_csv_path, mode='a', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([
                point['x'],
                point['y'],
                point['gpu'],
                point['attn'],
                point['sparse'],
                point['conv'],
                point['type']
            ])
        
        print(f"Saved custom point to main dataset: {point}")
        return JsonResponse({
            "status": "success", 
            "message": "Custom point saved to main dataset",
            "point": point
        })
        
    except Exception as e:
        print(f"Error saving custom point to dataset: {e}")
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@api_view(["GET"])
def generate_comparative_report(request):
    """
    Generate a comprehensive comparative report for two optimization runs.
    """
    try:
        import os
        import csv
        import re
        import json
        import numpy as np
        import dcor
        from datetime import datetime
        from api.ChatBot.model import ChatBotModel
        
        print(f"=== generate_comparative_report START ===")
        print(f"Request GET params: {request.GET}")
        
        # Get run IDs from request parameters
        run_a_id = request.GET.get('run_a_id')
        run_b_id = request.GET.get('run_b_id')
        
        print(f"Run A ID: {run_a_id}")
        print(f"Run B ID: {run_b_id}")
        
        if not run_a_id or not run_b_id:
            print("ERROR: Missing run IDs")
            return JsonResponse({"status": "error", "message": "Both run_a_id and run_b_id are required"}, status=400)
        
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        results_dir = os.path.join(WORKSPACE, 'dse/results')
        
        print(f"Results directory: {results_dir}")
        print(f"Results directory exists: {os.path.exists(results_dir)}")
        
        # Load data for both runs
        def load_run_data(run_id):
            """Load data for a specific run"""
            print(f"Loading data for run_id: {run_id}")
            
            if run_id.startswith('loaded_run_'):
                # Load from backup file
                timestamp_str = run_id.replace('loaded_run_', '')
                backup_filename = f"points_backup_{timestamp_str}.csv"
                file_path = os.path.join(results_dir, backup_filename)
                print(f"Loading from backup file: {file_path}")
            elif run_id.startswith('previous_'):
                # Load from backup file (previous run format)
                backup_filename = run_id.replace('previous_', '')
                file_path = os.path.join(results_dir, backup_filename)
                print(f"Loading from previous run file: {file_path}")
            else:
                # Load from current run
                file_path = os.path.join(results_dir, "points.csv")
                print(f"Loading from current run file: {file_path}")
            
            print(f"File path: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            
            if not os.path.exists(file_path):
                print(f"ERROR: File does not exist: {file_path}")
                return None
            
            points = []
            try:
                with open(file_path, mode='r') as file:
                    csv_reader = csv.reader(file)
                    for row in csv_reader:
                        if len(row) >= 6:
                            points.append({
                                'x': float(row[0]),  # time
                                'y': float(row[1]),  # energy
                                'gpu': float(row[2]),
                                'attn': float(row[3]),
                                'sparse': float(row[4]),
                                'conv': float(row[5])
                            })
                print(f"Loaded {len(points)} points from {file_path}")
            except Exception as e:
                print(f"ERROR loading data from {file_path}: {e}")
                return None
            
            return points
        
        run_a_data = load_run_data(run_a_id)
        run_b_data = load_run_data(run_b_id)
        
        print(f"Run A data loaded: {run_a_data is not None}, points: {len(run_a_data) if run_a_data else 0}")
        print(f"Run B data loaded: {run_b_data is not None}, points: {len(run_b_data) if run_b_data else 0}")
        
        if not run_a_data or not run_b_data:
            print("ERROR: Could not load data for one or both runs")
            return JsonResponse({"status": "error", "message": "Could not load data for one or both runs"}, status=400)
        
        print("Data loaded successfully, proceeding with analysis...")
        
        # Get Pareto fronts
        def is_pareto_dominant(point1, point2):
            return (point1['x'] <= point2['x'] and point1['y'] <= point2['y']) and \
                   (point1['x'] < point2['x'] or point1['y'] < point2['y'])
        
        def get_pareto_front(points):
            pareto_front = []
            for i, point in enumerate(points):
                is_dominated = False
                for j, other_point in enumerate(points):
                    if i != j and is_pareto_dominant(other_point, point):
                        is_dominated = True
                        break
                if not is_dominated:
                    pareto_front.append(point)
            return pareto_front
        
        print("Calculating Pareto fronts...")
        run_a_pareto = get_pareto_front(run_a_data)
        run_b_pareto = get_pareto_front(run_b_data)
        print(f"Run A Pareto front size: {len(run_a_pareto)}")
        print(f"Run B Pareto front size: {len(run_b_pareto)}")
        
        # Calculate statistical metrics
        def calculate_statistics(points):
            if not points:
                return {}
            
            times = [p['x'] for p in points]
            energies = [p['y'] for p in points]
            
            return {
                'avg_time': np.mean(times),
                'avg_energy': np.mean(energies),
                'min_time': np.min(times),
                'min_energy': np.min(energies),
                'max_time': np.max(times),
                'max_energy': np.max(energies),
                'std_time': np.std(times),
                'std_energy': np.std(energies),
                'total_points': len(points)
            }
        
        print("Calculating statistics...")
        run_a_stats = calculate_statistics(run_a_data)
        run_b_stats = calculate_statistics(run_b_data)
        run_a_pareto_stats = calculate_statistics(run_a_pareto)
        run_b_pareto_stats = calculate_statistics(run_b_pareto)
        print("Statistics calculated successfully")
        
        # Calculate Pareto dominance comparison
        def count_dominated_points(pareto_a, pareto_b):
            """Count how many points in pareto_b are dominated by points in pareto_a"""
            dominated_count = 0
            for point_b in pareto_b:
                for point_a in pareto_a:
                    if is_pareto_dominant(point_a, point_b):
                        dominated_count += 1
                        break
            return dominated_count
        
        print("Calculating Pareto dominance...")
        a_dominates_b = count_dominated_points(run_a_pareto, run_b_pareto)
        b_dominates_a = count_dominated_points(run_b_pareto, run_a_pareto)
        print(f"A dominates B: {a_dominates_b}")
        print(f"B dominates A: {b_dominates_a}")
        
        # Get rule mining results for both runs
        print("Initializing ChatBot for rule mining...")
        chat_bot = ChatBotModel()
        
        def get_rule_mining_results(run_id):
            """Get rule mining results for a specific run"""
            print(f"Getting rule mining results for run_id: {run_id}")
            
            if run_id.startswith('loaded_run_'):
                timestamp_str = run_id.replace('loaded_run_', '')
                backup_filename = f"points_backup_{timestamp_str}.csv"
                file_path = os.path.join(results_dir, backup_filename)
            else:
                file_path = os.path.join(results_dir, "points.csv")
            
            print(f"Rule mining file path: {file_path}")
            
            point_selection_params = {"file_path": file_path}
            try:
                rule_mining_str = chat_bot.rule_mining(point_selection_params)
                print(f"Rule mining completed for {run_id}")
                
                # Parse rules
                rules = []
                rule_pattern = re.compile(r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), conf\(p->f\): ([0-9.eE+-]+), lift: \[([0-9.eE+-]+)\]")
                for match in rule_pattern.finditer(rule_mining_str):
                    rules.append({
                        "rule": match.group(1),
                        "conf_p_to_f": float(match.group(2)),
                        "conf_f_to_p": float(match.group(3)),
                        "lift": float(match.group(4)),
                    })
                print(f"Parsed {len(rules)} rules for {run_id}")
                return rules
            except Exception as e:
                print(f"ERROR in rule mining for {run_id}: {e}")
                return []
        
        print("Running rule mining...")
        run_a_rules = get_rule_mining_results(run_a_id)
        run_b_rules = get_rule_mining_results(run_b_id)
        print(f"Rule mining completed. Run A: {len(run_a_rules)} rules, Run B: {len(run_b_rules)} rules")
        
        # Calculate distance correlations for both runs
        def calculate_correlations(points):
            if not points:
                return {}
            
            xs = [p['x'] for p in points]
            ys = [p['y'] for p in points]
            gpus = [p['gpu'] for p in points]
            attns = [p['attn'] for p in points]
            sparses = [p['sparse'] for p in points]
            convs = [p['conv'] for p in points]
            
            return {
                "GPU_vs_Energy": float(dcor.distance_correlation(np.array(gpus), np.array(ys))),
                "Attention_vs_Energy": float(dcor.distance_correlation(np.array(attns), np.array(ys))),
                "Sparse_vs_Energy": float(dcor.distance_correlation(np.array(sparses), np.array(ys))),
                "Convolution_vs_Energy": float(dcor.distance_correlation(np.array(convs), np.array(ys))),
                "GPU_vs_Time": float(dcor.distance_correlation(np.array(gpus), np.array(xs))),
                "Attention_vs_Time": float(dcor.distance_correlation(np.array(attns), np.array(xs))),
                "Sparse_vs_Time": float(dcor.distance_correlation(np.array(sparses), np.array(xs))),
                "Convolution_vs_Time": float(dcor.distance_correlation(np.array(convs), np.array(xs))),
            }
        
        print("Calculating distance correlations...")
        run_a_correlations = calculate_correlations(run_a_data)
        run_b_correlations = calculate_correlations(run_b_data)
        print("Distance correlations calculated successfully")
        
        # Generate timestamp
        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H%M%S")
        
        print("Generating HTML report...")
        
        # Helper function for rule formatting
        def format_rule_natural_language(rule_str):
            """Convert rule string to natural language description"""
            rule = rule_str.replace("GPU=", "GPU count = ").replace("ATTN=", "Attention count = ").replace("SPARSE=", "Sparse count = ").replace("CONV=", "Convolution count = ")
            rule = rule.replace("AND", " and ").replace("OR", " or ")
            return rule
        
        # Generate comprehensive HTML report
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparative Analysis Report - {date_str}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }}
        h2 {{ color: #34495e; margin-top: 30px; margin-bottom: 15px; }}
        h3 {{ color: #2c3e50; margin-top: 25px; margin-bottom: 10px; }}
        .section {{ margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #3498db; }}
        .comparison-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
        .run-column {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .run-header {{ background: #3498db; color: white; padding: 15px; border-radius: 6px; margin-bottom: 15px; text-align: center; font-weight: 600; }}
        .run-a-header {{ background: #e74c3c; }}
        .run-b-header {{ background: #27ae60; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; font-weight: 600; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #e8f4fd; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
        .stat-card {{ background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 1.5em; font-weight: bold; color: #3498db; }}
        .stat-label {{ color: #7f8c8d; margin-top: 5px; font-size: 0.9em; }}
        .header-info {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .header-info p {{ margin: 5px 0; color: #2c3e50; }}
        .highlight {{ background-color: #fff3cd; padding: 2px 4px; border-radius: 3px; }}
        .better {{ background-color: #d4edda; color: #155724; padding: 2px 4px; border-radius: 3px; }}
        .worse {{ background-color: #f8d7da; color: #721c24; padding: 2px 4px; border-radius: 3px; }}
        .insight-box {{ background: #e8f4fd; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; border-radius: 4px; }}
        .recommendation {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 15px; margin: 15px 0; border-radius: 4px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparative Analysis Report</h1>
        
        <div class="header-info">
            <p><strong>Date:</strong> {date_str}</p>
            <p><strong>Time:</strong> {time_str}</p>
            <p><strong>Report Type:</strong> Comparative Analysis</p>
            <p><strong>Run A ID:</strong> {run_a_id}</p>
            <p><strong>Run B ID:</strong> {run_b_id}</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="insight-box">
                <h3>Key Findings:</h3>
                <ul>
                    <li><strong>Total Designs:</strong> Run A: {run_a_stats['total_points']}, Run B: {run_b_stats['total_points']}</li>
                    <li><strong>Pareto Front Size:</strong> Run A: {len(run_a_pareto)}, Run B: {len(run_b_pareto)}</li>
                    <li><strong>Dominance:</strong> Run A dominates {a_dominates_b} of Run B's Pareto points</li>
                    <li><strong>Dominance:</strong> Run B dominates {b_dominates_a} of Run A's Pareto points</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>Engineering Recommendation:</h3>
                <p>
                    {f"Run A appears to be {'better' if a_dominates_b > b_dominates_a else 'worse' if b_dominates_a > a_dominates_b else 'comparable'} to Run B based on Pareto dominance analysis. "}
                    {f"Run A has {'more' if len(run_a_pareto) > len(run_b_pareto) else 'fewer'} Pareto optimal designs, suggesting {'better' if len(run_a_pareto) > len(run_b_pareto) else 'worse'} exploration of the design space."}
                </p>
            </div>
        </div>

        <div class="section">
            <h2>Performance Metrics Comparison</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-header run-a-header">Run A Performance</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{run_a_stats['avg_time']:.2f}</div>
                            <div class="stat-label">Avg Time (ms)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_a_stats['avg_energy']:.2f}</div>
                            <div class="stat-label">Avg Energy (mJ)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_a_stats['min_time']:.2f}</div>
                            <div class="stat-label">Best Time (ms)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_a_stats['min_energy']:.2f}</div>
                            <div class="stat-label">Best Energy (mJ)</div>
                        </div>
                    </div>
                </div>
                <div class="run-column">
                    <div class="run-header run-b-header">Run B Performance</div>
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{run_b_stats['avg_time']:.2f}</div>
                            <div class="stat-label">Avg Time (ms)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_b_stats['avg_energy']:.2f}</div>
                            <div class="stat-label">Avg Energy (mJ)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_b_stats['min_time']:.2f}</div>
                            <div class="stat-label">Best Time (ms)</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_b_stats['min_energy']:.2f}</div>
                            <div class="stat-label">Best Energy (mJ)</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Pareto Front Analysis</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-header run-a-header">Run A Pareto Front</div>
                    <p><strong>Size:</strong> {len(run_a_pareto)} designs</p>
                    <table>
                        <thead>
                            <tr>
                                <th>GPU</th>
                                <th>Attention</th>
                                <th>Sparse</th>
                                <th>Convolution</th>
                                <th>Time (ms)</th>
                                <th>Energy (mJ)</th>
                            </tr>
                        </thead>
                        <tbody>"""

        # Add Run A Pareto points
        for point in run_a_pareto[:10]:  # Show first 10 points
            html_content += f"""
                            <tr>
                                <td>{int(point['gpu'])}</td>
                                <td>{int(point['attn'])}</td>
                                <td>{int(point['sparse'])}</td>
                                <td>{int(point['conv'])}</td>
                                <td>{point['x']:.2f}</td>
                                <td>{point['y']:.2f}</td>
                            </tr>"""

        html_content += f"""
                        </tbody>
                    </table>
                    {f'<p><em>Showing first 10 of {len(run_a_pareto)} Pareto optimal designs</em></p>' if len(run_a_pareto) > 10 else ''}
                </div>
                <div class="run-column">
                    <div class="run-header run-b-header">Run B Pareto Front</div>
                    <p><strong>Size:</strong> {len(run_b_pareto)} designs</p>
                    <table>
                        <thead>
                            <tr>
                                <th>GPU</th>
                                <th>Attention</th>
                                <th>Sparse</th>
                                <th>Convolution</th>
                                <th>Time (ms)</th>
                                <th>Energy (mJ)</th>
                            </tr>
                        </thead>
                        <tbody>"""

        # Add Run B Pareto points
        for point in run_b_pareto[:10]:  # Show first 10 points
            html_content += f"""
                            <tr>
                                <td>{int(point['gpu'])}</td>
                                <td>{int(point['attn'])}</td>
                                <td>{int(point['sparse'])}</td>
                                <td>{int(point['conv'])}</td>
                                <td>{point['x']:.2f}</td>
                                <td>{point['y']:.2f}</td>
                            </tr>"""

        html_content += f"""
                        </tbody>
                    </table>
                    {f'<p><em>Showing first 10 of {len(run_b_pareto)} Pareto optimal designs</em></p>' if len(run_b_pareto) > 10 else ''}
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Association Rule Mining Comparison</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-header run-a-header">Run A Rules ({len(run_a_rules)})</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Rule Description</th>
                                <th>Conf(F→P)</th>
                                <th>Lift</th>
                            </tr>
                        </thead>
                        <tbody>"""

        # Add Run A rules
        for rule in run_a_rules[:5]:  # Show first 5 rules
            formatted_rule = format_rule_natural_language(rule['rule'])
            html_content += f"""
                            <tr>
                                <td>{formatted_rule}</td>
                                <td>{rule['conf_f_to_p']:.2f}</td>
                                <td>{rule['lift']:.2f}</td>
                            </tr>"""

        html_content += f"""
                        </tbody>
                    </table>
                    {f'<p><em>Showing first 5 of {len(run_a_rules)} rules</em></p>' if len(run_a_rules) > 5 else ''}
                </div>
                <div class="run-column">
                    <div class="run-header run-b-header">Run B Rules ({len(run_b_rules)})</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Rule Description</th>
                                <th>Conf(F→P)</th>
                                <th>Lift</th>
                            </tr>
                        </thead>
                        <tbody>"""

        # Add Run B rules
        for rule in run_b_rules[:5]:  # Show first 5 rules
            formatted_rule = format_rule_natural_language(rule['rule'])
            html_content += f"""
                            <tr>
                                <td>{formatted_rule}</td>
                                <td>{rule['conf_f_to_p']:.2f}</td>
                                <td>{rule['lift']:.2f}</td>
                            </tr>"""

        html_content += f"""
                        </tbody>
                    </table>
                    {f'<p><em>Showing first 5 of {len(run_b_rules)} rules</em></p>' if len(run_b_rules) > 5 else ''}
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Distance Correlation Analysis</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-header run-a-header">Run A Correlations</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Chiplet Type</th>
                                <th>vs Energy</th>
                                <th>vs Time</th>
                            </tr>
                        </thead>
                        <tbody>"""

        chiplet_types = ['GPU', 'Attention', 'Sparse', 'Convolution']
        for chiplet in chiplet_types:
            html_content += f"""
                            <tr>
                                <td>{chiplet}</td>
                                <td>{run_a_correlations[f'{chiplet}_vs_Energy']:.3f}</td>
                                <td>{run_a_correlations[f'{chiplet}_vs_Time']:.3f}</td>
                            </tr>"""

        html_content += f"""
                        </tbody>
                    </table>
                </div>
                <div class="run-column">
                    <div class="run-header run-b-header">Run B Correlations</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Chiplet Type</th>
                                <th>vs Energy</th>
                                <th>vs Time</th>
                            </tr>
                        </thead>
                        <tbody>"""

        for chiplet in chiplet_types:
            html_content += f"""
                            <tr>
                                <td>{chiplet}</td>
                                <td>{run_b_correlations[f'{chiplet}_vs_Energy']:.3f}</td>
                                <td>{run_b_correlations[f'{chiplet}_vs_Time']:.3f}</td>
                            </tr>"""

        html_content += f"""
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Engineering Insights & Recommendations</h2>
            <div class="insight-box">
                <h3>Key Insights:</h3>
                <ul>
                    <li><strong>Design Space Coverage:</strong> {'Run A' if len(run_a_pareto) > len(run_b_pareto) else 'Run B'} explores more of the Pareto optimal design space</li>
                    <li><strong>Performance Stability:</strong> {'Run A' if run_a_stats['std_time'] < run_b_stats['std_time'] else 'Run B'} shows more consistent execution times</li>
                    <li><strong>Energy Efficiency:</strong> {'Run A' if run_a_stats['avg_energy'] < run_b_stats['avg_energy'] else 'Run B'} achieves better average energy efficiency</li>
                    <li><strong>Rule Quality:</strong> {'Run A' if len(run_a_rules) > len(run_b_rules) else 'Run B'} discovered more association rules</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>Recommendations:</h3>
                <ol>
                    <li><strong>Primary Choice:</strong> {'Run A' if a_dominates_b > b_dominates_a else 'Run B'} should be preferred for its superior Pareto dominance</li>
                    <li><strong>Design Strategy:</strong> Focus on configurations that appear in the Pareto front of the better performing run</li>
                    <li><strong>Future Optimizations:</strong> Use the parameter settings from the better run as a starting point</li>
                    <li><strong>Validation:</strong> Consider running additional optimizations with similar parameters to validate findings</li>
                </ol>
            </div>
        </div>
    </div>
</body>
</html>"""

        # Save report to file system
        reports_dir = os.path.join(results_dir, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        report_filename = f"comparative_report_{run_a_id}_vs_{run_b_id}_{time_str}.html"
        report_path = os.path.join(reports_dir, report_filename)
        
        print(f"Saving report to: {report_path}")
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            print("HTML report saved successfully")
        except Exception as e:
            print(f"ERROR saving HTML report: {e}")
            return JsonResponse({"status": "error", "message": f"Failed to save report: {str(e)}"}, status=500)
        
        # Save metadata
        metadata = {
            'run_a_id': run_a_id,
            'run_b_id': run_b_id,
            'timestamp': time_str,
            'date': date_str,
            'run_a_stats': run_a_stats,
            'run_b_stats': run_b_stats,
            'run_a_pareto_size': len(run_a_pareto),
            'run_b_pareto_size': len(run_b_pareto),
            'a_dominates_b': a_dominates_b,
            'b_dominates_a': b_dominates_a,
            'report_filename': report_filename
        }
        
        metadata_path = report_path.replace('.html', '_metadata.json')
        print(f"Saving metadata to: {metadata_path}")
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print("Metadata saved successfully")
        except Exception as e:
            print(f"ERROR saving metadata: {e}")
            # Don't fail the entire request if metadata saving fails
        
        # Create web link
        web_link = f"/api/Evaluator/cascade/chiplet_model/dse/results/reports/{report_filename}"
        
        print(f"Web link: {web_link}")
        print("=== generate_comparative_report SUCCESS ===")
        
        return JsonResponse({
            "status": "success",
            "message": "Comparative report generated successfully",
            "web_link": web_link,
            "download_link": web_link,
            "report_filename": report_filename,
            "metadata": metadata
        })
        
    except Exception as e:
        print(f"=== generate_comparative_report ERROR ===")
        print(f"Error in generate_comparative_report: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)