"""
Optimization-related API endpoints.
Restores live chart updates via background threads + polling pattern from views.py [1].
"""
import os
import sys
import json
import csv
import time
import threading
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path

from django.http import JsonResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response

from api.config.settings import EVALUATOR_BASE_PATHS
from api.data.loaders import PointsLoader
from api.Evaluator.gaCascade import runGACascade, runSingleCascade
from api.Evaluator.generator import generate_weighted_trace
from api.services.run_storage import RunStorageService
from api.models import OptimizationRun


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def convert_ndarrays(obj):
    """Recursively convert numpy types for JSON serialization."""
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


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_directory(run_id: str) -> str:
    """Create and return a run directory path."""
    WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
    run_dir = os.path.join(WORKSPACE, f'dse/results/myrun_{run_id}')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'pointContext'), exist_ok=True)
    return run_dir


def get_workspace_paths():
    """Return commonly used workspace paths."""
    WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
    RESULTS_DIR = os.path.join(WORKSPACE, 'dse/results')
    return WORKSPACE, RESULTS_DIR


# ─────────────────────────────────────────────
# KEY FIX: Polling endpoint for live chart updates
# This is what the frontend calls repeatedly to get new points [1]
# ─────────────────────────────────────────────

@api_view(["GET"])
def get_chart_data(request):
    """
    Polling endpoint for live chart updates during optimization.

    The frontend calls this repeatedly after receiving 'status: started'.
    It reads the live points.csv which the background GA thread writes to row-by-row.
    This is the core mechanism that enables live chart updates [1].
    """
    model = request.GET.get("model", "CASCADE").upper()
    run_id = request.GET.get("run_id")
    file_path = request.GET.get("file_path")
    requested_algorithm = request.GET.get("algorithm")
    
    from api.config.objectives import DEFAULT_OBJECTIVES, to_fields

    # Read the friendly names from the query string
    objectives_raw = request.GET.get("objectives") or ""
    objectives = [o.strip() for o in objectives_raw.split(",") if o.strip()]
    if not objectives:
        objectives = DEFAULT_OBJECTIVES.get(model.lower(), [])

    # Internal field names if you need them downstream
    objective_fields = to_fields(objectives)


    print(f"[get_chart_data] model={model}, run_id={run_id}, file_path={file_path}, requested_algorithm={requested_algorithm}, objectives={objectives}")

    # Determine algorithm display name from DB if possible
    algorithm_to_use = requested_algorithm or "Genetic Algorithm"
    if run_id:
        try:
            optimization_run = OptimizationRun.objects.get(run_id=run_id)
            algorithm_to_use = optimization_run.get_algorithm_display()
        except OptimizationRun.DoesNotExist:
            pass

    _, RESULTS_DIR = get_workspace_paths()

    if model == "CASCADE":
        # Use provided file_path or fall back to the live points.csv [1]
        if file_path and os.path.exists(file_path):
            points_csv_path = file_path
        else:
            points_csv_path = os.path.join(RESULTS_DIR, "points.csv")

        if not os.path.exists(points_csv_path):
            return Response({"data": []})

        loader = PointsLoader('cascade', run_id, num_objs=len(objectives))
        # print("A")
        points = loader.load_points_as_dicts(points_csv_path)

        trace_name = request.GET.get("trace", "gpt-j-65536-weighted")
        for point in points:
            if not point.get('algorithm'):
                point['algorithm'] = algorithm_to_use
            point['trace'] = trace_name
            point['model'] = 'CASCADE'

        # print("C", points[:2])  # Debug: print first 2 points loaded

        return Response({"data": points})

    elif model == "PISTIL":
        # For PISTIL, the run_id determines the directory [1]
        if file_path and os.path.exists(file_path):
            points_csv_path = file_path
        elif run_id:
            pistil_root = Path(sys.path[0]) / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean"
            output_dir = pistil_root / "dse" / "results" / run_id
            # print(f"[PISTIL] Looking for points.csv in {output_dir}")
            points_csv_path = str(output_dir / "points.csv")
        else:
            return Response({"data": []})

        if not os.path.exists(points_csv_path):
            return Response({"data": []})

        loader = PointsLoader('pistil', run_id, num_objs=len(objectives))
        points = loader.load_points_as_dicts(points_csv_path)

        trace_name = request.GET.get("trace", "pistil-default")
        for point in points:
            if not point.get('algorithm'):
                point['algorithm'] = algorithm_to_use
            point['trace'] = trace_name
            point['model'] = 'PISTIL'

        return Response({"data": points})

    return Response({"data": []})


# ─────────────────────────────────────────────
# Main optimization endpoint
# ─────────────────────────────────────────────

@api_view(["POST"])
def run_optimization(request):
    """
    Start an optimization run and return immediately with 'status: started'.

    KEY PATTERN (restored from views.py [1]):
      1. Clear points.csv
      2. Create DB record
      3. Start background thread that writes to points.csv row-by-row
      4. Return {status: 'started', run_id: ...} IMMEDIATELY
      5. Frontend polls get_chart_data() with run_id to receive live updates
    """
    try:
        data = json.loads(request.body)

        algorithm_raw = data.get('algorithm', 'Genetic Algorithm')
        model         = data.get('model', 'CASCADE')
        objectives    = data.get('objectives', [])
        traces        = data.get('traces', [])
        objectives    = data.get('objectives', [])
        population    = int(data.get('population_size', data.get('population', 50)))
        generations   = int(data.get('generations', 100))

        if not objectives:
            return JsonResponse({"status": "error", "message": "objectives are required"}, status=400)
        if not traces:
            return JsonResponse({"status": "error", "message": "At least one trace is required"}, status=400)

        # Resolve trace name
        if len(traces) == 1:
            trace_name = traces[0].get('name', traces[0]) if isinstance(traces[0], dict) else traces[0]
        else:
            trace_names = [t.get('name', t) if isinstance(t, dict) else t for t in traces]
            weights     = [t.get('weight', 1.0) if isinstance(t, dict) else 1.0 for t in traces]
            trace_name  = generate_weighted_trace(trace_names, weights, label='main')

        _, RESULTS_DIR = get_workspace_paths()

        # ── GENETIC ALGORITHM ──────────────────────────────────────────────────
        if algorithm_raw in ('Genetic Algorithm', 'GA'):
            if population <= 0:
                return JsonResponse({"status": "error",
                                    "message": f"population_size must be > 0, got {population}"}, status=400)
            if generations <= 0:
                return JsonResponse({"status": "error",
                                    "message": f"generations must be > 0, got {generations}"}, status=400)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # PISTIL GA handling
            if model.upper() == 'PISTIL':
                pistil_model = data.get('pistil_model', 'llama3-8b')
                run_id = f"pistil_run_{timestamp}"
                pistil_root = Path(sys.path[0]) / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean"
                output_dir = pistil_root / "dse" / "results" / run_id
                os.makedirs(output_dir, exist_ok=True)
                points_csv_path = str(output_dir / "points.csv")
                
                # Write PISTIL CSV header
                pistil_header = (
                    "num_cus,num_tmacs,mem_buf_cap,net_buf_cap,mem_banks_per_group,"
                    "mem_ranks,mem_frac_bank_cap,batch_size,kv_cache,"
                    "latency_per_token_ms,energy_per_inference_mJ,energy_per_token_mJ,"
                    "average_power_W,system_power_W,system_cost,"
                    "avg_comp_util,avg_mem_util,prefill_tokens_per_sec,"
                    "latency_ms,energy_mJ"
                )
                with open(points_csv_path, 'w') as f:
                    f.write(pistil_header + "\n")
                print(f"[GA PISTIL] Created output dir: {output_dir}")
                
                # Create DB record
                optimization_run = RunStorageService.create_optimization_run(
                    run_id=run_id,
                    algorithm='GA',
                    model='PISTIL',
                    population_size=population,
                    generations=generations,
                    objectives=objectives,
                    trace_name=pistil_model,
                    trace_sets={'main': pistil_model},
                    name="Optimization Run - PISTIL Genetic Algorithm",
                    description=f"PISTIL GA pop={population} gen={generations} model={pistil_model}"
                )
                
                def run_pistil_ga_job():
                    execution_time_seconds = 0
                    design_points = []
                    try:
                        start = time.time()
                        print(f"[GA PISTIL Background] Starting: pop={population}, gen={generations}, model={pistil_model}")
                        
                        # Import and run PISTIL GA
                        from api.Evaluator.gaPistil import runGAPistil
                        ga_result, design_points = runGAPistil(
                            pop_size=population,
                            n_gen=generations,
                            model_name=pistil_model,
                            objectives=objectives,
                            output_dir=str(output_dir)
                        )
                        
                        execution_time_seconds = time.time() - start
                        print(f"[GA PISTIL Background] Done. {len(design_points)} pts in {execution_time_seconds:.1f}s")
                        
                        # Store design points
                        if design_points:
                            RunStorageService.store_design_points(optimization_run, design_points, str(output_dir))
                        
                        # Run analytics
                        analytics_results = _run_analytics(points_csv_path, 'pistil', objectives=objectives)
                        
                        RunStorageService.complete_run(
                            optimization_run,
                            execution_time_seconds=execution_time_seconds,
                            analytics_results=analytics_results
                        )
                        print(f"[GA PISTIL Background] Run {optimization_run.run_id} complete.")
                        
                    except Exception as e:
                        print(f"[GA PISTIL Background] Error: {e}")
                        traceback.print_exc()
                        try:
                            RunStorageService.complete_run(
                                optimization_run,
                                execution_time_seconds=execution_time_seconds,
                                analytics_results={'rule_mining': '', 'distance_correlation': ''}
                            )
                        except Exception:
                            pass
                
                # START THREAD
                threading.Thread(target=run_pistil_ga_job, daemon=True).start()
                
                return JsonResponse({
                    'status':         'started',
                    'data':           [],
                    'plot_data':      [],
                    'run_id':         run_id,
                    'pistil_run_id':  run_id,
                    'run_directory':  run_id,
                    'db_run_id':      optimization_run.run_id,
                    'metadata': {
                        'model':           'PISTIL',
                        'algorithm':       'Genetic Algorithm',
                        'objectives':      objectives,
                        'population_size': population,
                        'generations':     generations,
                        'pistil_model':    pistil_model,
                    }
                })

            # CASCADE GA handling
            run_id  = generate_run_id()
            run_dir = create_run_directory(run_id)

            # CRITICAL: Clear points.csv so frontend polls from zero [1]
            current_points_file = os.path.join(RESULTS_DIR, "points.csv")
            with open(current_points_file, 'w') as f:
                f.write("")
            print(f"[GA] Cleared {current_points_file} for new run")

            # Create DB record BEFORE thread starts
            optimization_run = RunStorageService.create_optimization_run(
                run_id=run_id,
                algorithm='GA',
                model=model,
                population_size=population,
                generations=generations,
                objectives=objectives,
                trace_name=trace_name,
                trace_sets={'main': traces},
                name="Optimization Run - Genetic Algorithm",
                description=f"GA pop={population} gen={generations}"
            )

            def run_ga_job():
                execution_time_seconds = 0
                try:
                    start = time.time()
                    print(f"[GA Background] Starting: pop={population}, gen={generations}, trace={trace_name}")

                    # runGACascade writes each evaluated point to points.csv live [1]
                    ga_result = runGACascade(
                        pop_size=population,
                        n_gen=generations,
                        trace=trace_name,
                        objectives=objectives,
                        output_dir=RESULTS_DIR
                    )
                    ga_result = convert_ndarrays(ga_result)
                    execution_time_seconds = time.time() - start
                    print(f"[GA Background] Done. {len(ga_result)} pts in {execution_time_seconds:.1f}s")

                    # Timestamped backup after completion [1]
                    import shutil
                    if os.path.exists(current_points_file) and \
                       os.path.getsize(current_points_file) > 0:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        backup = os.path.join(RESULTS_DIR, f"points_backup_{ts}.csv")
                        shutil.copy2(current_points_file, backup)
                        print(f"[GA Background] Backup → {backup}")

                    # Persist design points to DB
                    design_points = []
                    for pt in ga_result:
                        if isinstance(pt, dict):
                            design_points.append({
                                'execution_time_ms': pt.get('execution_time_ms', pt.get('x', 0)),
                                'energy_mj':         pt.get('energy_mj', pt.get('y', 0)),
                                'chiplets':          pt.get('chiplets', {}),
                                'additional_metrics': pt.get('additional_metrics', {}),
                                'context_file_path': pt.get('context_file_path', '')
                            })

                    if design_points:
                        RunStorageService.store_design_points(optimization_run, design_points, run_dir)

                    # Run analytics
                    analytics_results = _run_analytics(current_points_file, 'cascade', objectives=objectives)

                    RunStorageService.complete_run(
                        optimization_run,
                        execution_time_seconds=execution_time_seconds,
                        analytics_results=analytics_results
                    )
                    print(f"[GA Background] Run {optimization_run.run_id} complete.")

                except Exception as e:
                    print(f"[GA Background] Error: {e}")
                    traceback.print_exc()
                    try:
                        RunStorageService.complete_run(
                            optimization_run,
                            execution_time_seconds=execution_time_seconds,
                            analytics_results={'rule_mining': '', 'distance_correlation': ''}
                        )
                    except Exception:
                        pass

            # START THREAD — return immediately so frontend can start polling [1]
            threading.Thread(target=run_ga_job, daemon=True).start()

            return JsonResponse({
                'status':       'started',   # ← tells frontend to start polling
                'data':         [],           # ← empty; frontend polls get_chart_data
                'plot_data':    [],
                'run_id':       optimization_run.run_id,
                'run_directory': run_id,
                'metadata': {
                    'model':           model,
                    'algorithm':       'Genetic Algorithm',
                    'objectives':      objectives,
                    'population_size': population,
                    'generations':     generations,
                    'trace':           trace_name,
                }
            })

        # ── FULL FACTORIAL ─────────────────────────────────────────────────────
        elif algorithm_raw == 'Full-Factorial':
            timeout_seconds = int(data.get('timeout_seconds', 300))
            selected_types  = data.get('selected_types', ['GPU', 'Attention', 'Sparse', 'Convolution'])
            n               = int(data.get('num_slots', 12))
            mode            = data.get('mode', 'online')
            m               = len(selected_types)

            # Pre-flight estimate request
            if data.get('estimate_only'):
                return _estimate_full_factorial(selected_types, n, m, trace_name)

            run_id  = generate_run_id()
            run_dir = create_run_directory(run_id)

            # Clear points.csv for live polling [1]
            current_points_file = os.path.join(RESULTS_DIR, "points.csv")
            with open(current_points_file, 'w') as f:
                f.write("")

            from math import comb
            design_space_size = comb(n + m - 1, m - 1)

            optimization_run = RunStorageService.create_optimization_run(
                run_id=run_id,
                algorithm='FF',
                model=model,
                population_size=0,
                generations=0,
                objectives=objectives,
                trace_name=trace_name,
                trace_sets={'main': traces},
                name="Optimization Run - Full-Factorial",
                description=f"FF m={m}, n={n}, types={selected_types}"
            )

            def compositions(total, parts):
                if parts == 1:
                    yield [total]; return
                for i in range(total + 1):
                    for rest in compositions(total - i, parts - 1):
                        yield [i] + rest

            def run_full_factorial_job():
                execution_time_seconds = 0
                design_points = []
                try:
                    start = time.time()
                    for counts in compositions(n, m):
                        if time.time() - start > timeout_seconds:
                            print(f"[FF Background] Timeout after {len(design_points)} evals")
                            break
                        chiplet_counts = {t: counts[i] for i, t in enumerate(selected_types)}
                        full_map = {"GPU": 0, "Attention": 0, "Sparse": 0, "Convolution": 0}
                        full_map.update({k: int(v) for k, v in chiplet_counts.items() if k in full_map})

                        # save_to_csv=True writes the point live to points.csv [1]
                        exec_ms, energy_mj = runSingleCascade(
                            chiplets=full_map,
                            trace=trace_name,
                            objectives=objectives,
                            save_to_csv=True,
                            source='Full-Factorial'
                        )
                        design_points.append({
                            'execution_time_ms': exec_ms,
                            'energy_mj':         energy_mj,
                            'chiplets':          full_map,
                            'additional_metrics': {},
                            'context_file_path': ''
                        })

                    execution_time_seconds = time.time() - start
                    print(f"[FF Background] {len(design_points)} pts in {execution_time_seconds:.1f}s")

                    if design_points:
                        RunStorageService.store_design_points(optimization_run, design_points, run_dir)

                except Exception as e:
                    print(f"[FF Background] Error: {e}")
                    traceback.print_exc()
                finally:
                    try:
                        RunStorageService.complete_run(
                            optimization_run,
                            execution_time_seconds=execution_time_seconds,
                            analytics_results={'rule_mining': '', 'distance_correlation': ''}
                        )
                    except Exception as e:
                        print(f"[FF Background] Error completing run: {e}")

            if mode == 'online':
                threading.Thread(target=run_full_factorial_job, daemon=True).start()
                return JsonResponse({
                    'status':       'started',
                    'data':         [],
                    'plot_data':    [],
                    'run_id':       optimization_run.run_id,
                    'run_directory': run_id,
                    'metadata': {
                        'model':              model,
                        'algorithm':          'Full-Factorial',
                        'objectives':         objectives,
                        'selected_types':     selected_types,
                        'num_slots':          n,
                        'design_space_size':  int(design_space_size),
                        'timeout_seconds':    timeout_seconds,
                        'mode':               mode,
                    }
                })
            else:
                # Offline / synchronous mode
                run_full_factorial_job()
                loader = PointsLoader('cascade', num_objs = len(objectives))
                points = loader.load_points_as_dicts(current_points_file)
                return JsonResponse({
                    'status':    'success',
                    'data':      points,
                    'plot_data': points,
                    'run_id':    optimization_run.run_id,
                })

        # ── DEEP RL ────────────────────────────────────────────────────────────
        elif algorithm_raw == 'Deep RL':
            episodes        = int(data.get('episodes', 100))
            mini_batch_size = int(data.get('mini_batch_size', 32))
            pistil_model    = data.get('pistil_model', 'llama3-8b')

            if episodes < 1:
                return JsonResponse({"status": "error",
                                     "message": f"episodes must be >= 1, got {episodes}"}, status=400)
            if mini_batch_size < 1:
                return JsonResponse({"status": "error",
                                     "message": f"mini_batch_size must be >= 1, got {mini_batch_size}"}, status=400)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if model.upper() == 'PISTIL':
                run_id      = f"pistil_run_{timestamp}"
                pistil_root = Path(sys.path[0]) / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean"
                output_dir  = pistil_root / "dse" / "results" / run_id
                os.makedirs(output_dir, exist_ok=True)
                points_csv_path = str(output_dir / "points.csv")

                # Write PISTIL CSV header so the live polling loader can parse it [1]
                pistil_header = (
                    "num_cus,num_tmacs,mem_buf_cap,net_buf_cap,mem_banks_per_group,"
                    "mem_ranks,mem_frac_bank_cap,batch_size,kv_cache,"
                    "latency_per_token_ms,energy_per_inference_mJ,energy_per_token_mJ,"
                    "average_power_W,system_power_W,system_cost,"
                    "avg_comp_util,avg_mem_util,prefill_tokens_per_sec,"
                    "latency_ms,energy_mJ"
                )
                with open(points_csv_path, 'w') as f:
                    f.write(pistil_header + "\n")
                print(f"[Deep RL] PISTIL output dir: {output_dir}")

            else:
                # CASCADE Deep RL
                run_id = f"deep_rl_run_{timestamp}"
                _, RESULTS_DIR = get_workspace_paths()
                points_csv_path = os.path.join(RESULTS_DIR, "points.csv")

                # Clear CASCADE points.csv for live polling [1]
                with open(points_csv_path, 'w') as f:
                    f.write("")
                print(f"[Deep RL] CASCADE output dir: {RESULTS_DIR}")

            run_dir = create_run_directory(run_id)

            optimization_run = RunStorageService.create_optimization_run(
                algorithm='DRL',
                model=model,
                population_size=0,
                generations=0,
                objectives=objectives,
                trace_name=pistil_model,
                trace_sets={'main': pistil_model},
                name=f"Deep RL Run ({model})",
                description=f"Deep RL episodes={episodes}, mini_batch_size={mini_batch_size}"
            )
            print(f"[Deep RL] Created run {optimization_run.run_id}")

            def run_deep_rl_job():
                """
                Background thread for Deep RL.
                For CASCADE: writes each point to points.csv live so frontend can poll [1].
                For PISTIL:  runPPOPistil writes to points_csv_path directly.
                """
                execution_time_seconds = 0
                design_points = []
                try:
                    start = time.time()
                    print(f"[Deep RL Background] Starting: model={model}, "
                          f"episodes={episodes}, mini_batch_size={mini_batch_size}")

                    if model.upper() == 'PISTIL':
                        from api.Evaluator.rlPistil import runPPOPistil
                        _, _, design_points = runPPOPistil(
                            num_epochs=episodes,
                            mini_batch_size=mini_batch_size,
                            model_name=pistil_model or "llama3-8b",
                            objectives=objectives,
                            output_dir=points_csv_path   # rlPistil writes live to this file
                        )

                    else:
                        # CASCADE: run PPO and write each point live [1]
                        from api.Evaluator.rlCascade import runPPOCascade
                        _, _, design_points = runPPOCascade(
                            num_epochs=episodes,
                            mini_batch_size=mini_batch_size,
                            trace=trace_name or "gpt-j-65536-weighted",
                            objectives=objectives
                        )
                        # Write each design point to CSV immediately so polling sees it [1]
                        for dp in design_points:
                            c = dp['chiplets']
                            with open(points_csv_path, 'a') as f:
                                f.write(
                                    f"{dp['execution_time_ms']},{dp['energy_mj']},"
                                    f"{c['GPU']},{c['Attention']},"
                                    f"{c['Sparse']},{c['Convolution']}\n"
                                )

                    execution_time_seconds = time.time() - start
                    print(f"[Deep RL Background] Completed {len(design_points)} "
                          f"points in {execution_time_seconds:.1f}s")

                    if design_points:
                        RunStorageService.store_design_points(
                            optimization_run, design_points, run_dir
                        )

                    analytics_results = _run_analytics(points_csv_path, model.lower(), objectives=objectives)

                except Exception as e:
                    print(f"[Deep RL Background] Error: {e}")
                    traceback.print_exc()
                    analytics_results = {'rule_mining': '', 'distance_correlation': ''}

                finally:
                    try:
                        RunStorageService.complete_run(
                            optimization_run,
                            execution_time_seconds=execution_time_seconds,
                            analytics_results=analytics_results
                        )
                        print(f"[Deep RL Background] Run {optimization_run.run_id} complete.")
                    except Exception as e:
                        print(f"[Deep RL Background] Error completing run: {e}")
                        traceback.print_exc()

            # START THREAD — return immediately so frontend can start polling [1]
            threading.Thread(target=run_deep_rl_job, daemon=True).start()

            response_data = {
                'status':        'started',
                'data':          [],
                'plot_data':     [],
                'run_id':        run_id,           # ← always the pistil_run_YYYYMMDD_HHMMSS format
                'db_run_id':     optimization_run.run_id,  # keep DB id separately if needed
                'deep_rl_run_id': run_id,          # keep for backward compat
                'metadata': {
                    'model':           model,
                    'algorithm':       'Deep RL',
                    'objectives':      objectives,
                    'episodes':        episodes,
                    'mini_batch_size': mini_batch_size,
                    'trace':           trace_name,
                    'pistil_model':    pistil_model if model.upper() == 'PISTIL' else None,
                }
            }

            return JsonResponse(response_data)

        # ── UNKNOWN ALGORITHM ──────────────────────────────────────────────────
        else:
            return JsonResponse({
                "status":  "error",
                "message": f"Unknown algorithm: '{algorithm_raw}'. "
                           f"Expected one of: 'Genetic Algorithm', 'Full-Factorial', 'Deep RL'."
            }, status=400)

    except json.JSONDecodeError as e:
        return JsonResponse({"status": "error", "message": f"Invalid JSON: {e}"}, status=400)
    except Exception as e:
        print(f"[run_optimization] Unhandled error: {e}")
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


# ─────────────────────────────────────────────
# Additional optimization endpoints
# ─────────────────────────────────────────────

@api_view(["POST"])
def restart_run(request):
    """
    Restart a GA run seeded from a previous backup file.
    Restores the background-thread + immediate-response pattern [1].
    """
    try:
        data = json.loads(request.body)
        backup_filename = data.get('backup_filename')
        generations     = int(data.get('generations', 10))
        traces          = data.get('traces', [])

        if not backup_filename:
            return JsonResponse(
                {"status": "error", "message": "backup_filename is required"}, status=400
            )

        _, RESULTS_DIR = get_workspace_paths()
        source_file = os.path.join(RESULTS_DIR, backup_filename)

        if not os.path.exists(source_file):
            return JsonResponse(
                {"status": "error",
                 "message": f"Backup file not found: {backup_filename}"}, status=404
            )

        run_id  = f"restarted_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(RESULTS_DIR, run_id)
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, 'pointContext'), exist_ok=True)

        # Keep original points for merging later
        original_points_file = os.path.join(run_dir, 'original_points.csv')
        with open(source_file, 'r') as src, open(original_points_file, 'w') as dst:
            for line in src:
                dst.write(line if line.endswith('\n') else line + '\n')

        # Read initial points to return immediately to frontend
        initial_points_data = []
        with open(original_points_file, 'r') as f:
            for row in csv.reader(f):
                if len(row) >= 6:
                    initial_points_data.append({
                        'x':         float(row[0]),
                        'y':         float(row[1]),
                        'gpu':       float(row[2]),
                        'attn':      float(row[3]),
                        'sparse':    float(row[4]),
                        'conv':      float(row[5]),
                        'algorithm': 'Genetic Algorithm',
                        'trace':     'gpt-j-65536-weighted'
                    })

        def run_seeded_ga():
            """
            Background thread: seeds GA from previous points and writes
            new points live to run_dir/points.csv for polling [1].
            """
            try:
                trace_name = 'gpt-j-65536-weighted'
                if isinstance(traces, list) and traces and \
                   isinstance(traces[0], dict) and traces[0].get('name'):
                    trace_name = traces[0]['name']

                # Build initial population from previous decisions [1]
                decisions = []
                with open(source_file, 'r') as f:
                    for row in csv.reader(f):
                        if len(row) >= 6:
                            gpu, attn, sparse, conv = (
                                int(float(row[2])), int(float(row[3])),
                                int(float(row[4])), int(float(row[5]))
                            )
                            decisions.append(([gpu, attn, sparse, conv] * 3)[:12])

                initial_sampling = np.array(decisions, dtype=int) if decisions else None
                pop_size = len(decisions) if decisions else 50

                # GA writes live to run_dir/points.csv [1]
                runGACascade(
                    pop_size=pop_size,
                    n_gen=generations,
                    trace=trace_name,
                    objectives=['Runtime', 'Energy'],
                    initial_population=initial_sampling,
                    output_dir=run_dir
                )

                # Merge original + new points
                new_points_file = os.path.join(run_dir, 'points.csv')
                merged_file = os.path.join(
                    run_dir,
                    f'points_merged_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
                )
                with open(merged_file, 'w') as merged:
                    for src_path in [original_points_file, new_points_file]:
                        if os.path.exists(src_path):
                            with open(src_path, 'r') as src:
                                for line in src:
                                    merged.write(line)

                print(f"[restart_run] Merged file: {merged_file}")

            except Exception as e:
                print(f"[restart_run] Background error: {e}")
                traceback.print_exc()

        threading.Thread(target=run_seeded_ga, daemon=True).start()

        return JsonResponse({
            "status":         "started",    # Frontend starts polling immediately [1]
            "run_id":         run_id,
            "initial_points": initial_points_data,
            "message":        f"Restarting run from {backup_filename} "
                              f"with {len(initial_points_data)} seed points"
        })

    except Exception as e:
        print(f"[restart_run] Error: {e}")
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


@api_view(["GET"])
def check_run_status(request):
    """
    Check the status of a running optimization.
    Returns current point count so frontend can show progress [1].
    """
    run_id = request.GET.get("run_id")
    model  = request.GET.get("model", "CASCADE").upper()

    if not run_id:
        return Response({"status": "error", "message": "run_id required"}, status=400)

    _, RESULTS_DIR = get_workspace_paths()

    # Determine which points file to check
    if model == 'PISTIL' and run_id.startswith('pistil_run_'):
        pistil_root     = Path(sys.path[0]) / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean"
        points_csv_path = str(pistil_root / "dse" / "results" / run_id / "points.csv")
    elif run_id.startswith('restarted_run_'):
        points_csv_path = os.path.join(RESULTS_DIR, run_id, 'points.csv')
    else:
        points_csv_path = os.path.join(RESULTS_DIR, "points.csv")

    # Count current points for progress reporting
    current_count = 0
    if os.path.exists(points_csv_path):
        try:
            with open(points_csv_path, 'r') as f:
                current_count = sum(
                    1 for row in csv.reader(f)
                    if row and not row[0].startswith('#')
                    and not row[0].startswith('num_cus')   # skip PISTIL header
                )
        except Exception:
            pass

    # Check DB for completion status
    is_complete = False
    try:
        optimization_run = OptimizationRun.objects.get(run_id=run_id)
        is_complete = optimization_run.status == 'completed'
    except OptimizationRun.DoesNotExist:
        pass

    return Response({
        "run_id":        run_id,
        "current_count": current_count,
        "is_complete":   is_complete,
        "points_file":   points_csv_path,
        "file_exists":   os.path.exists(points_csv_path),
    })


@api_view(["GET"])
def estimate_full_factorial(request):
    """
    Estimate the size of a full factorial design space without running it.
    """
    from math import comb
    selected_types = request.GET.getlist("selected_types",
                                         ['GPU', 'Attention', 'Sparse', 'Convolution'])
    n = int(request.GET.get("num_slots", 12))
    m = len(selected_types)

    design_space_size = comb(n + m - 1, m - 1)

    return Response({
        "design_space_size": design_space_size,
        "selected_types":    selected_types,
        "num_slots":         n,
        "estimated_time_s":  design_space_size * 0.1,   # rough estimate
        "recommendation":    "online" if design_space_size < 500 else "offline"
    })


# ─────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────

def _run_analytics(points_csv_path: str, evaluator: str, objectives=None) -> dict:
    """
    Run rule mining and distance correlation after optimization completes.
    `objectives` is a list of *friendly* names (e.g. ['Energy', 'Runtime']).
    """
    from api.chatbot.bot import ChatBot
    from api.config.objectives import to_fields, DEFAULT_OBJECTIVES
    from api.config.evaluators import get_evaluator_config
    import csv

    analytics_results = {'rule_mining': '', 'distance_correlation': ''}
    try:
        chat_bot = ChatBot(evaluator=evaluator)
        if objectives:
            chat_bot.set_objectives(objectives)

        cfg = get_evaluator_config(evaluator)

        # Resolve which objective columns/decisions to feed to dcorr
        friendly = objectives or DEFAULT_OBJECTIVES.get(evaluator.lower(), [])
        wanted_fields = to_fields(friendly)
        all_obj_cols = cfg.objective_columns
        obj_indices = [all_obj_cols.index(f) for f in wanted_fields if f in all_obj_cols]
        if not obj_indices:
            obj_indices = list(range(len(all_obj_cols)))
            friendly = all_obj_cols  # fall back

        # Load CSV → numeric arrays projected onto selected objectives
        objective_vals, design_vals = [], []
        n_obj = len(all_obj_cols)
        with open(points_csv_path) as f:
            for row in csv.reader(f):
                if not row or row[0].startswith(('num_cus', 'energy', 'exe_time')):
                    continue  # skip headers
                try:
                    nums = [float(x) for x in row]
                except ValueError:
                    continue
                if len(nums) < n_obj + len(cfg.decision_columns):
                    continue
                objective_vals.append([nums[i] for i in obj_indices])
                design_vals.append(nums[n_obj : n_obj + len(cfg.decision_columns)])

        if objective_vals:
            analytics_results['distance_correlation'] = chat_bot.get_distance_correlations(
                objective_vals, design_vals,
                metric_names=friendly,                  # human-readable
                decision_names=cfg.decision_columns,
            )

        # Rule mining (already centralized via ChatBot.rule_mining → RuleMiningAgent)
        analytics_results['rule_mining'] = chat_bot.rule_mining()

    except Exception as e:
        import traceback; traceback.print_exc()

    return analytics_results


def _estimate_full_factorial(selected_types, n, m, trace_name):
    """Return a quick estimate JsonResponse for Full-Factorial design space."""
    from math import comb
    design_space_size = comb(n + m - 1, m - 1)
    return JsonResponse({
        'status':             'estimate',
        'design_space_size':  int(design_space_size),
        'selected_types':     selected_types,
        'num_slots':          n,
        'trace':              trace_name,
        'recommendation':     'online' if design_space_size < 500 else 'offline',
        'estimated_time_s':   design_space_size * 0.1,
    })

@api_view(["GET"])
def compute_sum(request):
    """Legacy utility endpoint [1]."""
    try:
        num1 = int(request.GET.get("num1", 0))
        num2 = int(request.GET.get("num2", 0))
        return Response({"result": num1 + num2})
    except (ValueError, TypeError):
        return Response({"error": "Invalid numbers provided."}, status=400)


@api_view(["GET"])
def update_data(request):
    """
    Legacy data update endpoint.
    Returns current points data from CSV [1].
    """
    try:
        _, RESULTS_DIR = get_workspace_paths()
        points_csv_path = os.path.join(RESULTS_DIR, "points.csv")
        
        if not os.path.exists(points_csv_path):
            return Response({"data": []})
        
        loader = PointsLoader('cascade', num_objs=2)
        points = loader.load_points_as_dicts(points_csv_path)
        
        return Response({"data": points})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(["GET"])
def test_endpoint(request):
    """Simple health check endpoint [1]."""
    return Response({
        "status": "ok",
        "message": "API is running"
    })


@api_view(["POST"])
def integrate_custom_point_to_ga(request):
    """
    Integrate a custom point into the current GA generation.
    Preserved from views.py [1].
    """
    try:
        data = json.loads(request.body)
        run_id  = data.get('run_id')
        gpu     = int(data.get('gpu', 0))
        attn    = int(data.get('attn', 0))
        sparse  = int(data.get('sparse', 0))
        conv    = int(data.get('conv', 0))
        trace   = data.get('trace', 'gpt-j-65536-weighted')

        if not run_id:
            return JsonResponse({"status": "error", "message": "run_id is required"}, status=400)

        total = gpu + attn + sparse + conv
        if total != 12:
            return JsonResponse({
                "status": "error",
                "message": f"Total chiplets must equal 12, got {total}"
            }, status=400)

        chiplets = {
            "GPU": gpu, "Attention": attn,
            "Sparse": sparse, "Convolution": conv
        }

        # Evaluate the custom point
        exec_ms, energy_mj = runSingleCascade(
            chiplets=chiplets,
            trace=trace,
            save_to_csv=True,   # writes to live points.csv for polling [1]
            source='User'
        )

        return JsonResponse({
            "status": "success",
            "point": {
                "x":       exec_ms,
                "y":       energy_mj,
                "gpu":     gpu,
                "attn":    attn,
                "sparse":  sparse,
                "conv":    conv,
                "algorithm": "Custom",
                "trace":   trace,
            }
        })

    except Exception as e:
        traceback.print_exc()
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
            "Attention":   int(request.GET.get("Attention", 0)),
            "GPU":         int(request.GET.get("GPU", 0)),
            "Sparse":      int(request.GET.get("Sparse", 0)),
            "Convolution": int(request.GET.get("Convolution", 0)),
        }

        print("Chiplets received:", chiplets)
        print("Trace:", trace)
        print("Total chiplets:", sum(chiplets.values()))

        # Validate total chiplet count
        total = sum(chiplets.values())
        if total != 12:
            return Response({
                "error": f"Total chiplets must equal 12, got {total}"
            }, status=400)

        # Evaluate the point — save_to_csv=True writes it to the
        # live points.csv so it appears on the chart immediately [1]
        objectives = runSingleCascade(chiplets, trace, save_to_csv=True, source='User')
        
        print("Objectives returned from runSingleCascade:", objectives)
        
        evaluated_point = {
            "x":         objectives[0],
            "y":         objectives[1],
            "gpu":       chiplets["GPU"],
            "attn":      chiplets["Attention"],
            "sparse":    chiplets["Sparse"],
            "conv":      chiplets["Convolution"],
            "algorithm": "Custom",
            "trace":     trace,
        }
        
        return Response({
            "evaluated_point": evaluated_point,
            "message": "Point evaluated successfully"
        })
        
    except Exception as e:
        print(f"Error in evaluate_point_inputs: {e}")
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)
    

@api_view(["GET"])
def evaluate_point(request):
    """
    Evaluate a point using a list of chiplet keys.
    Legacy endpoint preserved from views.py [1].
    """
    print("Made it to the evaluate_point function")
    try:
        chiplet_keys = request.GET.getlist("chiplets[]")
        trace = request.GET.get("trace", "gpt-j-65536-weighted")
        
        print("Chiplets:", chiplet_keys)
        print("Trace:", trace)
        
        # Count occurrences of each chiplet type
        chiplets = {}
        for key in chiplet_keys:
            chiplets[key] = chiplets.get(key, 0) + 1
        
        # Ensure all expected keys exist
        full_chiplets = {
            "GPU":         chiplets.get("GPU", 0),
            "Attention":   chiplets.get("Attention", 0),
            "Sparse":      chiplets.get("Sparse", 0),
            "Convolution": chiplets.get("Convolution", 0),
        }

        objectives = runSingleCascade(full_chiplets, trace, save_to_csv=True, source='User')
        
        evaluated_point = {
            "x":         objectives[0],
            "y":         objectives[1],
            "gpu":       full_chiplets["GPU"],
            "attn":      full_chiplets["Attention"],
            "sparse":    full_chiplets["Sparse"],
            "conv":      full_chiplets["Convolution"],
            "algorithm": "Custom",
            "trace":     trace,
        }
        
        return Response({
            "evaluated_point": evaluated_point,
            "message": "Point evaluated successfully"
        })
        
    except Exception as e:
        print(f"Error in evaluate_point: {e}")
        traceback.print_exc()
        return Response({"error": str(e)}, status=500)