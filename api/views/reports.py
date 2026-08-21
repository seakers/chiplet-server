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
from pathlib import Path
import traceback

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
from api.config.objectives import to_field, to_fields, to_axis_label, DEFAULT_OBJECTIVES


@api_view(["GET"])
def generate_report(request):
    """
    Generate an HTML report for the current optimization run.
    Evaluator-aware: routes CASCADE vs PISTIL runs to the correct
    points file, agents, and output directory.
    """
    try:
        from api.models import OptimizationRun
        import sys
        from pathlib import Path

        # --- 1. Read query params FIRST ---
        run_id = request.GET.get("run_id")
        evaluator_hint = request.GET.get("evaluator", "cascade")
        selected_objectives = request.GET.getlist("objectives[]")  # optional list of objectives

        # --- 2. Resolve evaluator (DB > run_id prefix > hint > default) ---
        run_obj = None
        if run_id:
            run_obj = OptimizationRun.objects.filter(run_id=run_id).first()

        if run_obj:
            evaluator = run_obj.model.lower()
        elif run_id and run_id.startswith('pistil_run_'):
            evaluator = 'pistil'
            print(f"[generate_report] No DB record for {run_id}; inferring PISTIL from prefix.")
        elif run_id and run_id.startswith('restarted_run_'):
            evaluator = 'cascade'  # restarted CASCADE runs
            print(f"[generate_report] No DB record for {run_id}; inferring CASCADE from prefix.")
        else:
            evaluator = evaluator_hint.lower()

        print(f"[generate_report] run_id={run_id}, evaluator={evaluator}")

        # --- 3. Choose correct points file ---
        if evaluator == 'pistil':
            if not run_id:
                return JsonResponse(
                    {"status": "error", "message": "PISTIL reports require run_id"},
                    status=400
                )
            file_path = str(
                Path(sys.path[0])
                / "api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results"
                / run_id / "points.csv"
            )
        else:
            file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"

        loader = PointsLoader(evaluator, run_id)
        points = loader.load_points_as_dicts(file_path)

        if not points:
            exists = os.path.exists(file_path)
            msg = (
                f"No data available for report. "
                f"Evaluator={evaluator}, run_id={run_id}, "
                f"file={file_path}, exists={exists}."
            )
            print(f"[generate_report] {msg}")
            return JsonResponse({"status": "error", "message": msg}, status=400)

        # --- 4. Build run_params from DB (with sensible defaults) ---
        default_objs = DEFAULT_OBJECTIVES.get(evaluator, ['Runtime', 'Energy'])
        run_params = {
            'model': evaluator.upper(),
            'algorithm': 'Genetic Algorithm',
            'objectives': selected_objectives if selected_objectives else default_objs,
            'population_size': 50,
            'generations': 100,
            'trace_name': 'gpt-j-65536-weighted' if evaluator == 'cascade' else 'pistil-default',
        }
        if run_obj:
            run_params = {
                'model': run_obj.model,
                'algorithm': run_obj.get_algorithm_display(),
                'objectives': selected_objectives if selected_objectives else run_obj.objectives or default_objs,
                'population_size': run_obj.population_size,
                'generations': run_obj.generations,
                'trace_name': run_obj.trace_name or run_params['trace_name'],
            }
            print(f"[generate_report] Using params from DB: {run_params}")

        # --- 5. Pareto front ---
        pareto_calculator = ParetoCalculator()
        pareto_points = pareto_calculator.get_pareto_front(points)

        # --- 6. Rule mining via RuleMiningAgent (ChatBot.rule_mining() is gone) ---
        from api.chatbot.agents.rule_mining_agent import RuleMiningAgent
        rm_agent = RuleMiningAgent(evaluator, run_id)
        rm_result = rm_agent.execute({
            'evaluator': evaluator,
            'run_id': run_id,
            'objectives': run_params.get('objectives'),
            'max_pareto_rank': 3,
            'use_all_points': True,   # reports analyze the whole run
        })

        rules = []
        if rm_result.success and rm_result.data:
            # rm_result.data typically has a 'rules' list with structured entries
            raw_rules = rm_result.data.get('rules', []) if isinstance(rm_result.data, dict) else []
            for r in raw_rules:
                rules.append({
                    "rule": r.get('rule', ''),
                    "conf_f_to_p": float(r.get('conf_f_to_p', 0.0)),
                    "conf_p_to_f": float(r.get('conf_p_to_f', 0.0)),
                    "lift": float(r.get('lift', 0.0)),
                })
        else:
            print(f"[generate_report] Rule mining failed or empty: {rm_result.message}")

        # --- 7. Distance correlation ---
        analyzer = DistanceCorrelationAnalyzer(evaluator)
        corr_results = analyzer.analyze_from_points(
            points,
            requested_objectives=run_params.get('objectives'),
        )
        correlations = corr_results.get('correlations', {})

        # --- 8. Generate HTML ---
        report_generator = ReportGenerator(evaluator)
        html_content = report_generator.generate_single_run_report(
            run_params=run_params,
            points=points,
            pareto_points=pareto_points,
            rules=rules,
            correlations=correlations,
            run_id=run_id,
        )

        # --- 9. Save to evaluator-appropriate directory ---
        if evaluator == 'pistil':
            reports_dir = os.path.join(
                sys.path[0],
                'api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/reports'
            )
            web_link_prefix = '/api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/reports'
        else:
            reports_dir = os.path.join(
                sys.path[0],
                'api/Evaluator/cascade/chiplet_model/dse/results/reports'
            )
            web_link_prefix = '/api/Evaluator/cascade/chiplet_model/dse/results/reports'

        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"report_{run_id or 'current'}_{timestamp}.html"
        report_path = os.path.join(reports_dir, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        web_link = f"{web_link_prefix}/{report_filename}"

        return JsonResponse({
            "status": "success",
            "report_content": html_content,
            "web_link": web_link,
            "download_link": web_link,
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
        corr_results = analyzer.analyze_from_points(
            points,
            requested_objectives=run_params.get('objectives'),
        )
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

        # ── Determine evaluator and resolve file path ──────────────────────────
        is_pistil = backup_filename.startswith('pistil_run_')

        if is_pistil:
            pistil_root = Path(sys.path[0]) / "api" / "Evaluator" / "sim-v2-4-pistil-sim-clean" / "dse" / "results"
            backup_path = pistil_root / backup_filename / "points.csv"
            evaluator = 'pistil'
        else:
            backup_path = Path(os.path.join(results_dir, backup_filename))
            evaluator = 'cascade'

        if not os.path.exists(str(backup_path)):
            return JsonResponse(
                {"status": "error", "message": f"File not found: {backup_filename}"},
                status=404
            )

        # ── Build a run_id from the filename ──────────────────────────────────
        run_id = backup_filename

        # ── Load points ────────────────────────────────────────────────────────
        loader = PointsLoader(evaluator)
        points = loader.load_points_as_dicts(str(backup_path))

        if not points:
            return JsonResponse(
                {"status": "error", "message": "No valid data found in file"},
                status=400
            )

        # ── Tag each point with model info ────────────────────────────────────
        for point in points:
            if not point.get('model'):
                point['model'] = 'PISTIL' if is_pistil else 'CASCADE'

        # ── Write a temp CSV for live-polling compatibility ────────────────────
        if is_pistil:
            # PISTIL results live in their own directory; write a temp CSV
            # alongside so the chart polling endpoint can find it
            temp_dir = pistil_root / f"temp_points_{run_id}"
            os.makedirs(str(temp_dir), exist_ok=True)
            temp_path = str(temp_dir / "points.csv")
        else:
            temp_path = os.path.join(results_dir, f"temp_points_{run_id}.csv")

        try:
            with open(temp_path, 'w', newline='') as f:
                writer = csv.writer(f)
                if points:
                    writer.writeheader() if hasattr(writer, 'writeheader') else None
                    # Write rows based on evaluator type
                    if is_pistil:
                        for p in points:
                            writer.writerow([
                                p.get('num_cus', ''),
                                p.get('num_tmacs', ''),
                                p.get('mem_buf_cap', ''),
                                p.get('batch_size', ''),
                                p.get('latency_per_token_ms', p.get('x', '')),
                                p.get('energy_per_inference_mJ', p.get('y', '')),
                            ])
                    else:
                        for p in points:
                            writer.writerow([
                                p.get('x', ''),
                                p.get('y', ''),
                                p.get('gpu', ''),
                                p.get('attn', ''),
                                p.get('sparse', ''),
                                p.get('conv', ''),
                            ])
        except Exception as write_err:
            print(f"[load_previous_run] Warning: could not write temp CSV: {write_err}")

        # ── Load zip metadata if available (CASCADE only) ─────────────────────
        metadata = {}
        if not is_pistil:
            try:
                run_data_loader = RunDataLoader('cascade')
                metadata = run_data_loader.load_run_metadata(str(backup_filename))
            except Exception as meta_err:
                print(f"[load_previous_run] Warning: could not load metadata: {meta_err}")
        else:
            # Build basic metadata from the PISTIL run directory name
            try:
                timestamp = datetime.strptime(backup_filename.replace("pistil_run_", ""), "%Y%m%d_%H%M%S")
                metadata = {
                    'model': 'PISTIL',
                    'algorithm': 'Genetic Algorithm',
                    'timestamp': timestamp.isoformat(),
                    'point_count': len(points),
                }
            except ValueError:
                metadata = {
                    'model': 'PISTIL',
                    'algorithm': 'Unknown',
                    'timestamp': None,
                    'point_count': len(points),
                }

        return JsonResponse({
            "status": "success",
            "data": points,
            "metadata": metadata,
            "run_id": run_id,
            "evaluator": evaluator.upper(),
            "source": "file",
            "backup_filename": backup_filename,
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
    

# api/views/reports.py

@api_view(["GET"])
def generate_comparative_report(request):
    """
    Generate an HTML report comparing two optimization runs.

    Query params:
        run_a_id (str, required): run_id of the first run
        run_b_id (str, required): run_id of the second run

    Both runs must exist in the DB (or be loadable via PointsLoader).
    Cross-evaluator comparisons are rejected — the ReportGenerator
    warns about them and the frontend already blocks them [1].
    """
    try:
        run_a_id = request.GET.get("run_a_id")
        run_b_id = request.GET.get("run_b_id")

        # NEW: user-selected objectives (comma-separated friendly names)
        objectives_param = request.GET.get("objectives")
        requested_objectives = (
            [o.strip() for o in objectives_param.split(",") if o.strip()]
            if objectives_param else None
        )

        objs = (requested_objectives
            or run_a_info["params"].get("objectives")
            or DEFAULT_OBJECTIVES.get(evaluator, []))

        if not run_a_id or not run_b_id:
            return JsonResponse({
                "status": "error",
                "message": "Both run_a_id and run_b_id are required."
            }, status=400)

        if run_a_id == run_b_id:
            return JsonResponse({
                "status": "error",
                "message": "Cannot compare a run against itself."
            }, status=400)

        # --- 1. Resolve evaluator + params for each run ---
        run_a_info = _resolve_run_for_report(run_a_id)
        run_b_info = _resolve_run_for_report(run_b_id)

        if run_a_info["evaluator"] != run_b_info["evaluator"]:
            return JsonResponse({
                "status": "error",
                "message": (
                    f"Cannot compare runs across different evaluators "
                    f"(Run A: {run_a_info['evaluator'].upper()}, "
                    f"Run B: {run_b_info['evaluator'].upper()}). "
                    f"Please select two runs using the same evaluator."
                ),
            }, status=400)

        evaluator = run_a_info["evaluator"]
        print(f"[generate_comparative_report] {run_a_id} vs {run_b_id} ({evaluator})")

        # --- 2. Load points for each run ---
        loader_a = PointsLoader(evaluator, run_a_id)
        loader_b = PointsLoader(evaluator, run_b_id)
        points_a = loader_a.load_points_as_dicts()
        points_b = loader_b.load_points_as_dicts()

        if not points_a:
            return JsonResponse({
                "status": "error",
                "message": f"Run A ({run_a_id}) has no points."
            }, status=400)
        if not points_b:
            return JsonResponse({
                "status": "error",
                "message": f"Run B ({run_b_id}) has no points."
            }, status=400)

        # --- 3. Pareto fronts ---
        pareto_a = ParetoCalculator.get_pareto_front(points_a)
        pareto_b = ParetoCalculator.get_pareto_front(points_b)

        # Make selected objectives authoritative on both params dicts
        run_a_info["params"]["objectives"] = objs
        run_b_info["params"]["objectives"] = objs

        # --- 4. Rule mining per run ---
        run_a_rules = _mine_rules_for_run(evaluator, run_a_id, run_a_info["params"])
        run_b_rules = _mine_rules_for_run(evaluator, run_b_id, run_b_info["params"])

        # --- 5. Distance correlation per run (same objectives for both) ---
        analyzer = DistanceCorrelationAnalyzer(evaluator)
        corr_a = analyzer.analyze_from_points(points_a, requested_objectives=objs).get("correlations", {})
        corr_b = analyzer.analyze_from_points(points_b, requested_objectives=objs).get("correlations", {})

        # --- 6. Assemble run_data dicts the ReportGenerator expects ---
        run_a_data = {
            "params":         run_a_info["params"],
            "points":         points_a,
            "pareto_points":  pareto_a,        # was "pareto" — generator reads "pareto_points" [19]
            "rules":          run_a_rules,
            "correlations":   corr_a,
        }
        run_b_data = {
            "params":         run_b_info["params"],
            "points":         points_b,
            "pareto_points":  pareto_b,        # was "pareto"
            "rules":          run_b_rules,
            "correlations":   corr_b,
        }

        # --- 7. Generate HTML ---
        report_generator = ReportGenerator(evaluator)
        html_content = report_generator.generate_comparative_report(
            run_a_id=run_a_id,
            run_b_id=run_b_id,
            run_a_data=run_a_data,
            run_b_data=run_b_data,
            requested_objectives=objs,          # NEW
        )

        # --- 8. Save to disk (same directory scheme as single-run reports [8]) ---
        if evaluator == "pistil":
            reports_dir = os.path.join(
                sys.path[0],
                "api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/reports",
            )
            web_link_prefix = "/api/Evaluator/sim-v2-4-pistil-sim-clean/dse/results/reports"
        else:
            reports_dir = os.path.join(
                sys.path[0],
                "api/Evaluator/cascade/chiplet_model/dse/results/reports",
            )
            web_link_prefix = "/api/Evaluator/cascade/chiplet_model/dse/results/reports"

        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"comparative_report_{run_a_id}_vs_{run_b_id}_{timestamp}.html"
        report_path = os.path.join(reports_dir, report_filename)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        web_link = f"{web_link_prefix}/{report_filename}"

        return JsonResponse({
            "status":          "success",
            "web_link":        web_link,
            "download_link":   web_link,
            "report_filename": report_filename,
            "metadata": {
                "run_a_id":       run_a_id,
                "run_b_id":       run_b_id,
                "evaluator":      evaluator.upper(),
                "objectives":     objs,
                "run_a_points":   len(points_a),
                "run_b_points":   len(points_b),
                "run_a_pareto":   len(pareto_a),
                "run_b_pareto":   len(pareto_b),
                "run_a_rules":    len(run_a_rules),
                "run_b_rules":    len(run_b_rules),
            },
        })

    except Exception as e:
        print(f"[generate_comparative_report] Error: {e}")
        traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _resolve_run_for_report(run_id: str) -> dict:
    """
    Resolve evaluator + run_params for a given run_id, mirroring the logic
    the single-run report already uses [8].

    Returns:
        {
            "evaluator": "cascade" | "pistil",
            "params":    { model, algorithm, objectives, trace_name, ... },
        }
    """
    run_obj = OptimizationRun.objects.filter(run_id=run_id).first()

    # Resolve evaluator from DB > prefix > default
    if run_obj:
        evaluator = run_obj.model.lower()
    elif run_id.startswith("pistil_run_"):
        evaluator = "pistil"
    elif run_id.startswith("restarted_run_"):
        evaluator = "cascade"
    else:
        evaluator = "cascade"

    default_objs = DEFAULT_OBJECTIVES.get(evaluator, ["Runtime", "Energy"])

    if run_obj:
        params = {
            "model":           run_obj.model,
            "algorithm":       run_obj.get_algorithm_display(),
            "objectives":      run_obj.objectives or default_objs,
            "population_size": run_obj.population_size,
            "generations":     run_obj.generations,
            "trace_name":      run_obj.trace_name or (
                "gpt-j-65536-weighted" if evaluator == "cascade" else "pistil-default"
            ),
        }
    else:
        params = {
            "model":           evaluator.upper(),
            "algorithm":       "Genetic Algorithm",
            "objectives":      default_objs,
            "population_size": 50,
            "generations":     100,
            "trace_name":      "gpt-j-65536-weighted" if evaluator == "cascade" else "pistil-default",
        }

    return {"evaluator": evaluator, "params": params}


def _mine_rules_for_run(evaluator: str, run_id: str, run_params: dict) -> list:
    """
    Run rule mining for a single run, using the same regex-parse approach
    the single-run report already uses [8]. Returns a list of rule dicts
    with keys: rule, conf_f_to_p, conf_p_to_f, lift.
    """
    try:
        chat_bot = ChatBot(evaluator=evaluator, run_id=run_id)
        objectives = run_params.get("objectives")
        if objectives:
            chat_bot.set_objectives(objectives)

        rule_mining_str = chat_bot.rule_mining()

        rules = []
        pattern = re.compile(
            r"Rule: (.*?), conf\(f->p\): ([0-9.eE+-]+), "
            r"conf\(p->f\): ([0-9.eE+-]+), lift: \(?([0-9.eE+-]+)\)?"
        )
        for m in pattern.finditer(rule_mining_str):
            rules.append({
                "rule":         m.group(1),
                "conf_f_to_p":  float(m.group(2)),
                "conf_p_to_f":  float(m.group(3)),
                "lift":         float(m.group(4)),
            })
        return rules
    except Exception as e:
        print(f"[_mine_rules_for_run] Failed for {run_id}: {e}")
        return []