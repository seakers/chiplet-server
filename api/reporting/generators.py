"""
Report generation utilities.
Evaluator- and objective-agnostic — driven by EvaluatorConfig + selected
objectives instead of CASCADE-specific hardcoding.
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from api.analysis.rule_mining import RuleFormatter
from api.analysis.pareto import ParetoCalculator
from api.config.evaluators import get_evaluator_config, EvaluatorConfig
from api.config.objectives import to_axis_label

from .formatters import (
    ReportFormatter,
    get_point_objective_value,
    get_point_decision_value,
    resolve_objectives,
)


class ReportGenerator:
    """
    Generates HTML reports for optimization runs.
    Works for any evaluator (CASCADE, PISTIL, ...) and any set of user-selected
    objectives.
    """

    BASE_STYLES = """
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; margin-top: 30px; }
    h3 { color: #7f8c8d; }
    .section { margin-bottom: 30px; padding: 20px; background: #f9f9f9; border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; margin: 15px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background-color: #3498db; color: white; }
    tr:hover { background-color: #f5f5f5; }
    .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
    .stat-card { background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    .stat-number { font-size: 1.6em; font-weight: bold; color: #3498db; }
    .stat-label { color: #7f8c8d; margin-top: 5px; font-size: 0.9em; }
    .header-info { background: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
    .header-info p { margin: 5px 0; color: #2c3e50; }
    .insight-box { background: #e8f6f3; padding: 15px; border-radius: 8px; border-left: 4px solid #1abc9c; margin: 10px 0; }
    .recommendation { background: #fef9e7; padding: 15px; border-radius: 8px; border-left: 4px solid #f39c12; margin: 10px 0; }
    .wide-table { overflow-x: auto; }
    """

    COMPARATIVE_STYLES = """
    .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .run-column { padding: 15px; border-radius: 8px; }
    .run-a-header { background-color: #3498db; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .run-b-header { background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .winner { background-color: #d5f4e6; border: 2px solid #27ae60; }
    .loser  { background-color: #fce4e4; border: 2px solid #e74c3c; }
    """

    def __init__(self, evaluator: str = 'cascade'):
        self.evaluator = evaluator.lower()
        self.config: EvaluatorConfig = get_evaluator_config(self.evaluator)
        self.pareto_calculator = ParetoCalculator()
        self.rule_formatter = RuleFormatter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_single_run_report(self,
                                   run_params: Dict[str, Any],
                                   points: List[Dict[str, Any]],
                                   pareto_points: List[Dict[str, Any]],
                                   rules: List[Dict[str, Any]],
                                   correlations: Dict[str, float],
                                   run_id: str = None) -> str:
        """Generate HTML report for a single optimization run."""
        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H:%M:%S")

        objectives = resolve_objectives(run_params, self.config)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Report - {run_id or 'Single Run'}</title>
    <style>
{self.BASE_STYLES}
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimization Report</h1>

        <div class="header-info">
            <p><strong>Date:</strong> {date_str}</p>
            <p><strong>Time:</strong> {time_str}</p>
            <p><strong>Evaluator:</strong> {self.evaluator.upper()}</p>
            <p><strong>Report Type:</strong> Tradespace Exploration Run</p>
            {f'<p><strong>Run ID:</strong> {run_id}</p>' if run_id else ''}
        </div>

        {self._generate_problem_formulation_section(run_params, objectives)}
        {self._generate_results_summary_section(points, pareto_points, objectives)}
        {self._generate_rule_mining_section(rules)}
        {self._generate_correlation_section(correlations, objectives)}
        {self._generate_pareto_front_section(pareto_points, objectives)}
    </div>
</body>
</html>"""
        return html_content

    def generate_comparative_report(self,
                                    run_a_id: str,
                                    run_b_id: str,
                                    run_a_data: Dict[str, Any],
                                    run_b_data: Dict[str, Any],
                                    requested_objectives: List[str] = None) -> str:
        """Generate HTML report comparing two optimization runs."""
        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H:%M:%S")

        # Extract data from run dictionaries
        run_a_points       = run_a_data.get('points', [])
        run_b_points       = run_b_data.get('points', [])
        run_a_pareto       = run_a_data.get('pareto_points', [])
        run_b_pareto       = run_b_data.get('pareto_points', [])
        run_a_rules        = run_a_data.get('rules', [])
        run_b_rules        = run_b_data.get('rules', [])
        run_a_correlations = run_a_data.get('correlations', {})
        run_b_correlations = run_b_data.get('correlations', {})
        run_a_params       = run_a_data.get('params', {})
        run_b_params       = run_b_data.get('params', {})

        # Resolve objectives. User selection wins; else prefer Run A's, then Run B's.
        objectives_a = resolve_objectives(run_a_params, self.config)
        objectives_b = resolve_objectives(run_b_params, self.config)
        objectives = requested_objectives or objectives_a or objectives_b
        objectives_match = objectives_a == objectives_b

        # Recompute stats per objective from raw points so we're in sync
        run_a_stats = ReportFormatter.format_statistics(run_a_points, objectives)
        run_b_stats = ReportFormatter.format_statistics(run_b_points, objectives)
        run_a_stats['pareto_count'] = len(run_a_pareto)
        run_b_stats['pareto_count'] = len(run_b_pareto)

        # Pareto dominance (uses x/y — first two objectives)
        a_dominates_b = self.pareto_calculator.count_dominated_points(run_a_pareto, run_b_pareto)
        b_dominates_a = self.pareto_calculator.count_dominated_points(run_b_pareto, run_a_pareto)

        objectives_warning = ""
        if not objectives_match:
            objectives_warning = (
                f'<p style="color:#c0392b;"><strong>Note:</strong> Run A and Run B '
                f'optimized different objectives ({", ".join(objectives_a)} vs '
                f'{", ".join(objectives_b)}). Comparison uses Run A\'s objectives.</p>'
            )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comparative Report - {run_a_id} vs {run_b_id}</title>
    <style>
{self.BASE_STYLES}
{self.COMPARATIVE_STYLES}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comparative Optimization Report</h1>

        <div class="header-info">
            <p><strong>Date:</strong> {date_str}</p>
            <p><strong>Time:</strong> {time_str}</p>
            <p><strong>Evaluator:</strong> {self.evaluator.upper()}</p>
            <p><strong>Run A:</strong> {run_a_id}</p>
            <p><strong>Run B:</strong> {run_b_id}</p>
            {objectives_warning}
        </div>

        {self._generate_comparative_summary_section(run_a_stats, run_b_stats, a_dominates_b, b_dominates_a, objectives)}
        {self._generate_comparative_params_section(run_a_params, run_b_params)}
        {self._generate_comparative_pareto_section(run_a_pareto, run_b_pareto, objectives)}
        {self._generate_comparative_rules_section(run_a_rules, run_b_rules)}
        {self._generate_comparative_correlations_section(run_a_correlations, run_b_correlations, objectives)}
        {self._generate_insights_section(run_a_stats, run_b_stats, run_a_pareto, run_b_pareto, run_a_rules, run_b_rules, a_dominates_b, b_dominates_a, objectives)}
    </div>
</body>
</html>"""
        return html_content

    # ------------------------------------------------------------------
    # Single-run sections
    # ------------------------------------------------------------------

    def _generate_problem_formulation_section(self,
                                              run_params: Dict[str, Any],
                                              objectives: List[str]) -> str:
        """Problem formulation — decision-variable table + objectives."""
        # Decision-variable table (dynamic per evaluator)
        dec_rows = "".join(f"<li>{dec}</li>" for dec in self.config.decision_columns)

        # Slot / total-chiplet constraint for CASCADE
        slot_note = ""
        if self.evaluator == 'cascade' and getattr(self.config, 'num_slots', None):
            slot_note = (
                f"<li>Total chiplet slots: <strong>{self.config.num_slots}</strong> "
                f"(sum of all chiplet counts must equal {self.config.num_slots})</li>"
            )

        # Objectives — use friendly axis label with units [13]
        objectives_html = "".join(
            f'<li>Objective {i+1}: Minimize <strong>{to_axis_label(obj)}</strong></li>'
            if self.config.minimize_for(
                # convert friendly → internal field to check direction
                __import__('api.config.objectives', fromlist=['to_field']).to_field(obj)
            )
            else f'<li>Objective {i+1}: Maximize <strong>{to_axis_label(obj)}</strong></li>'
            for i, obj in enumerate(objectives)
        )

        return f"""
        <div class="section">
            <h2>Problem Formulation</h2>
            <p><strong>Decision Variables:</strong></p>
            <ul>
                {dec_rows}
                {slot_note}
            </ul>
            <p><strong>Objectives:</strong></p>
            <ul>
                <li>Optimization type: {"Multi-objective" if len(objectives) > 1 else "Single-objective"}</li>
                {objectives_html}
            </ul>
            <p><strong>Model and Search Parameters:</strong></p>
            <ul>
                <li>Model: {run_params.get('model', self.evaluator.upper())}</li>
                <li>Trace: {run_params.get('trace_name', 'N/A')}</li>
                <li>Search Algorithm: {run_params.get('algorithm', 'Genetic Algorithm')}</li>
                <li>Population Size: {run_params.get('population_size', 'N/A')}</li>
                <li>Generations: {run_params.get('generations', 'N/A')}</li>
            </ul>
        </div>"""

    def _generate_results_summary_section(self,
                                          points: List[Dict[str, Any]],
                                          pareto_points: List[Dict[str, Any]],
                                          objectives: List[str]) -> str:
        """Results summary — stat cards for total, Pareto, and per-objective avg."""
        stats = ReportFormatter.format_statistics(points, objectives)

        cards = [
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(points)}</div>
                <div class="stat-label">Total Designs Evaluated</div>
            </div>""",
            f"""
            <div class="stat-card">
                <div class="stat-number">{len(pareto_points)}</div>
                <div class="stat-label">Designs on Pareto Front</div>
            </div>""",
        ]

        # One card per selected objective showing the average
        for obj_name in objectives:
            s = stats['per_objective'][obj_name]
            cards.append(f"""
            <div class="stat-card">
                <div class="stat-number">{s['avg']:.2f}</div>
                <div class="stat-label">Avg {s['label']}</div>
            </div>""")

        return f"""
        <div class="section">
            <h2>Results Summary</h2>
            <div class="stats">
                {''.join(cards)}
            </div>
        </div>"""

    def _generate_pareto_front_section(self,
                                       pareto_points: List[Dict[str, Any]],
                                       objectives: List[str]) -> str:
        """Pareto front table — dynamic decision + objective columns."""
        dec_cols = self.config.decision_columns

        # Header
        header_cells = "".join(f"<th>{d}</th>" for d in dec_cols)
        header_cells += "".join(f"<th>{to_axis_label(obj)}</th>" for obj in objectives)

        # Rows
        rows_html = ""
        for point in pareto_points:
            cells = ""
            for dec in dec_cols:
                val = get_point_decision_value(point, dec)
                try:
                    cells += f"<td>{int(float(val))}</td>"
                except (TypeError, ValueError):
                    cells += f"<td>{val}</td>"
            for i, obj in enumerate(objectives):
                v = get_point_objective_value(point, obj, i)
                cells += f"<td>{v:.2f}</td>"
            rows_html += f"<tr>{cells}</tr>"

        # Axis-label caption (only meaningful for the first two)
        caption_bits = []
        if len(objectives) >= 1:
            caption_bits.append(f"X → {to_axis_label(objectives[0])}")
        if len(objectives) >= 2:
            caption_bits.append(f"Y → {to_axis_label(objectives[1])}")
        if len(objectives) >= 3:
            caption_bits.append(f"Z → {to_axis_label(objectives[2])}")
        caption = ", ".join(caption_bits)

        return f"""
        <div class="section">
            <h2>All Points</h2>
            <p>{caption}</p>
            <div class="wide-table">
            <table>
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
            </div>
        </div>"""

    def _generate_rule_mining_section(self, rules: List[Dict[str, Any]]) -> str:
        """Rule mining — unchanged in structure, still evaluator-agnostic."""
        if not rules:
            body = "<tr><td colspan='4'><em>No significant rules found</em></td></tr>"
        else:
            body = ""
            for rule in rules:
                formatted_rule = RuleFormatter.format_rule_natural_language(rule.get('rule', ''))
                body += f"""
                <tr>
                    <td>{formatted_rule}</td>
                    <td>{rule.get('conf_f_to_p', 0):.2f}</td>
                    <td>{rule.get('conf_p_to_f', 0):.2f}</td>
                    <td>{rule.get('lift', 0):.2f}</td>
                </tr>"""

        return f"""
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
                <tbody>{body}</tbody>
            </table>
        </div>"""

    def _generate_correlation_section(self,
                                      correlations: Dict[str, float],
                                      objectives: List[str]) -> str:
        """Distance correlation — rows = decision vars, cols = objectives."""
        rows = ReportFormatter.format_correlations_table(correlations, self.config, objectives)

        header_cells = "<th>Decision Variable</th>" + "".join(
            f"<th>vs {to_axis_label(obj)}</th>" for obj in objectives
        )

        body = ""
        for row in rows:
            cells = f"<td>{row['decision']}</td>"
            for entry in row['values']:
                cells += f"<td>{entry['value']:.3f}</td>"
            body += f"<tr>{cells}</tr>"

        return f"""
        <div class="section">
            <h2>Distance Correlation Analysis</h2>
            <p>Correlation between decision variables and objectives</p>
            <div class="wide-table">
            <table>
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{body}</tbody>
            </table>
            </div>
        </div>"""

    # ------------------------------------------------------------------
    # Comparative-report sections
    # ------------------------------------------------------------------

    def _generate_comparative_summary_section(self,
                                              run_a_stats: Dict[str, Any],
                                              run_b_stats: Dict[str, Any],
                                              a_dominates_b: int,
                                              b_dominates_a: int,
                                              objectives: List[str]) -> str:
        """Side-by-side summary — dynamic per-objective stats."""
        def render_column(stats, header_class, header_label, dominates_label, dominates_count, is_winner):
            winner_class = 'winner' if is_winner else ('loser' if not is_winner and (a_dominates_b != b_dominates_a) else '')
            cards = f"""
            <div class="stat-card">
                <div class="stat-number">{stats.get('total', 0)}</div>
                <div class="stat-label">Total Points</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats.get('pareto_count', 0)}</div>
                <div class="stat-label">Pareto Points</div>
            </div>"""

            obj_lines = ""
            for obj_name in objectives:
                s = stats['per_objective'].get(obj_name, {})
                obj_lines += (
                    f"<p><strong>Avg {s.get('label', obj_name)}:</strong> "
                    f"{s.get('avg', 0):.2f}</p>"
                    f"<p><strong>Min {s.get('label', obj_name)}:</strong> "
                    f"{s.get('min', 0):.2f}</p>"
                )

            return f"""
            <div class="run-column {winner_class}">
                <div class="{header_class}">{header_label}</div>
                <div class="stats">{cards}</div>
                {obj_lines}
                <p><strong>{dominates_label}:</strong> {dominates_count} points</p>
            </div>"""

        col_a = render_column(
            run_a_stats, 'run-a-header', 'Run A',
            'Dominates Run B', a_dominates_b,
            is_winner=a_dominates_b > b_dominates_a,
        )
        col_b = render_column(
            run_b_stats, 'run-b-header', 'Run B',
            'Dominates Run A', b_dominates_a,
            is_winner=b_dominates_a > a_dominates_b,
        )

        return f"""
        <div class="section">
            <h2>Comparative Summary</h2>
            <div class="comparison-grid">{col_a}{col_b}</div>
        </div>"""

    def _generate_comparative_params_section(self,
                                             run_a_params: Dict[str, Any],
                                             run_b_params: Dict[str, Any]) -> str:
        """Side-by-side run configuration table."""
        def objs_str(p):
            o = p.get('objectives', [])
            return ', '.join(o) if isinstance(o, list) else str(o)

        rows = [
            ('Model',           run_a_params.get('model', 'N/A'),           run_b_params.get('model', 'N/A')),
            ('Algorithm',       run_a_params.get('algorithm', 'N/A'),       run_b_params.get('algorithm', 'N/A')),
            ('Objectives',      objs_str(run_a_params),                     objs_str(run_b_params)),
            ('Trace',           run_a_params.get('trace_name', 'N/A'),      run_b_params.get('trace_name', 'N/A')),
            ('Population Size', run_a_params.get('population_size', 'N/A'), run_b_params.get('population_size', 'N/A')),
            ('Generations',     run_a_params.get('generations', 'N/A'),     run_b_params.get('generations', 'N/A')),
        ]

        body = "".join(
            f"<tr><td>{label}</td><td>{a}</td><td>{b}</td></tr>"
            for label, a, b in rows
        )

        return f"""
        <div class="section">
            <h2>Run Configuration Comparison</h2>
            <table>
                <thead>
                    <tr>
                        <th>Parameter</th>
                        <th>Run A</th>
                        <th>Run B</th>
                    </tr>
                </thead>
                <tbody>{body}</tbody>
            </table>
        </div>"""

    def _generate_comparative_pareto_section(self,
                                             run_a_pareto: List[Dict[str, Any]],
                                             run_b_pareto: List[Dict[str, Any]],
                                             objectives: List[str]) -> str:
        """Side-by-side Pareto front tables — dynamic decision + objective columns."""
        dec_cols = self.config.decision_columns

        # Shared header
        header_cells = "".join(f"<th>{d}</th>" for d in dec_cols)
        header_cells += "".join(f"<th>{to_axis_label(obj)}</th>" for obj in objectives)

        def generate_pareto_table(pareto_points: List[Dict[str, Any]], max_rows: int = 10) -> str:
            rows_html = ""
            total_cols = len(dec_cols) + len(objectives)
            for point in pareto_points[:max_rows]:
                cells = ""
                for dec in dec_cols:
                    val = get_point_decision_value(point, dec)
                    try:
                        cells += f"<td>{int(float(val))}</td>"
                    except (TypeError, ValueError):
                        cells += f"<td>{val}</td>"
                for i, obj in enumerate(objectives):
                    v = get_point_objective_value(point, obj, i)
                    cells += f"<td>{v:.2f}</td>"
                rows_html += f"<tr>{cells}</tr>"

            if len(pareto_points) > max_rows:
                rows_html += (
                    f"<tr><td colspan='{total_cols}'>"
                    f"<em>... and {len(pareto_points) - max_rows} more points</em>"
                    f"</td></tr>"
                )
            return rows_html

        return f"""
        <div class="section">
            <h2>Pareto Front Comparison</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-a-header">Run A Pareto Front ({len(run_a_pareto)} points)</div>
                    <div class="wide-table">
                    <table>
                        <thead><tr>{header_cells}</tr></thead>
                        <tbody>{generate_pareto_table(run_a_pareto)}</tbody>
                    </table>
                    </div>
                </div>
                <div class="run-column">
                    <div class="run-b-header">Run B Pareto Front ({len(run_b_pareto)} points)</div>
                    <div class="wide-table">
                    <table>
                        <thead><tr>{header_cells}</tr></thead>
                        <tbody>{generate_pareto_table(run_b_pareto)}</tbody>
                    </table>
                    </div>
                </div>
            </div>
        </div>"""

    def _generate_comparative_rules_section(self,
                                            run_a_rules: List[Dict[str, Any]],
                                            run_b_rules: List[Dict[str, Any]]) -> str:
        """Side-by-side rule mining tables (evaluator-agnostic)."""
        def generate_rules_table(rules: List[Dict[str, Any]]) -> str:
            if not rules:
                return "<tr><td colspan='4'><em>No significant rules found</em></td></tr>"
            rows_html = ""
            for rule in rules[:5]:
                formatted_rule = RuleFormatter.format_rule_natural_language(rule.get('rule', ''))
                rows_html += f"""
                <tr>
                    <td>{formatted_rule}</td>
                    <td>{rule.get('conf_f_to_p', 0):.2f}</td>
                    <td>{rule.get('conf_p_to_f', 0):.2f}</td>
                    <td>{rule.get('lift', 0):.2f}</td>
                </tr>"""
            return rows_html

        return f"""
        <div class="section">
            <h2>Rule Mining Comparison</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-a-header">Run A Rules ({len(run_a_rules)} found)</div>
                    <table>
                        <thead>
                            <tr><th>Rule</th><th>F→P</th><th>P→F</th><th>Lift</th></tr>
                        </thead>
                        <tbody>{generate_rules_table(run_a_rules)}</tbody>
                    </table>
                </div>
                <div class="run-column">
                    <div class="run-b-header">Run B Rules ({len(run_b_rules)} found)</div>
                    <table>
                        <thead>
                            <tr><th>Rule</th><th>F→P</th><th>P→F</th><th>Lift</th></tr>
                        </thead>
                        <tbody>{generate_rules_table(run_b_rules)}</tbody>
                    </table>
                </div>
            </div>
        </div>"""

    def _generate_comparative_correlations_section(self,
                                                   run_a_correlations: Dict[str, float],
                                                   run_b_correlations: Dict[str, float],
                                                   objectives: List[str]) -> str:
        """Side-by-side correlation tables — dynamic decision rows and objective columns."""
        dec_cols = self.config.decision_columns

        header_cells = "<th>Decision Variable</th>" + "".join(
            f"<th>vs {to_axis_label(obj)}</th>" for obj in objectives
        )

        def generate_correlations_table(correlations: Dict[str, float]) -> str:
            rows_html = ""
            for dec in dec_cols:
                cells = f"<td>{dec}</td>"
                for obj in objectives:
                    key = f"{dec}_vs_{obj}"
                    cells += f"<td>{correlations.get(key, 0.0):.3f}</td>"
                rows_html += f"<tr>{cells}</tr>"
            return rows_html

        return f"""
        <div class="section">
            <h2>Distance Correlation Comparison</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-a-header">Run A Correlations</div>
                    <div class="wide-table">
                    <table>
                        <thead><tr>{header_cells}</tr></thead>
                        <tbody>{generate_correlations_table(run_a_correlations)}</tbody>
                    </table>
                    </div>
                </div>
                <div class="run-column">
                    <div class="run-b-header">Run B Correlations</div>
                    <div class="wide-table">
                    <table>
                        <thead><tr>{header_cells}</tr></thead>
                        <tbody>{generate_correlations_table(run_b_correlations)}</tbody>
                    </table>
                    </div>
                </div>
            </div>
        </div>"""

    def _generate_insights_section(self,
                                   run_a_stats: Dict[str, Any],
                                   run_b_stats: Dict[str, Any],
                                   run_a_pareto: List[Dict[str, Any]],
                                   run_b_pareto: List[Dict[str, Any]],
                                   run_a_rules: List[Dict[str, Any]],
                                   run_b_rules: List[Dict[str, Any]],
                                   a_dominates_b: int,
                                   b_dominates_a: int,
                                   objectives: List[str]) -> str:
        """
        Engineering insights — per-objective winners derived dynamically,
        using config.minimize_for() to know the direction of each objective.
        """
        from api.config.objectives import to_field  # local import to avoid clutter at top

        # Coverage & rule counts (objective-agnostic)
        better_pareto_coverage = 'Run A' if len(run_a_pareto) > len(run_b_pareto) else 'Run B'
        better_rules = 'Run A' if len(run_a_rules) > len(run_b_rules) else 'Run B'
        primary_choice = (
            'Run A' if a_dominates_b > b_dominates_a
            else 'Run B' if b_dominates_a > a_dominates_b
            else 'Tie'
        )

        # Per-objective winners (avg + stability)
        per_obj_lines = []
        for obj_name in objectives:
            field = to_field(obj_name)
            minimize = self.config.minimize_for(field)

            a_avg = run_a_stats['per_objective'].get(obj_name, {}).get('avg', 0.0)
            b_avg = run_b_stats['per_objective'].get(obj_name, {}).get('avg', 0.0)
            a_std = run_a_stats['per_objective'].get(obj_name, {}).get('std', 0.0)
            b_std = run_b_stats['per_objective'].get(obj_name, {}).get('std', 0.0)
            label = run_a_stats['per_objective'].get(obj_name, {}).get('label', obj_name)

            if minimize:
                avg_winner = 'Run A' if a_avg < b_avg else ('Run B' if b_avg < a_avg else 'Tie')
                direction = 'lower'
            else:
                avg_winner = 'Run A' if a_avg > b_avg else ('Run B' if b_avg > a_avg else 'Tie')
                direction = 'higher'

            stab_winner = 'Run A' if a_std < b_std else ('Run B' if b_std < a_std else 'Tie')

            per_obj_lines.append(
                f"<li><strong>{label}:</strong> {avg_winner} achieves a better average "
                f"({direction} is better; A={a_avg:.2f}, B={b_avg:.2f}). "
                f"More stable: {stab_winner} (σ_A={a_std:.2f}, σ_B={b_std:.2f}).</li>"
            )

        obj_insights_html = "\n".join(per_obj_lines)

        coverage_count = (
            len(run_a_pareto) if better_pareto_coverage == 'Run A' else len(run_b_pareto)
        )
        rules_count = (
            len(run_a_rules) if better_rules == 'Run A' else len(run_b_rules)
        )

        return f"""
        <div class="section">
            <h2>Engineering Insights & Recommendations</h2>
            <div class="insight-box">
                <h3>Key Insights:</h3>
                <ul>
                    <li><strong>Design Space Coverage:</strong> {better_pareto_coverage} explores more of the Pareto-optimal design space ({coverage_count} points).</li>
                    <li><strong>Rule Quality:</strong> {better_rules} discovered more association rules ({rules_count} rules).</li>
                    <li><strong>Pareto Dominance:</strong> Run A dominates {a_dominates_b} of Run B's points; Run B dominates {b_dominates_a} of Run A's points.</li>
                </ul>
                <h3>Per-Objective Performance:</h3>
                <ul>
                    {obj_insights_html}
                </ul>
            </div>

            <div class="recommendation">
                <h3>Recommendations:</h3>
                <ol>
                    <li><strong>Primary Choice:</strong> {primary_choice} should be preferred based on Pareto dominance.</li>
                    <li><strong>Design Strategy:</strong> Focus on configurations that appear on the Pareto front of {primary_choice}.</li>
                    <li><strong>Future Optimizations:</strong> Use the parameter settings from {primary_choice} as a starting point.</li>
                    <li><strong>Validation:</strong> Consider re-running with the same objectives ({', '.join(objectives)}) to confirm these findings.</li>
                </ol>
            </div>
        </div>"""