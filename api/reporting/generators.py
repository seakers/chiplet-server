"""
Report generation utilities.
Consolidates the HTML generation logic from views.py [1].
"""
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from api.analysis.rule_mining import RuleFormatter
from api.analysis.pareto import ParetoCalculator
from .formatters import ReportFormatter


class ReportGenerator:
    """
    Generates HTML reports for optimization runs.
    Consolidates the embedded HTML generation from views.py [1].
    """
    
    # Base CSS styles used across all reports
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
        .stat-number { font-size: 2em; font-weight: bold; color: #3498db; }
        .stat-label { color: #7f8c8d; margin-top: 5px; }
        .header-info { background: #ecf0f1; padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .header-info p { margin: 5px 0; color: #2c3e50; }
        .insight-box { background: #e8f6f3; padding: 15px; border-radius: 8px; border-left: 4px solid #1abc9c; margin: 10px 0; }
        .recommendation { background: #fef9e7; padding: 15px; border-radius: 8px; border-left: 4px solid #f39c12; margin: 10px 0; }
    """
    
    # Additional styles for comparative reports
    COMPARATIVE_STYLES = """
        .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .run-column { padding: 15px; border-radius: 8px; }
        .run-a-header { background-color: #3498db; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
        .run-b-header { background-color: #e74c3c; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
        .winner { background-color: #d5f4e6; border: 2px solid #27ae60; }
        .loser { background-color: #fce4e4; border: 2px solid #e74c3c; }
    """
    
    def __init__(self, evaluator: str = 'cascade'):
        self.evaluator = evaluator.lower()
        self.pareto_calculator = ParetoCalculator()
        self.rule_formatter = RuleFormatter()
    
    def generate_single_run_report(self,
                                    run_params: Dict[str, Any],
                                    points: List[Dict[str, Any]],
                                    pareto_points: List[Dict[str, Any]],
                                    rules: List[Dict[str, Any]],
                                    correlations: Dict[str, float],
                                    run_id: str = None) -> str:
        """
        Generate HTML report for a single optimization run.
        
        This consolidates the report generation logic from views.py [1].
        """
        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H:%M:%S")
        
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
            <p><strong>Report Type:</strong> Tradespace Exploration Run</p>
            {f'<p><strong>Run ID:</strong> {run_id}</p>' if run_id else ''}
        </div>

        {self._generate_problem_formulation_section(run_params)}
        {self._generate_results_summary_section(points, pareto_points)}
        {self._generate_pareto_front_section(pareto_points)}
        {self._generate_rule_mining_section(rules)}
        {self._generate_correlation_section(correlations)}
    </div>
</body>
</html>"""
        
        return html_content
    
    def generate_comparative_report(self,
                                     run_a_id: str,
                                     run_b_id: str,
                                     run_a_data: Dict[str, Any],
                                     run_b_data: Dict[str, Any]) -> str:
        """
        Generate HTML report comparing two optimization runs.
        
        This consolidates the comparative report generation from views.py [1].
        """
        now = datetime.now()
        date_str = now.strftime("%m/%d/%Y")
        time_str = now.strftime("%H:%M:%S")
        
        # Extract data from run dictionaries
        run_a_points = run_a_data.get('points', [])
        run_b_points = run_b_data.get('points', [])
        run_a_pareto = run_a_data.get('pareto_points', [])
        run_b_pareto = run_b_data.get('pareto_points', [])
        run_a_stats = run_a_data.get('stats', {})
        run_b_stats = run_b_data.get('stats', {})
        run_a_rules = run_a_data.get('rules', [])
        run_b_rules = run_b_data.get('rules', [])
        run_a_correlations = run_a_data.get('correlations', {})
        run_b_correlations = run_b_data.get('correlations', {})
        run_a_params = run_a_data.get('params', {})
        run_b_params = run_b_data.get('params', {})
        
        # Calculate Pareto dominance
        a_dominates_b = self.pareto_calculator.count_dominated_points(run_a_pareto, run_b_pareto)
        b_dominates_a = self.pareto_calculator.count_dominated_points(run_b_pareto, run_a_pareto)
        
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
            <p><strong>Run A:</strong> {run_a_id}</p>
            <p><strong>Run B:</strong> {run_b_id}</p>
        </div>

        {self._generate_comparative_summary_section(run_a_stats, run_b_stats, a_dominates_b, b_dominates_a)}
        {self._generate_comparative_params_section(run_a_params, run_b_params)}
        {self._generate_comparative_pareto_section(run_a_pareto, run_b_pareto)}
        {self._generate_comparative_rules_section(run_a_rules, run_b_rules)}
        {self._generate_comparative_correlations_section(run_a_correlations, run_b_correlations)}
        {self._generate_insights_section(run_a_stats, run_b_stats, run_a_pareto, run_b_pareto, 
                                         run_a_rules, run_b_rules, a_dominates_b, b_dominates_a)}
    </div>
</body>
</html>"""
        
        return html_content
    
    def _generate_problem_formulation_section(self, run_params: Dict[str, Any]) -> str:
        """Generate the problem formulation section."""
        objectives_html = ''.join([
            f'<li>Objective {i+1}: Minimize {obj.lower()}</li>'
            for i, obj in enumerate(run_params.get('objectives', ['Energy', 'Runtime']))
        ])
        
        return f"""
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
                {objectives_html}
            </ul>
            <p><strong>Models and parameters:</strong></p>
            <ul>
                <li>Model: {run_params.get('model', 'CASCADE')}</li>
                <li>Trace: {run_params.get('trace_name', 'gpt-j-65536-weighted')}</li>
                <li>Search Algorithm: {run_params.get('algorithm', 'Genetic Algorithm')}</li>
                <li>Population Size: {run_params.get('population_size', 50)}</li>
                <li>Generation Size: {run_params.get('generations', 100)}</li>
            </ul>
        </div>"""
    
    def _generate_results_summary_section(self, 
                                           points: List[Dict[str, Any]], 
                                           pareto_points: List[Dict[str, Any]]) -> str:
        """Generate the results summary section with stat cards."""
        return f"""
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
        </div>"""
    
    def _generate_pareto_front_section(self, pareto_points: List[Dict[str, Any]]) -> str:
        """Generate the Pareto front composition table."""
        rows_html = ""
        for point in pareto_points:
            rows_html += f"""
                    <tr>
                        <td>{int(point.get('gpu', 0))}</td>
                        <td>{int(point.get('attn', 0))}</td>
                        <td>{int(point.get('sparse', 0))}</td>
                        <td>{int(point.get('conv', 0))}</td>
                        <td>{point.get('x', 0):.2f}</td>
                        <td>{point.get('y', 0):.2f}</td>
                    </tr>"""
        
        return f"""
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
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>"""
    
    def _generate_rule_mining_section(self, rules: List[Dict[str, Any]]) -> str:
        """Generate the association rule mining results section."""
        rows_html = ""
        for rule in rules:
            formatted_rule = RuleFormatter.format_rule_natural_language(rule.get('rule', ''))
            rows_html += f"""
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
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>"""
    
    def _generate_correlation_section(self, correlations: Dict[str, float]) -> str:
        """Generate the distance correlation analysis section."""
        chiplet_types = ['GPU', 'Attention', 'Sparse', 'Convolution']
        rows_html = ""
        
        for chiplet in chiplet_types:
            energy_corr = correlations.get(f'{chiplet}_vs_Energy', 0)
            time_corr = correlations.get(f'{chiplet}_vs_Time', 0)
            rows_html += f"""
                    <tr>
                        <td>{chiplet}</td>
                        <td>{energy_corr:.3f}</td>
                        <td>{time_corr:.3f}</td>
                    </tr>"""
        
        return f"""
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
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>"""
    
    def _generate_comparative_summary_section(self,
                                               run_a_stats: Dict[str, Any],
                                               run_b_stats: Dict[str, Any],
                                               a_dominates_b: int,
                                               b_dominates_a: int) -> str:
        """Generate comparative summary section."""
        return f"""
        <div class="section">
            <h2>Comparative Summary</h2>
            <div class="comparison-grid">
                <div class="run-column {'winner' if a_dominates_b > b_dominates_a else 'loser' if b_dominates_a > a_dominates_b else ''}">
                    <div class="run-a-header">Run A</div>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">{run_a_stats.get('total_points', 0)}</div>
                            <div class="stat-label">Total Points</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_a_stats.get('pareto_count', 0)}</div>
                            <div class="stat-label">Pareto Points</div>
                        </div>
                    </div>
                    <p><strong>Avg Energy:</strong> {run_a_stats.get('avg_energy', 0):.2f} mJ</p>
                    <p><strong>Avg Time:</strong> {run_a_stats.get('avg_time', 0):.2f} ms</p>
                    <p><strong>Min Energy:</strong> {run_a_stats.get('min_energy', 0):.2f} mJ</p>
                    <p><strong>Min Time:</strong> {run_a_stats.get('min_time', 0):.2f} ms</p>
                    <p><strong>Dominates Run B:</strong> {a_dominates_b} points</p>
                </div>
                <div class="run-column {'winner' if b_dominates_a > a_dominates_b else 'loser' if a_dominates_b > b_dominates_a else ''}">
                    <div class="run-b-header">Run B</div>
                    <div class="stats">
                        <div class="stat-card">
                            <div class="stat-number">{run_b_stats.get('total_points', 0)}</div>
                            <div class="stat-label">Total Points</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{run_b_stats.get('pareto_count', 0)}</div>
                            <div class="stat-label">Pareto Points</div>
                        </div>
                    </div>
                    <p><strong>Avg Energy:</strong> {run_b_stats.get('avg_energy', 0):.2f} mJ</p>
                    <p><strong>Avg Time:</strong> {run_b_stats.get('avg_time', 0):.2f} ms</p>
                    <p><strong>Min Energy:</strong> {run_b_stats.get('min_energy', 0):.2f} mJ</p>
                    <p><strong>Min Time:</strong> {run_b_stats.get('min_time', 0):.2f} ms</p>
                    <p><strong>Dominates Run A:</strong> {b_dominates_a} points</p>
                </div>
            </div>
        </div>"""
    
    def _generate_comparative_params_section(self,
                                              run_a_params: Dict[str, Any],
                                              run_b_params: Dict[str, Any]) -> str:
        """Generate comparative parameters section."""
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
                <tbody>
                    <tr>
                        <td>Model</td>
                        <td>{run_a_params.get('model', 'N/A')}</td>
                        <td>{run_b_params.get('model', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Algorithm</td>
                        <td>{run_a_params.get('algorithm', 'N/A')}</td>
                        <td>{run_b_params.get('algorithm', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Trace</td>
                        <td>{run_a_params.get('trace_name', 'N/A')}</td>
                        <td>{run_b_params.get('trace_name', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Population Size</td>
                        <td>{run_a_params.get('population_size', 'N/A')}</td>
                        <td>{run_b_params.get('population_size', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Generations</td>
                        <td>{run_a_params.get('generations', 'N/A')}</td>
                        <td>{run_b_params.get('generations', 'N/A')}</td>
                    </tr>
                </tbody>
            </table>
        </div>"""
    
    def _generate_comparative_pareto_section(self,
                                              run_a_pareto: List[Dict[str, Any]],
                                              run_b_pareto: List[Dict[str, Any]]) -> str:
        """Generate comparative Pareto front section."""
        
        def generate_pareto_table(pareto_points: List[Dict[str, Any]], max_rows: int = 10) -> str:
            rows_html = ""
            for point in pareto_points[:max_rows]:
                rows_html += f"""
                    <tr>
                        <td>{int(point.get('gpu', 0))}</td>
                        <td>{int(point.get('attn', 0))}</td>
                        <td>{int(point.get('sparse', 0))}</td>
                        <td>{int(point.get('conv', 0))}</td>
                        <td>{point.get('x', 0):.2f}</td>
                        <td>{point.get('y', 0):.2f}</td>
                    </tr>"""
            if len(pareto_points) > max_rows:
                rows_html += f"""
                    <tr>
                        <td colspan="6"><em>... and {len(pareto_points) - max_rows} more points</em></td>
                    </tr>"""
            return rows_html
        
        return f"""
        <div class="section">
            <h2>Pareto Front Comparison</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-a-header">Run A Pareto Front ({len(run_a_pareto)} points)</div>
                    <table>
                        <thead>
                            <tr>
                                <th>GPU</th>
                                <th>Attn</th>
                                <th>Sparse</th>
                                <th>Conv</th>
                                <th>Time</th>
                                <th>Energy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_pareto_table(run_a_pareto)}
                        </tbody>
                    </table>
                </div>
                <div class="run-column">
                    <div class="run-b-header">Run B Pareto Front ({len(run_b_pareto)} points)</div>
                    <table>
                        <thead>
                            <tr>
                                <th>GPU</th>
                                <th>Attn</th>
                                <th>Sparse</th>
                                <th>Conv</th>
                                <th>Time</th>
                                <th>Energy</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_pareto_table(run_b_pareto)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>"""
    
    def _generate_comparative_rules_section(self,
                                             run_a_rules: List[Dict[str, Any]],
                                             run_b_rules: List[Dict[str, Any]]) -> str:
        """Generate comparative rule mining section."""
        
        def generate_rules_table(rules: List[Dict[str, Any]]) -> str:
            if not rules:
                return "<tr><td colspan='4'><em>No significant rules found</em></td></tr>"
            
            rows_html = ""
            for rule in rules[:5]:  # Limit to top 5 rules
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
                            <tr>
                                <th>Rule</th>
                                <th>F→P</th>
                                <th>P→F</th>
                                <th>Lift</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_rules_table(run_a_rules)}
                        </tbody>
                    </table>
                </div>
                <div class="run-column">
                    <div class="run-b-header">Run B Rules ({len(run_b_rules)} found)</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Rule</th>
                                <th>F→P</th>
                                <th>P→F</th>
                                <th>Lift</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_rules_table(run_b_rules)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>"""
    
    def _generate_comparative_correlations_section(self,
                                                    run_a_correlations: Dict[str, float],
                                                    run_b_correlations: Dict[str, float]) -> str:
        """Generate comparative distance correlations section."""
        chiplet_types = ['GPU', 'Attention', 'Sparse', 'Convolution']
        
        def generate_correlations_table(correlations: Dict[str, float]) -> str:
            rows_html = ""
            for chiplet in chiplet_types:
                energy_corr = correlations.get(f'{chiplet}_vs_Energy', 0)
                time_corr = correlations.get(f'{chiplet}_vs_Time', 0)
                rows_html += f"""
                    <tr>
                        <td>{chiplet}</td>
                        <td>{energy_corr:.3f}</td>
                        <td>{time_corr:.3f}</td>
                    </tr>"""
            return rows_html
        
        return f"""
        <div class="section">
            <h2>Distance Correlation Comparison</h2>
            <div class="comparison-grid">
                <div class="run-column">
                    <div class="run-a-header">Run A Correlations</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Chiplet Type</th>
                                <th>vs Energy</th>
                                <th>vs Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_correlations_table(run_a_correlations)}
                        </tbody>
                    </table>
                </div>
                <div class="run-column">
                    <div class="run-b-header">Run B Correlations</div>
                    <table>
                        <thead>
                            <tr>
                                <th>Chiplet Type</th>
                                <th>vs Energy</th>
                                <th>vs Time</th>
                            </tr>
                        </thead>
                        <tbody>
                            {generate_correlations_table(run_b_correlations)}
                        </tbody>
                    </table>
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
                                    b_dominates_a: int) -> str:
        """Generate engineering insights and recommendations section."""
        
        # Determine which run is better in various categories
        better_pareto_coverage = 'Run A' if len(run_a_pareto) > len(run_b_pareto) else 'Run B'
        better_stability = 'Run A' if run_a_stats.get('std_time', float('inf')) < run_b_stats.get('std_time', float('inf')) else 'Run B'
        better_energy = 'Run A' if run_a_stats.get('avg_energy', float('inf')) < run_b_stats.get('avg_energy', float('inf')) else 'Run B'
        better_rules = 'Run A' if len(run_a_rules) > len(run_b_rules) else 'Run B'
        primary_choice = 'Run A' if a_dominates_b > b_dominates_a else 'Run B'
        
        return f"""
        <div class="section">
            <h2>Engineering Insights & Recommendations</h2>
            <div class="insight-box">
                <h3>Key Insights:</h3>
                <ul>
                    <li><strong>Design Space Coverage:</strong> {better_pareto_coverage} explores more of the Pareto optimal design space ({len(run_a_pareto) if better_pareto_coverage == 'Run A' else len(run_b_pareto)} points)</li>
                    <li><strong>Performance Stability:</strong> {better_stability} shows more consistent execution times</li>
                    <li><strong>Energy Efficiency:</strong> {better_energy} achieves better average energy efficiency</li>
                    <li><strong>Rule Quality:</strong> {better_rules} discovered more association rules ({len(run_a_rules) if better_rules == 'Run A' else len(run_b_rules)} rules)</li>
                    <li><strong>Pareto Dominance:</strong> Run A dominates {a_dominates_b} of Run B's points; Run B dominates {b_dominates_a} of Run A's points</li>
                </ul>
            </div>
            
            <div class="recommendation">
                <h3>Recommendations:</h3>
                <ol>
                    <li><strong>Primary Choice:</strong> {primary_choice} should be preferred for its superior Pareto dominance</li>
                    <li><strong>Design Strategy:</strong> Focus on configurations that appear in the Pareto front of {primary_choice}</li>
                    <li><strong>Future Optimizations:</strong> Use the parameter settings from {primary_choice} as a starting point</li>
                    <li><strong>Validation:</strong> Consider running additional optimizations with similar parameters to validate findings</li>
                </ol>
            </div>
        </div>"""