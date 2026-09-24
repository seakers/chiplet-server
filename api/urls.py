"""
URL configuration for the chiplet design API.
"""
from django.urls import path
from api.views import analysis, optimization, reports, chatbot, highlighting

urlpatterns = [

    # ── Health / legacy ───────────────────────────────────────────────────────
    path('test/',                           optimization.test_endpoint,             name='test_endpoint'),
    path('compute-sum/',                    optimization.compute_sum,               name='compute_sum'),
    path('update-data/',                    optimization.update_data,               name='update_data'),

    # ── Optimization ──────────────────────────────────────────────────────────
    path('run-optimization/',               optimization.run_optimization,          name='run_optimization'),
    path('chart-data/',                     optimization.get_chart_data,            name='get_chart_data'),
    path('check-run-status/',               optimization.check_run_status,          name='check_run_status'),
    path('evaluate-point/',                 optimization.evaluate_point,            name='evaluate_point'),
    path('evaluate-point-inputs/',          optimization.evaluate_point_inputs,     name='evaluate_point_inputs'),
    path('restart-run/',                    optimization.restart_run,               name='restart_run'),
    path('estimate-full-factorial/',        optimization.estimate_full_factorial,   name='estimate_full_factorial'),
    path('integrate-custom-point/',         optimization.integrate_custom_point_to_ga, name='integrate_custom_point'),

    # ── Analysis ──────────────────────────────────────────────────────────────
    path('rule-mining/',                    analysis.rule_mining,                   name='rule_mining'),
    path('rule-mining-insights/',           analysis.rule_mining_insights,          name='rule_mining_insights'),
    path('distance-correlation/',           analysis.distance_correlation,          name='distance_correlation'),
    path('distance-correlation-insights/',  analysis.distance_correlation_insights, name='distance_correlation_insights'),
    path('pdf/',                            analysis.serve_pdf,                     name='serve_pdf'),

    # ── Reports ───────────────────────────────────────────────────────────────
    path('generate-report/',                reports.generate_report,                name='generate_report'),
    path('generate-optimization-report/',   reports.generate_report,                name='generate_optimization_report'),
    path('generate-comparative-report/',    reports.generate_comparative_report,    name='generate_comparative_report'),
    path('get-previous-run-report/',        reports.get_previous_run_report,        name='get_previous_run_report'),
    path('list-backup-files/',              reports.list_backup_files,              name='list_backup_files'),
    path('load-previous-run/',              reports.load_previous_run,              name='load_previous_run'),

    # ── ChatBot ───────────────────────────────────────────────────────────────
    path('chat/',                           chatbot.chat,                           name='chat'),
    path('chat-response/',                  chatbot.get_chat_response,              name='get_chat_response'),
    path('add-info/',                       chatbot.add_info,                       name='add_info'),
    path('add-run-context/',                chatbot.add_run_context,                name='add_run_context'),
    path('add-point-context/',              chatbot.add_point_context,              name='add_point_context'),
    path('add-insights-context/',           chatbot.add_insights_context,           name='add_insights_context'),
    path('add-enhanced-insights-context/',  chatbot.add_enhanced_insights_context,  name='add_enhanced_insights_context'),
    path('data-mining-followup/',           chatbot.data_mining_followup,           name='data_mining_followup'),
    path('clear-chat/',                     chatbot.clear_chat_history,             name='clear_chat_history'),
    path('get-latest-run-directory/',       chatbot.get_latest_run_directory,       name='get_latest_run_directory'),

    # ── Highlighting / Point context ──────────────────────────────────────────
    path('get-designs-by-constraint/',      highlighting.get_designs_by_constraint, name='get_designs_by_constraint'),
    path('get-highlighted-points/',         highlighting.get_highlighted_points,    name='get_highlighted_points'),
    path('get-point-context/',              highlighting.get_point_context,         name='get_point_context'),
    path('get-kernel-breakdown/',           highlighting.get_kernel_breakdown,      name='get_kernel_breakdown'),
]