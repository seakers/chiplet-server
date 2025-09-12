from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import TaskViewSet, compute_sum
from .views import get_chart_data
from .views import get_chat_response
from .views import clear_chat
from .views import evaluate_point
from .views import evaluate_point_inputs
from .views import add_info
from .views import update_data
from .views import rule_mining
from .views import distance_correlation
from .views import rule_mining_insights
from .views import run_optimization
from .views import add_insights_context
from .views import distance_correlation_insights
from .views import generate_optimization_report
from .views import get_designs_by_constraint
from .views import add_custom_point
from .views import test_endpoint
from .views import list_backup_files
from .views import load_previous_run
from .views import get_previous_run_report
from .views import data_mining_followup
from .views import get_point_context
from .views import add_enhanced_insights_context
from .views import integrate_custom_point_to_ga
from .views import save_custom_point_to_dataset
from .views import generate_comparative_report
from .views import get_latest_run_directory

# Import new run management views
from .run_views import (
    list_runs, get_run_details, get_run_for_plotting, get_multiple_runs_for_plotting,
    compare_runs, search_runs, get_pareto_fronts, create_comparative_study,
    list_comparative_studies, migrate_existing_data, delete_run, get_run_statistics,
    export_run_to_zip, export_comparative_study_to_zip, export_multiple_runs_to_zip, cleanup_exports
)

router = DefaultRouter()
router.register(r"tasks", TaskViewSet)

urlpatterns = [
    path("compute-sum/", compute_sum, name="compute_sum"),
    path("chart-data/", get_chart_data, name="chart_data"),
    path("update-data/", update_data, name="update_data"),
    path("chat-response/", get_chat_response, name="chat_response"),
    path("clear-chat/", clear_chat, name="clear_chat"),
    path("evaluate-point/", evaluate_point, name="evaluate_point"),
    path("evaluate-point-inputs/", evaluate_point_inputs, name="evaluate_point_inputs"),
    path("add-info/", add_info, name="add_info"),
    path("rule-mining/", rule_mining, name="rule_mining"),
    path("rule-mining-insights/", rule_mining_insights, name="rule_mining_insights"),
    path("distance-correlation/", distance_correlation, name="distance_correlation"),
    path("distance-correlation-insights/", distance_correlation_insights, name="distance_correlation_insights"),
    path("data-mining-followup/", data_mining_followup, name="data_mining_followup"),
    path("add-insights-context/", add_insights_context, name="add_insights_context"),
    path("add-enhanced-insights-context/", add_enhanced_insights_context, name="add_enhanced_insights_context"),
    path("get-point-context/", get_point_context, name="get_point_context"),
    path("generate-optimization-report/", generate_optimization_report, name="generate_optimization_report"),
    path("generate-comparative-report/", generate_comparative_report, name="generate_comparative_report"),
    path("run-optimization/", run_optimization, name="run_optimization"),
    path("get-designs-by-constraint/", get_designs_by_constraint, name="get_designs_by_constraint"),
    path("add-custom-point/", add_custom_point, name="add_custom_point"),
    path("integrate-custom-point-to-ga/", integrate_custom_point_to_ga, name="integrate_custom_point_to_ga"),
    path("save-custom-point-to-dataset/", save_custom_point_to_dataset, name="save_custom_point_to_dataset"),
    path("test-endpoint/", test_endpoint, name="test_endpoint"),
    path("list-backup-files/", list_backup_files, name="list_backup_files"),
    path("load-previous-run/", load_previous_run, name="load_previous_run"),
    path("get-previous-run-report/", get_previous_run_report, name="get_previous_run_report"),
    path("get-latest-run-directory/", get_latest_run_directory, name="get_latest_run_directory"),
    
    # New run management endpoints
    path("runs/", list_runs, name="list_runs"),
    path("runs/statistics/", get_run_statistics, name="get_run_statistics"),
    path("runs/search/", search_runs, name="search_runs"),
    path("runs/plot/", get_multiple_runs_for_plotting, name="get_multiple_runs_for_plotting"),
    path("runs/compare/", compare_runs, name="compare_runs"),
    path("runs/pareto-fronts/", get_pareto_fronts, name="get_pareto_fronts"),
    path("runs/<str:run_id>/", get_run_details, name="get_run_details"),
    path("runs/<str:run_id>/plot/", get_run_for_plotting, name="get_run_for_plotting"),
    path("runs/<str:run_id>/delete/", delete_run, name="delete_run"),
    path("runs/<str:run_id>/export/", export_run_to_zip, name="export_run_to_zip"),
    
    # Export endpoints
    path("exports/comparative-study/", export_comparative_study_to_zip, name="export_comparative_study_to_zip"),
    path("exports/multiple-runs/", export_multiple_runs_to_zip, name="export_multiple_runs_to_zip"),
    path("exports/cleanup/", cleanup_exports, name="cleanup_exports"),
    
    # Comparative studies
    path("comparative-studies/", list_comparative_studies, name="list_comparative_studies"),
    path("comparative-studies/create/", create_comparative_study, name="create_comparative_study"),
    
    # Data migration
    path("migrate-data/", migrate_existing_data, name="migrate_existing_data"),
] + router.urls