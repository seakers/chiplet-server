"""
Report Agent for generating optimization reports via the existing report endpoint.
"""
from typing import Dict, Any
from .base import BaseAgent, AgentResult
from .registry import AgentRegistry


@AgentRegistry.register
class ReportAgent(BaseAgent):
    """Generates an HTML report for the current (or specified) optimization run."""

    @property
    def name(self) -> str:
        return "report_agent"

    @property
    def description(self) -> str:
        return (
            "Generates a downloadable HTML report for an optimization run. "
            "Uses the current run_id by default. Includes Pareto front, "
            "rule mining, and distance correlation analysis."
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        try:
            from rest_framework.test import APIRequestFactory
            from api.views.reports import generate_report

            run_id = context.get('run_id') or self.run_id
            if not run_id:
                return AgentResult(
                    success=False,
                    message="No run_id available. Start or load an optimization run before generating a report.",
                    error="Missing run_id"
                )
            
            objectives = context.get('objectives')

            factory = APIRequestFactory()
            params = {'run_id': run_id, 'objectives[]': objectives}
            drf_req = factory.get('/api/generate-optimization-report/', params)
            response = generate_report(drf_req)
            import json
            # Django JsonResponse doesn't expose .data; parse JSON from content
            if hasattr(response, 'data'):
                body = response.data
            else:
                try:
                    body = json.loads(response.content.decode())
                except Exception:
                    body = {}

            if body.get('status') != 'success':
                return AgentResult(
                    success=False,
                    message=f"Report generation failed: {body.get('message', 'unknown error')}",
                    error=body.get('message')
                )

            web_link = body.get('web_link')
            return AgentResult(
                success=True,
                message=(
                    f"Report generated successfully. View it at "
                    f"http://localhost:8000{web_link}" if web_link else
                    "Report generated successfully."
                ),
                data={
                    'web_link': web_link,
                    'download_link': body.get('download_link'),
                    'report_content': body.get('report_content'),
                    'run_id': run_id,
                }
            )

        except Exception as e:
            import traceback; traceback.print_exc()
            return AgentResult(
                success=False,
                message=f"Error generating report: {e}",
                error=str(e)
            )

    def get_parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "objectives": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Friendly objective names to analyze."
                },
            },
            "required": []
        }