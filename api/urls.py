from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import TaskViewSet, compute_sum
from .views import get_chart_data
from .views import get_chat_response

router = DefaultRouter()
router.register(r"tasks", TaskViewSet)

urlpatterns = [
    path("compute-sum/", compute_sum, name="compute_sum"),
    path("chart-data/", get_chart_data, name="chart_data"),
    path("chat-response/", get_chat_response, name="chat_response"),
] + router.urls