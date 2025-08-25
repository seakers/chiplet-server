# api/serializers.py
from rest_framework import serializers
from .models import OptimizationRun, DesignPoint, AnalyticsResult, ComparativeStudy, Task

class TaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Task
        fields = '__all__'

class DesignPointSerializer(serializers.ModelSerializer):
    total_chiplets = serializers.ReadOnlyField()
    
    class Meta:
        model = DesignPoint
        fields = '__all__'

class AnalyticsResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalyticsResult
        fields = '__all__'

class OptimizationRunSerializer(serializers.ModelSerializer):
    design_points = DesignPointSerializer(many=True, read_only=True)
    analytics_results = AnalyticsResultSerializer(many=True, read_only=True)
    run_label = serializers.ReadOnlyField(source='get_run_label')
    
    class Meta:
        model = OptimizationRun
        fields = '__all__'

class OptimizationRunListSerializer(serializers.ModelSerializer):
    """Simplified serializer for listing runs"""
    run_label = serializers.ReadOnlyField(source='get_run_label')
    pareto_front_size = serializers.ReadOnlyField()
    
    class Meta:
        model = OptimizationRun
        fields = [
            'id', 'run_id', 'name', 'description', 'algorithm', 'model',
            'population_size', 'generations', 'trace_name', 'status',
            'created_at', 'completed_at', 'total_designs_evaluated',
            'pareto_front_size', 'run_label'
        ]

class ComparativeStudySerializer(serializers.ModelSerializer):
    run_a = OptimizationRunListSerializer(read_only=True)
    run_b = OptimizationRunListSerializer(read_only=True)
    
    class Meta:
        model = ComparativeStudy
        fields = '__all__'

class PlotDataSerializer(serializers.Serializer):
    """Serializer for plot data"""
    run_id = serializers.CharField()
    run_name = serializers.CharField()
    design_points = DesignPointSerializer(many=True)
    pareto_front = DesignPointSerializer(many=True)
    
class RunComparisonSerializer(serializers.Serializer):
    """Serializer for comparing multiple runs"""
    runs = OptimizationRunListSerializer(many=True)
    comparison_data = serializers.DictField()
    pareto_fronts = serializers.DictField()
