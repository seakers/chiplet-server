# api/models.py
from django.db import models
import json

class OptimizationRun(models.Model):
    """Stores metadata for each optimization run"""
    ALGORITHM_CHOICES = [
        ('GA', 'Genetic Algorithm'),
        ('NSGA2', 'NSGA-II'),
        ('MOEA', 'Multi-Objective Evolutionary Algorithm'),
    ]
    
    MODEL_CHOICES = [
        ('CASCADE', 'Cascade Model'),
        ('CUSTOM', 'Custom Model'),
    ]
    
    # Run identification
    run_id = models.CharField(max_length=50, unique=True, help_text="Unique identifier for the run")
    name = models.CharField(max_length=255, blank=True, help_text="Human-readable name for the run")
    description = models.TextField(blank=True, help_text="Description of the optimization run")
    
    # Run parameters
    algorithm = models.CharField(max_length=20, choices=ALGORITHM_CHOICES, default='GA')
    model = models.CharField(max_length=20, choices=MODEL_CHOICES, default='CASCADE')
    population_size = models.IntegerField()
    generations = models.IntegerField()
    objectives = models.JSONField(default=list, help_text="List of objective names")
    
    # Trace information
    trace_name = models.CharField(max_length=255, blank=True)
    trace_sets = models.JSONField(default=dict, help_text="Trace sets configuration for comparative studies")
    
    # Run metadata
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, default='running', choices=[
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
        ('cancelled', 'Cancelled'),
    ])
    
    # Results summary
    total_designs_evaluated = models.IntegerField(default=0)
    pareto_front_size = models.IntegerField(default=0)
    execution_time_seconds = models.FloatField(null=True, blank=True)
    
    # File paths for detailed data
    results_directory = models.CharField(max_length=500, blank=True)
    analytics_directory = models.CharField(max_length=500, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.run_id}: {self.name or 'Unnamed Run'}"
    
    def get_run_label(self):
        """Get the run label (A, B, etc.) for comparative studies"""
        if self.trace_sets:
            # For comparative studies, determine label from trace_sets
            if 'A' in self.trace_sets:
                return 'A'
            elif 'B' in self.trace_sets:
                return 'B'
        return 'single'

class DesignPoint(models.Model):
    """Stores individual design points from optimization runs"""
    run = models.ForeignKey(OptimizationRun, on_delete=models.CASCADE, related_name='design_points')
    
    # Design configuration
    gpu_count = models.IntegerField()
    attention_count = models.IntegerField()
    sparse_count = models.IntegerField()
    convolution_count = models.IntegerField()
    
    # Objective values
    execution_time_ms = models.FloatField()
    energy_mj = models.FloatField()
    
    # Additional metrics (stored as JSON for flexibility)
    additional_metrics = models.JSONField(default=dict, blank=True)
    
    # Pareto front information
    is_pareto_optimal = models.BooleanField(default=False)
    pareto_rank = models.IntegerField(null=True, blank=True)
    
    # Evaluation metadata
    evaluated_at = models.DateTimeField(auto_now_add=True)
    evaluation_time_seconds = models.FloatField(null=True, blank=True)
    
    # File reference for detailed context
    context_file_path = models.CharField(max_length=500, blank=True)
    
    class Meta:
        unique_together = ['run', 'gpu_count', 'attention_count', 'sparse_count', 'convolution_count']
        ordering = ['run', 'execution_time_ms']
    
    def __str__(self):
        return f"Run {self.run.run_id}: {self.gpu_count}GPU-{self.attention_count}ATTN-{self.sparse_count}SPARSE-{self.convolution_count}CONV"
    
    @property
    def total_chiplets(self):
        return self.gpu_count + self.attention_count + self.sparse_count + self.convolution_count

class AnalyticsResult(models.Model):
    """Stores analytics results for optimization runs"""
    ANALYTICS_TYPES = [
        ('rule_mining', 'Association Rule Mining'),
        ('distance_correlation', 'Distance Correlation'),
        ('pareto_analysis', 'Pareto Front Analysis'),
        ('sensitivity', 'Sensitivity Analysis'),
    ]
    
    run = models.ForeignKey(OptimizationRun, on_delete=models.CASCADE, related_name='analytics_results')
    analytics_type = models.CharField(max_length=30, choices=ANALYTICS_TYPES)
    
    # Results data
    results_data = models.JSONField(default=dict)
    
    # File reference
    file_path = models.CharField(max_length=500, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    computation_time_seconds = models.FloatField(null=True, blank=True)
    
    class Meta:
        unique_together = ['run', 'analytics_type']
        ordering = ['run', 'analytics_type']
    
    def __str__(self):
        return f"{self.run.run_id} - {self.get_analytics_type_display()}"

class ComparativeStudy(models.Model):
    """Stores comparative study metadata"""
    study_id = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255, blank=True)
    description = models.TextField(blank=True)
    
    # Associated runs
    run_a = models.ForeignKey(OptimizationRun, on_delete=models.CASCADE, related_name='comparative_studies_a', null=True, blank=True)
    run_b = models.ForeignKey(OptimizationRun, on_delete=models.CASCADE, related_name='comparative_studies_b', null=True, blank=True)
    
    # Study parameters
    shared_parameters = models.JSONField(default=dict)
    
    # Results summary
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Study {self.study_id}: {self.name or 'Unnamed Study'}"

# Legacy Task model for backward compatibility
class Task(models.Model):
    title = models.CharField(max_length=255)
    completed = models.BooleanField(default=False)

    def __str__(self):
        return self.title
