from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Task
from .serializers import TaskSerializer
import random

class DataGenerator:
    def __init__(self):
        self.data = []

    def generate_data(self):
        self.data = [{"x": i, "y": random.randint(0, 10)} for i in range(0, 20)]

    def get_data(self):
        return self.data
    
    def clear_data(self):
        self.data = []
    
    def add_random_data(self):
        self.data.append({"x": random.randint(0, 20), "y": random.randint(0, 10)})
   
dataGenerator = DataGenerator()
dataGenerator.generate_data()

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer

@api_view(['GET'])
def compute_sum(request):
    try:
        num1 = int(request.GET.get('num1', 0))
        num2 = int(request.GET.get('num2', 0))
        result = num1 + num2
        return Response({"result": result})
    except (ValueError, TypeError):
        return Response({"error": "Invalid numbers provided."}, status=400)
    
@api_view(['GET'])
def get_chart_data(request):
    # Simulate dynamic data
    if len(dataGenerator.get_data()) > 20:
        dataGenerator.clear_data()
    dataGenerator.add_random_data()
    data = dataGenerator.get_data()
    return Response({"data": data})
