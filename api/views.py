from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Task
from .serializers import TaskSerializer
import random

import sys
sys.path.append('chiplet-model/dse')
from gaCascades import runGACascades

class DataGenerator:
    def __init__(self):
        self.data = []

    def generate_data(self):
        runGACascades(pop_size=5, n_gen=5)

    def get_data(self):
        try:
            with open('chiplet-model/dse/results/points.txt', 'r') as file:
                self.data = []
                for line in file:
                    x, y = map(float, line.strip().split(','))
                    self.data.append({"x": x, "y": y})
        except FileNotFoundError:
            self.data = []
    
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
    data = dataGenerator.get_data()
    print(data)
    return Response({"data": data})
