from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Task
from .serializers import TaskSerializer

import random
from api.Evaluator.generator import DataGenerator
from api.ChatBot.model import ChatBotModel

################ Instantiate the necessary objects ################
dataGenerator = DataGenerator()
chat_bot = ChatBotModel()
################ Instantiate the necessary objects ################

class TaskViewSet(viewsets.ModelViewSet):
    queryset = Task.objects.all()
    serializer_class = TaskSerializer

@api_view(["GET"])
def compute_sum(request):
    try:
        num1 = int(request.GET.get("num1", 0))
        num2 = int(request.GET.get("num2", 0))
        result = num1 + num2
        return Response({"result": result})
    except (ValueError, TypeError):
        return Response({"error": "Invalid numbers provided."}, status=400)
    
@api_view(["GET"])
def get_chart_data(request):
    """
    Generate and return chart data.
    """
    # Send null data first time
    if dataGenerator.initial:
        data = []
        dataGenerator.initial = False
        return Response({"data": data})
    
    pop_size = int(request.GET.get("pop_size", 0))
    n_gen = int(request.GET.get("n_gen", 0))
    trace = request.GET.get("trace", "")
    dataGenerator.generate_data(pop_size=pop_size, n_gen=n_gen, trace=trace)

    # Simulate dynamic data
    data = dataGenerator.get_data()
    print("Data Points: ", data)
    return Response({"data": data})

@api_view(["GET"])
def get_chat_response(request):
    """
    Get response from chatbot.
    """
    content = request.GET.get("content")
    role = request.GET.get("role")
    response = chat_bot.get_response(content, role)
    return Response({"response": response})

@api_view(["POST"])
def clear_chat(request):
    """
    Clear the chat history in the chatbot.
    """
    chat_bot.clear_history()
    return Response({"message": "Chat history cleared successfully."})

@api_view(["GET"])
def evaluate_point(request):
    """
    Evaluate a point using the DataGenerator.
    """
    print("Made it to the evaluate_point function")
    try:
        chipletKeys = request.GET.getlist("chiplets[]")
        trace = request.GET.get("trace")
        print("Chiplets:", chipletKeys)
        print("Trace:", trace)
        chiplets = {}
        for key in chipletKeys:
            chiplets[key] = chiplets.get(key, 0) + 1
                 
        dataGenerator.evaluate_point(chiplets, trace)
        data = dataGenerator.get_data()
        return Response({"data": data})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
