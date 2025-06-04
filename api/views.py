from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Task
from .serializers import TaskSerializer

import random
from api.Evaluator.generator import DataGenerator
from api.ChatBot.model import ChatBotModel
import dcor
import numpy as np

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
    print("get_chart_data called")
    pop_size = int(request.GET.get("pop_size", 0))
    n_gen = int(request.GET.get("n_gen", 0))
    trace = request.GET.get("trace", "")
    print(f"Params: pop_size={pop_size}, n_gen={n_gen}, trace={trace}")
    dataGenerator.generate_data(pop_size=pop_size, n_gen=n_gen, trace=trace)
    data = dataGenerator.get_data()
    print("Data after GA:", data)
    return Response({"data": data})

@api_view(["GET"])
def update_data(request):
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

@api_view(["GET"])
def evaluate_point_inputs(request):
    """
    Evaluate a point using the DataGenerator.
    """
    print("Made it to the evaluate_point_inputs function")
    try:
        trace = request.GET.get("trace")

        chiplets = {
            "Attention": int(request.GET.get("Attention", 0)),
            "GPU": int(request.GET.get("GPU", 0)),
            "Sparse": int(request.GET.get("Sparse", 0)),
            "Convolution": int(request.GET.get("Convolution", 0)),
        }

        print("Chiplets:", chiplets)
        print("Trace:", trace)

        dataGenerator.evaluate_point(chiplets, trace)
        data = dataGenerator.get_data()
        return Response({"data": data})
    except Exception as e:
        return Response({"error": str(e)}, status=500)
    
@api_view(["GET"])
def add_info(request):
    """
    Add information to the chatbot.
    """
    # print("Made it to the add_info function")
    exe = request.GET.get("exe")
    energy = request.GET.get("energy")
    gpu = request.GET.get("gpu")
    attn = request.GET.get("attn")
    sparse = request.GET.get("sparse")
    conv = request.GET.get("conv")
    content = f"The design is evaluated to have an energy of {energy} and an execution time of {exe}.\n"
    content += f"The design has {gpu} GPU chiplets, {attn} attention chiplets, {sparse} sparse chiplets, and {conv} convolution chiplets.\n"
    chiplet_file_path = f"api/Evaluator/cascade/chiplet_model/dse/results/pointContext/{gpu}gpu{attn}attn{sparse}sparse{conv}conv.json"
    # try:
    #     with open(chiplet_file_path, "r") as file:
    #         content += file.read()
    # except FileNotFoundError:
    #     return Response({"error": "Chiplet file not found."}, status=404)
    chat_bot.add_information(chiplet_file_path)
    # chat_bot.add_information("I have received context on this design! I am ready to answer questions about it.", "assistant")
    chat_bot.messages.append(
        {
            "role": "assistant",
            "content": "I have received context on this design! I am ready to answer questions about it."
        }
    )
    return Response({"message": "Information added successfully."})

@api_view(["GET"])
def rule_mining(request):
    """
    Run rule mining and return the results in a structured format for the frontend table.
    """
    import time
    print("[rule_mining] Called rule_mining endpoint.")
    start_time = time.time()
    try:
        rule_mining_str = chat_bot.rule_mining()
        print("[rule_mining] Finished rule_mining() call.")
        # Parse the rule_mining_str into a list of dicts for the frontend
        import re
        rules = []
        rule_pattern = re.compile(r"Rule: (.*?), conf\\(f->p\\): ([0-9.eE+-]+), conf\\(p->f\\): ([0-9.eE+-]+), lift: ([0-9.eE+-]+)")
        for match in rule_pattern.finditer(rule_mining_str):
            rules.append({
                "rule": match.group(1),
                "conf_p_to_f": float(match.group(2)),
                "conf_f_to_p": float(match.group(3)),
                "lift": float(match.group(4)),
            })
        elapsed = time.time() - start_time
        print(f"[rule_mining] Returning {len(rules)} rules. Time taken: {elapsed:.2f} seconds.")
        return Response({"rules": rules, "elapsed": elapsed})
    except Exception as e:
        print(f"[rule_mining] Exception: {e}")
        return Response({"error": str(e)}, status=500)

@api_view(["GET"])
def distance_correlation(request):
    """
    Compute distance correlation between each chiplet type and Energy (y column) and Time (x column).
    """
    try:
        # Load data from CSV
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        import csv
        xs, ys, gpus, attns, sparses, convs = [], [], [], [], [], []
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                xs.append(float(row[0]))
                ys.append(float(row[1]))
                gpus.append(float(row[2]))
                attns.append(float(row[3]))
                sparses.append(float(row[4]))
                convs.append(float(row[5]))
        # Compute distance correlation for each chiplet type vs Energy and vs Time
        result = {
            "GPU_vs_Energy": float(dcor.distance_correlation(np.array(gpus), np.array(ys))),
            "Attention_vs_Energy": float(dcor.distance_correlation(np.array(attns), np.array(ys))),
            "Sparse_vs_Energy": float(dcor.distance_correlation(np.array(sparses), np.array(ys))),
            "Convolution_vs_Energy": float(dcor.distance_correlation(np.array(convs), np.array(ys))),
            "GPU_vs_Time": float(dcor.distance_correlation(np.array(gpus), np.array(xs))),
            "Attention_vs_Time": float(dcor.distance_correlation(np.array(attns), np.array(xs))),
            "Sparse_vs_Time": float(dcor.distance_correlation(np.array(sparses), np.array(xs))),
            "Convolution_vs_Time": float(dcor.distance_correlation(np.array(convs), np.array(xs))),
        }
        return Response(result)
    except Exception as e:
        return Response({"error": str(e)}, status=500)