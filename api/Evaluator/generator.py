import random
from api.Evaluator.gaCascade import runGACascade, runSingleCascade
import csv
import os
import numpy as np
from typing import List, Dict, Any

class DataGenerator:
    def __init__(self):
        self.data = []
        self.initial = True

    def generate_data(self, pop_size=0, n_gen=0, trace=""):
        print(f"generate_data called with pop_size={pop_size}, n_gen={n_gen}, trace={trace}")
        result = []
        if not pop_size == 0 and not n_gen == 0:
            result = runGACascade(pop_size=pop_size, n_gen=n_gen, trace=trace)
        print("GA result:", result)
        for row in result:
            self.data.append({"x": row[0], "y": row[1]})

    def get_data(self):
        file_path = "api/Evaluator/cascade/chiplet_model/dse/results/points.csv"
        with open(file_path, mode='r') as file:
            csv_reader = csv.reader(file)
            csvData = [
            {
                "x": float(row[0]),
                "y": float(row[1]),
                "gpu": float(row[2]),
                "attn": float(row[3]),
                "sparse": float(row[4]),
                "conv": float(row[5])
            }
            for row in csv_reader
            ]
        # print("CSV Data: ", csvData)
        return csvData
        # return self.data
    
    def clear_data(self):
        self.data = []
    
    def evaluate_point(self, chiplets, trace):
        objectives = runSingleCascade(chiplets, trace)
        self.data.append({"x": objectives[0], "y": objectives[1]})

def generate_weighted_trace(traces, weights=None, label=None):
    """
    Simulate loading each trace, normalizing, and creating a weighted composite trace.
    In real use, load each trace's data from CSV/model, normalize, and combine.
    Optionally use 'label' to name the output or for logging.
    """
    # Dummy: each trace has metrics: latency, energy, slot_dist (list of 4)
    dummy_traces = {
        "gpt-j-65536-weighted": {"latency": 100, "energy": 200, "slot_dist": [0.4, 0.3, 0.2, 0.1]},
        "gpt-j-1024-weighted": {"latency": 120, "energy": 180, "slot_dist": [0.3, 0.3, 0.2, 0.2]},
        "sd-test": {"latency": 90, "energy": 210, "slot_dist": [0.2, 0.4, 0.2, 0.2]},
        "ogbn-products-test": {"latency": 110, "energy": 190, "slot_dist": [0.25, 0.25, 0.25, 0.25]},
        "resnet50-test": {"latency": 130, "energy": 170, "slot_dist": [0.1, 0.2, 0.3, 0.4]},
    }
    # Weighted average
    if weights is None:
        weights = [t.get('weight', 1.0) for t in traces]
    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = 1.0
    composite = {"latency": 0, "energy": 0, "slot_dist": np.zeros(4)}
    for t, w in zip(traces, weights):
        name = t["name"] if isinstance(t, dict) else t
        weight = w / total_weight
        trace_data = dummy_traces.get(name, {"latency": 100, "energy": 200, "slot_dist": [0.25, 0.25, 0.25, 0.25]})
        composite["latency"] += trace_data["latency"] * weight
        composite["energy"] += trace_data["energy"] * weight
        composite["slot_dist"] += np.array(trace_data["slot_dist"]) * weight
    composite["slot_dist"] = composite["slot_dist"].tolist()
    composite["type"] = "composite"
    composite["components"] = traces
    if label:
        composite["label"] = label
    return composite