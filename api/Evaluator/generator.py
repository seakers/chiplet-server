import random
from api.Evaluator.gaCascade import runGACascade, runSingleCascade
import csv

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