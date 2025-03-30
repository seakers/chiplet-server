import random
from api.Evaluator.gaCascade import runGACascade, runSingleCascade

class DataGenerator:
    def __init__(self):
        self.data = []
        self.initial = True

    def generate_data(self, pop_size=0, n_gen=0, trace=""):
        result = []
        if not pop_size == 0 and not n_gen == 0:
            result = runGACascade(pop_size=pop_size, n_gen=n_gen, trace=trace)
        for row in result:
            self.data.append({"x": row[0], "y": row[1]})

    def get_data(self):
        return self.data
    
    def clear_data(self):
        self.data = []
    
    def evaluate_point(self, chiplets, trace):
        objectives = runSingleCascade(chiplets, trace)
        self.data.append({"x": objectives[0], "y": objectives[1]})