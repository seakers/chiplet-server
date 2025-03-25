import random
from api.Evaluator.gaCascade import runGACascade

class DataGenerator:
    def __init__(self):
        self.data = []
        self.initial = True

    def generate_data(self):
        result = runGACascade(pop_size=5, n_gen=5)
        for row in result:
            self.data.append({"x": row[0], "y": row[1]})

    def get_data(self):
        return self.data
    
    def clear_data(self):
        self.data = []
    
    def add_random_data(self):
        self.data.append({"x": random.randint(0, 20), "y": random.randint(0, 10)})