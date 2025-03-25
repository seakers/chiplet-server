import random
from api.Evaluator.gaCascade import runGACascade

class DataGenerator:
    def __init__(self):
        self.data = []
        self.initial = True

    def generate_data(self, pop_size=0, n_gen=0):
        result = []
        if not pop_size == 0 and not n_gen == 0:
            result = runGACascade(pop_size=pop_size, n_gen=n_gen)
        for row in result:
            self.data.append({"x": row[0], "y": row[1]})

    def get_data(self):
        return self.data
    
    def clear_data(self):
        self.data = []
    
    def add_random_data(self):
        self.data.append({"x": random.randint(0, 20), "y": random.randint(0, 10)})