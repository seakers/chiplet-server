import math
import torch

class LinearLayer:
    def __init__(self, my_model, layer_id, weight_rows, weight_cols, dtype):
        self.my_model           = my_model
        self.layer_id           = layer_id
        self.weight_rows        = weight_rows
        self.weight_cols        = weight_cols
        self.dtype              = dtype

        self.weights            = None
    
    def load_from_pretrained(self, params):
        self.weights            = params.to(self.dtype)

    def init_random(self):
        self.weights            = torch.rand(self.weight_cols, self.weight_rows, dtype=self.dtype)
    
    def size(self):
        return self.weights.size()
    
    def forward(self, layer_input):
        return torch.matmul(layer_input, self.weights.T)
