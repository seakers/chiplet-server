import math
import torch

       
class EmbeddingTable:
    def __init__(self, parent, vocab_size, model_dim, dtype):
        self.parent             = parent
        self.vocab_size         = vocab_size
        self.model_dim          = model_dim
        self.dtype              = dtype
        
        self.weights            = None
        self.shards             = None


    def init_random(self):  
        self.weights            = torch.rand(self.vocab_size, self.model_dim).to(self.dtype)

    def load_from_pretrained(self, params):
        self.weights            = params
    
    def forward(self, layer_input):
        return self.weights[layer_input]

