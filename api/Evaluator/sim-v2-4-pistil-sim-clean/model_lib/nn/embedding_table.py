import math
import torch 

class EmbeddingTable:
    def __init__(self, vocab_size, model_dim, dtype):
        self.vocab_size         = vocab_size
        self.model_dim          = model_dim
        self.dtype              = dtype
        
        self.weights            = None

    def init_random(self):  
        self.weights            = torch.rand(self.vocab_size, self.model_dim, dtype=self.dtype)

    def load_emb_table_from_pretrained(self, params):
        self.weights            = params.to(self.dtype)
    
    def forward(self, layer_input):
        return self.weights[layer_input]
