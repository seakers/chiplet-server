import math
import torch

class RotaryEmbeddings:
    def __init__(self, my_model, layer_id, cos, sin, kv_cache):
        self.my_model           = my_model
        self.layer_id           = layer_id
        self.cos                = cos
        self.sin                = sin
        self.kv_cache           = kv_cache
    
    def forward(self, seq_len, q_vector, k_vector):
        # rotary embeddings
        # apply rotary embeddings
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # rotary embeddings
        position_ids    = torch.arange(self.kv_cache.cached_ids, self.kv_cache.cached_ids + seq_len).unsqueeze(0)
        cos, sin        = self.cos[position_ids], self.sin[position_ids] 
        cos, sin        = cos.unsqueeze(1), sin.unsqueeze(1)

        q_vector        = (q_vector * cos) + (rotate_half(q_vector) * sin)
        k_vector        = (k_vector * cos) + (rotate_half(k_vector) * sin)    
        return q_vector, k_vector