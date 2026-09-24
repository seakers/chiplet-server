import math
import torch


class LayerNorm:
    def __init__(self, parent, layer_id, model_dim, rms_norm_eps, dtype):
        self.parent             = parent
        self.layer_id           = layer_id
        self.model_dim          = model_dim
        self.rms_norm_eps       = rms_norm_eps
        self.dtype              = dtype

        self.weights            = None
        self.shards             = None
        
    
    def load_from_pretrained(self, params):
        self.weights            = params

    def init_random(self):
        self.weights            = torch.rand(self.weights).to(self.dtype)
    
    def size(self):
        return self.weights.size()
    
    def forward_weights(self, layer_input):
        hidden_shard        = (self.weights.to(torch.float32) * layer_input.to(torch.float32)).to(self.dtype) 
        variance_shard      = layer_input.to(torch.float32).pow(2).mean(-1, keepdim=True) * layer_input.size()[-1]
        return variance_shard, hidden_shard

    def forward_var(self, shard, var_sum):
        hidden_states       = shard.to(torch.float32)
        shard               = (hidden_states * torch.rsqrt(var_sum / self.model_dim + self.rms_norm_eps)).to(self.dtype)
        return shard


