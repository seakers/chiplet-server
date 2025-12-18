import math
import torch

class LayerNorm:
    def __init__(self, my_model, layer_id, model_dim, rms_norm_eps, dtype):
        self.my_model           = my_model
        self.layer_id           = layer_id
        self.model_dim          = model_dim
        self.rms_norm_eps       = rms_norm_eps
        self.dtype              = dtype

        self.weights            = None
    
    def load_from_pretrained(self, params):
        self.weights            = params.to(self.dtype)

    def init_random(self):
        self.weights            = torch.rand(self.model_dim, dtype=self.dtype)
    
    def size(self):
        return self.weights.size()
    
    def forward_weights(self, layer_input):
        hidden_states       = (self.weights.to(torch.float32) * layer_input.to(torch.float32)).to(self.dtype) 
        self.variance       = layer_input.to(torch.float32).pow(2).mean(-1, keepdim=True) * layer_input.size()[-1]
        return hidden_states

    def forward_var(self, layer_input):
        hidden_states       = layer_input.to(torch.float32)
        hidden_states       = (hidden_states * torch.rsqrt(self.variance / self.model_dim + self.rms_norm_eps)).to(self.dtype)
        return hidden_states

    # original implementation of layer norm
    def forward(self, layer_input):
        #var_sum = layer_input.to(torch.float32).pow(2).mean(-1, keepdim=True)*layer_input.size()[-1]
        hidden_states   = layer_input.to(torch.float32)                             # convert to FP32 for better normalization accuracy
        variance        = hidden_states.pow(2).mean(-1, keepdim=True)               # calculate variance
        hidden_states   = hidden_states * torch.rsqrt(variance + self.rms_norm_eps) # rescale - one number
        hidden_states   = (self.weights.float() * hidden_states).to(self.dtype)      # normalize using factors
        return hidden_states