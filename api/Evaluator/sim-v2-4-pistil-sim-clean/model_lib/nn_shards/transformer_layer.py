import math
import torch

from model_lib.nn_shards.linear import LinearLayer
from model_lib.nn_shards.layer_norm import LayerNorm
from model_lib.nn_shards.rotary_emb import RotaryEmbeddings
from model_lib.nn_shards.kv_cache import KVCache

class TransformerLayer:
    def __init__(self, parent, layer_id, model_dim, ff_dim, head_dim, num_heads, kv_heads, max_seq_len, cos, sin, rms_norm_eps, act_fn, attn_softcap, q_pre_attn_scalar, dtype):
        self.parent             = parent
        self.layer_id           = layer_id
        self.batch_size         = 1
        self.model_dim          = model_dim 
        self.ff_dim             = ff_dim      
        self.head_dim           = head_dim    
        self.num_heads          = num_heads   
        self.kv_heads           = kv_heads
        self.gqa                = int(num_heads / kv_heads)  
        self.max_seq_len        = max_seq_len  
        self.cos                = cos 
        self.sin                = sin  
        self.rms_norm_eps       = rms_norm_eps
        self.act_fn             = act_fn
        self.attn_softcap       = attn_softcap
        self.q_pre_attn_scalar  = q_pre_attn_scalar
        self.input_dtype        = dtype

        self.wQ                 = LinearLayer(self.parent, self.layer_id, self.model_dim, self.head_dim * self.num_heads, self.input_dtype)
        self.wK                 = LinearLayer(self.parent, self.layer_id, self.model_dim, self.head_dim * self.kv_heads, self.input_dtype)
        self.wV                 = LinearLayer(self.parent, self.layer_id, self.model_dim, self.head_dim * self.kv_heads, self.input_dtype)
        self.wQKV               = LinearLayer(self.parent, self.layer_id, self.model_dim, self.head_dim * self.num_heads + 2 * self.head_dim * self.kv_heads, self.input_dtype)
        self.wO                 = LinearLayer(self.parent, self.layer_id, self.head_dim * self.num_heads, self.model_dim, self.input_dtype)
        self.gate_proj          = LinearLayer(self.parent, self.layer_id, self.model_dim, self.ff_dim, self.input_dtype)
        self.up_proj            = LinearLayer(self.parent, self.layer_id, self.model_dim, self.ff_dim, self.input_dtype)
        self.gate_up_proj       = LinearLayer(self.parent, self.layer_id, self.model_dim, 2*self.ff_dim, self.input_dtype)
        self.down_proj          = LinearLayer(self.parent, self.layer_id, self.ff_dim, self.model_dim, self.input_dtype)
        self.input_norm         = LayerNorm(self.parent, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.post_atten_norm    = LayerNorm(self.parent, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.pre_ff_norm        = LayerNorm(self.parent, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.post_ff_norm       = LayerNorm(self.parent, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.kv_cache           = KVCache(self, self.layer_id, self.batch_size, self.gqa, self.head_dim, self.q_pre_attn_scalar, self.input_dtype)
        self.rotary_emb         = RotaryEmbeddings(self, self.layer_id, self.cos, self.sin, self.kv_cache)

    def reset(self):
        self.kv_cache.reset()
