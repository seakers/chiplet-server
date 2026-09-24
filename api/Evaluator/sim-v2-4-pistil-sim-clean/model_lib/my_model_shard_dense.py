import os
import time
import math
import torch
import pickle

from model_lib.nn_shards.linear import LinearLayer
from model_lib.nn_shards.layer_norm import LayerNorm
from model_lib.nn_shards.embedding_table import EmbeddingTable
from model_lib.nn_shards.transformer_layer import TransformerLayer


class MyLLMShard:
    def __init__(self, parent, model_dim, num_layers, ff_dim, head_dim, num_heads, kv_heads, max_seq_len, rope_theta, rms_norm_eps, vocab_size, act_fn, attn_softcap, q_pre_attn_scalar, dtype):
        self.parent             = parent
        self.model_dim          = model_dim
        self.num_layers         = num_layers
        self.ff_dim             = ff_dim
        self.head_dim           = head_dim
        self.num_heads          = num_heads
        self.kv_heads           = kv_heads
        self.gqa                = int(num_heads / kv_heads)
        self.max_seq_len        = max_seq_len
        self.rope_theta         = rope_theta
        self.rms_norm_eps       = rms_norm_eps
        self.act_fn             = act_fn
        self.attn_softcap       = attn_softcap
        self.q_pre_attn_scalar  = q_pre_attn_scalar
        self.vocab_size         = vocab_size

        self.dtype              = dtype

        # model layers 
        self.emb_table          = EmbeddingTable(self, self.vocab_size, self.model_dim, self.dtype)
        self.model_norm         = LayerNorm(self, -1, self.model_dim, self.rms_norm_eps, self.dtype)
        self.lm_head            = LinearLayer(self, -1, self.vocab_size, self.model_dim, self.dtype) # LMHead(self, self.vocab_size, self.model_dim, self.dtype)
        self.cos, self.sin      = self.init_rotary_embeddings()
        self.all_layers         = []
        for layer_id in range(self.num_layers):
            self.all_layers.append(TransformerLayer(self, layer_id, self.model_dim, self.ff_dim, self.head_dim, self.num_heads, self.kv_heads, self.max_seq_len, self.cos, self.sin, self.rms_norm_eps, self.act_fn, self.attn_softcap, self.q_pre_attn_scalar, self.dtype))
        self.fused_wQKV         = True
        
    
    # clear KV$ and 
    def reset(self):
        for layer_id in range(self.num_layers):
            self.all_layers[layer_id].reset()

    def init_random_params(self):
        self.emb_table.init_random()
        self.model_norm.init_random()
        self.lm_head.init_random()
        for layer in self.all_layers:
            layer.init_random()
    
    def init_rotary_embeddings(self):
        theta               = self.rope_theta
        head_dim            = self.head_dim
        scaling_factor      = 1.0
        max_seq_len_2x      = 2 * self.max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float()/head_dim)) # correct
        t = torch.arange(max_seq_len_2x, dtype=torch.int64).type_as(inv_freq)
        t = t / scaling_factor
        freqs = torch.outer(t, inv_freq)
        
        # from modeling lamma src - https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(torch.bfloat16) # correct - 2x max seq len
        sin_cached = emb.sin().to(torch.bfloat16) # correct

        return cos_cached, sin_cached
