import os
import time
import math
import torch
import pickle

from utils.save_state import SaveState

from model_lib.nn.linear import LinearLayer
from model_lib.nn.layer_norm import LayerNorm
from model_lib.nn.embedding_table import EmbeddingTable
from model_lib.nn.transformer_layer import TransformerLayer

class MyLLM:
    def __init__(self, llm_manager):
        self.llm_manager        = llm_manager
        self.model_dim          = llm_manager.model_dim
        self.num_layers         = llm_manager.num_layers
        self.ff_dim             = llm_manager.ff_dim
        self.head_dim           = llm_manager.head_dim
        self.num_heads          = llm_manager.num_heads
        self.kv_heads           = llm_manager.kv_heads
        self.gqa                = llm_manager.gqa
        self.max_seq_len        = llm_manager.max_seq_len
        self.rope_theta         = llm_manager.rope_theta
        self.rms_norm_eps       = llm_manager.rms_norm_eps
        self.vocab_size         = llm_manager.vocab_size
        self.act_fn             = llm_manager.act_fn
        self.attn_softcap       = llm_manager.attn_softcap
        self.q_pre_attn_scalar  = llm_manager.q_pre_attn_scalar
        self.dtype              = llm_manager.dtype

        
        # debugger
        self.save_state         = SaveState()

        # model layers 
        self.cos, self.sin      = self.init_rotary_embeddings()

        self.emb_table          = EmbeddingTable(self.vocab_size, self.model_dim, self.dtype)
        self.model_norm         = LayerNorm(self, -1, self.model_dim, self.rms_norm_eps, self.dtype)
        self.lm_head            = LinearLayer(self, -1, self.model_dim, self.vocab_size, self.dtype)
        self.all_layers         = []
        for layer_id in range(self.num_layers):
            self.all_layers.append(TransformerLayer(self, layer_id, self.model_dim, self.ff_dim, self.head_dim, self.num_heads, self.kv_heads, self.max_seq_len, self.cos, self.sin, self.rms_norm_eps, self.act_fn, self.attn_softcap, self.q_pre_attn_scalar, self.dtype))
        self.fused_wQKV         = True
    
    # clear KV$ and 
    def reset(self):
        for layer_id in range(self.num_layers):
            self.all_layers[layer_id].reset()

    def init_random_params(self, sim_num_layers=None):
        self.emb_table.init_random()
        self.model_norm.init_random()
        for i, layer in enumerate(self.all_layers):
            if sim_num_layers == i:
                break 
            layer.init_random()
    
        #if sim_num_layers == -1:
        self.lm_head.init_random()
    
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
        cos_cached = emb.cos().to(self.dtype) # correct - 2x max seq len
        sin_cached = emb.sin().to(self.dtype) # correct

        return cos_cached, sin_cached

    def forward(self, input_ids, max_new_tokens=16, valid_output_ids=None, sim_num_layers=None):
        if self.llm_manager.verbose > 0:
            print("Starting Model Generation for My-LLM Implementation")
            
        self.input_ids = input_ids

        seq_len = self.input_ids.size()[1]

        input_ids = self.input_ids
        generated_token_ids = []
        for i in range(max_new_tokens):
            if self.llm_manager.validate_pistil_intermediates:
                if i + 1 == max_new_tokens:
                    self.save_state.enable_save_state()
                
            t1 = time.perf_counter_ns()
            layer_input = self.emb_table.forward(input_ids)
            num_layers = self.num_layers if sim_num_layers == None else sim_num_layers

            for layer_id in range(num_layers):
                layer = self.all_layers[layer_id]
                layer_input = layer.forward(layer_input, layer_id)
            
            # only if simulating all layers
            if sim_num_layers == -1:
                layer_input = self.model_norm.forward(layer_input)
                output = self.lm_head.forward(layer_input)
                output = torch.argmax(output[:, -1, :]).unsqueeze(0)
            
                input_ids = output.unsqueeze(0)
                t2 = time.perf_counter_ns()

                print("Token Output %i - %0.3fs" % (i, (t2 - t1)/10**9))
                print("\tGenerated ID:\t%i" % output[0].item())
                generated_token_ids.append(output[0].item())

            if valid_output_ids != None:
                print("\tExpected ID:\t%i" % valid_output_ids[0][i].item())
                if output[0] != valid_output_ids[0][i]:
                    print("Warning: Token generated does not match expected token. ")
                print("\t")

            self.generated_token_ids = generated_token_ids
            
        if self.llm_manager.from_pretrained == True:
            generated_text = self.llm_manager.base_tokenizer.decode(torch.tensor(generated_token_ids), skip_special_tokens=True)
            print("My LLM Conversation:\n\tPrompt:\t\t%s\n\tGenerated Text:\t%s" % (self.llm_manager.prompt[0]["content"], generated_text))


    