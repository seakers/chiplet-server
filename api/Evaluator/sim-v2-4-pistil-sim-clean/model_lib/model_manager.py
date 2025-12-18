
import os
import time
import math
import pickle
import torch
import json 

from model_lib.my_model_dense import MyLLM
from model_lib.my_model_moe import MyLLMMoE


class ModelManager:
    def __init__(self, args):
        self.args                   = args
        self.model_name             = args.model_name
        self.from_pretrained        = args.from_pretrained
        self.weight_dir             = args.weight_dir
        self.verbose                = args.verbose

        self.regenerate_output      = args.regenerate_output
        self.validate_output_ids    = args.validate_output_ids
        
        self.w_dtype                = args.w_dtype
        self.norm_dtype             = args.norm_dtype
        self.kv_dtype               = args.kv_dtype        
        self.sim_num_layers         = args.sim_num_layers
        self.validate_pistil_intermediates = args.validate_pistil_intermediates

        if self.args.sim_prefill_batch:
            self.real_batch_size = self.args.sim_batch_size - self.args.sim_prefill_chunk_size + 1
        else:
            self.real_batch_size = self.args.sim_batch_size 

        self.batch_size             = 1
        self.max_new_tokens         = args.max_new_tokens
        self.dtype                  = torch.bfloat16
        self.prompt                 = [
            {"role": "user", "content": "You are a pirate chatbot who always responds in pirate speak! Who are you and tell me a story?"},
        ]
        self.random_tokens_init     = args.kv_cache
        self.prefill_input_ids      = None
        self.decode_output_ids      = None
        self.saved_prefill_file     = f"./validate_model_io/{self.model_name}-prefill.pt"
        self.saved_decode_file      = f"./validate_model_io/{self.model_name}-decode.pt"

        self.base_tokenizer         = None
        self.base_model             = None
        self.my_model               = None
        self.model_type             = None

        self.model_dim              = None
        self.num_layers             = None
        self.ff_dim                 = None
        self.head_dim               = None
        self.num_heads              = None
        self.kv_heads               = None
        self.gqa                    = None
        self.max_seq_len            = args.sim_kv_cache
        self.rope_theta             = None
        self.rms_norm_eps           = None
        self.vocab_size             = None
        self.act_fn                 = None
        self.attn_softcap           = None
        self.q_pre_attn_scalar      = None

        self.init_model()

        if self.model_type == "dense":
            self.calculate_capacity_dense()
        elif self.model_type == "moe":
            self.calculate_capacity_moe()

    def init_model(self):
        self.init_random_model_params()

    def init_random_model_params(self):
        if self.args.model_cfg != None:
            import json
            with open(self.args.model_cfg, 'r') as file:
                model_cfg = json.load(file)

            self.model_name             = model_cfg["name"]
            self.num_layers             = model_cfg["num_hidden_layers"]
            self.model_dim              = model_cfg["hidden_size"]
            self.num_heads              = model_cfg["num_attention_heads"]
            self.head_dim               = int(self.model_dim / self.num_heads) if "head_dim" not in model_cfg else model_cfg["head_dim"]
            self.kv_heads               = model_cfg["num_key_value_heads"]
            self.gqa                    = int(self.num_heads/self.kv_heads)
            self.max_seq_len            = self.args.sim_kv_cache #model_cfg["max_position_embeddings"]
            self.rope_theta             = model_cfg["rope_theta"]
            self.rms_norm_eps           = model_cfg["rms_norm_eps"]
            self.vocab_size             = model_cfg["vocab_size"]
            self.act_fn                 = model_cfg["hidden_act"]
            self.q_pre_attn_scalar      = self.head_dim**(-0.5)

            if "type" not in model_cfg or model_cfg["type"] == "dense":
                self.ff_dim                 = model_cfg["intermediate_size"]
                self.model_type             = "dense"
            else:
                self.num_experts_per_tok    = model_cfg["num_experts_per_tok"]
                self.num_local_experts      = model_cfg["num_local_experts"]
                self.ff_dim_moe             = model_cfg["intermediate_size_moe"]
                self.ff_dim_mlp             = model_cfg["intermediate_size_mlp"]
                self.moe_layers             = model_cfg["moe_layers"]
                self.no_rope_layers         = model_cfg["no_rope_layers"] # Chunked Attention
                self.attention_chunk_size   = model_cfg["attention_chunk_size"] 
                self.interleave_rope        = model_cfg["interleave_rope"]
                self.interleave_moe         = model_cfg["interleave_moe"]
                self.model_type             = "moe"

        else:
            print("Model random initializer not configured\n\t%s" % self.model_name)
            exit()
            
        if self.model_type == "dense":
            self.my_model               = MyLLM(self)
        elif self.model_type == "moe":
            self.my_model               = MyLLMMoE(self)

        if self.args.inference:
            if self.verbose > 0:
                print("Initializing Random Model Parameters")
            t1 = time.perf_counter()
            self.my_model.init_random_params(self.sim_num_layers)
            t2 = time.perf_counter()
            if self.verbose > 0:
                print("\tInit Time: %0.2fs" % (t2 - t1))
            self.encode_prompt_from_random()

    def encode_prompt_from_pretrained(self):
        if os.path.exists(self.saved_prefill_file) and self.validate_output_ids:
            self.prefill_input_ids = torch.load(self.saved_prefill_file, weights_only=True)
            self.decode_output_ids = torch.load(self.saved_decode_file, weights_only=True)
            max_new_tokens_in_file = self.decode_output_ids.size()[-1]

            if max_new_tokens_in_file >= self.max_new_tokens:
                self.decode_output_ids = self.decode_output_ids[:, :self.max_new_tokens]
            elif self.validate_output_ids == True:
                if not self.regenerate_output:
                    print("\tWarning: Number of tokens desired not previously generated - please rerun base_model_generation using `--regenerate-output=True` to capture decoded output.")
                    exit()
        else:
            input_ids = self.base_tokenizer.apply_chat_template(
                self.prompt,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            self.prefill_input_ids = input_ids
            if not self.regenerate_output and self.validate_output_ids == True:
                print("\tWarning: Decoded token IDs not previously generated. Must run using `--regenerate-output=True` to capture decoded output")
                exit()
         
    def encode_prompt_from_random(self):
        self.prefill_input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.random_tokens_init))

    def print_row(self):
        print("########################################")

    def print_minor_row(self):
        print("----------------------------------------")

    def calculate_capacity_dense(self):
        # layer param breakdown
        attention_params    = 2 * self.model_dim * (self.head_dim * self.num_heads) + 2 * self.model_dim * (self.head_dim * self.kv_heads)
        feed_forward_params = self.model_dim * (self.ff_dim * 2) + self.model_dim * self.ff_dim
        layer_norm_params   = self.model_dim

        # kv cache breakdown
        max_kv_cache_params = 2 * self.max_seq_len * (self.head_dim * self.kv_heads) # *2 for K and V
        
        norm_layers         = 4 if self.model_name == "gemma2-2B" else 2

        # all params
        layer_params        = attention_params + feed_forward_params + norm_layers * layer_norm_params 
        kv_params           = max_kv_cache_params * self.num_layers 
        emb_params          = self.vocab_size * self.model_dim
        lm_head             = self.vocab_size * self.model_dim
        model_params        = layer_params * self.num_layers + emb_params + lm_head

        w_capacity          = ((attention_params + feed_forward_params) * self.w_dtype + (norm_layers * layer_norm_params) * self.kv_dtype)*self.num_layers + (emb_params + lm_head) * self.w_dtype + (self.max_seq_len*self.head_dim*2)*self.kv_dtype
        kv_capacity         = kv_params * self.kv_dtype 
        total_capacity      = w_capacity + kv_capacity * self.real_batch_size

        if self.verbose > 0:
            self.print_row()
            print("Model Capacity Analysis")
            print("\tModel Name:\t%s" % self.model_name.rjust(12))
            print("\tModel Dim:\t%12.0f" % (self.model_dim))
            print("\tNum Layers:\t%12.0f" % (self.num_layers))
            print("\tFF Dim:\t\t%12.0f" % (self.ff_dim))
            print("\tHead Dim:\t%12.0f" % (self.head_dim))
            print("\tNum Heads:\t%12.0f" % (self.num_heads))
            print("\tKV Heads:\t%12.0f" % (self.kv_heads))
            print("\tMax Seq:\t%12.0f" % (self.max_seq_len))
            print("\tVocab Size:\t%12.0f" % (self.vocab_size))

            self.print_minor_row()
            print("\tModel Layer Params:%12.2f B Params" % (layer_params * self.num_layers/(10**9)))
            print("\tModel Params:\t%12.2f B Params" % (model_params/(10**9)))
            print("\tKV Params (max):%12.2f B Params" % (kv_params/(10**9)))
            print("\tLayer Params:\t%12.2f M Params" % (layer_params/(10**6)))
            print("\tLayer KV (max): %12.2f M Params" % (max_kv_cache_params/(10**6)))
            print("\tWeight Size:\t%12.2f B / Param" % (self.w_dtype))
            print("\tKV Size:\t%12.2f B / Param" % (self.kv_dtype))
            self.print_minor_row()
            print("\tTotal Weights:\t%12.2f GB" % (w_capacity/(1024**3)))
            print("\tKV$ Per Batch:\t%12.2f GB - BS=%i" % (kv_capacity/(1024**3), self.real_batch_size))
            print("\tTotal Capacity:\t%12.2f GB" % (total_capacity/(1024**3)))
            self.print_row()
        
        self.w_capacity     = w_capacity
        self.kv_capacity    = kv_capacity
        self.total_capacity = total_capacity


    def calculate_embedding_params(self):
        emb_params = self.vocab_size * self.model_dim
        return emb_params
    
    def calculate_rotary_emb_params(self):
        rotary_emb_params = 2 * self.max_seq_len * self.head_dim
        return rotary_emb_params

    def calculate_lm_head_params(self):
        lm_head_params = self.model_dim * self.vocab_size
        return lm_head_params
    
    def calculate_rms_norm_params(self):
        norm_params = self.model_dim
        return norm_params

    def calculate_attention_params(self):
        q_proj = self.model_dim * (self.num_heads * self.head_dim)
        k_proj = self.model_dim * (self.kv_heads * self.head_dim)
        v_proj = self.model_dim * (self.kv_heads * self.head_dim)
        o_proj = (self.num_heads * self.head_dim) * self.model_dim
        return q_proj + k_proj + v_proj + o_proj

    def calculate_chunked_attention_params(self):
        attention_params = self.calculate_attention_params()
        kv_params = 2 * (self.kv_heads * (self.attention_chunk_size * self.head_dim)) # e.g. (8k * 128) * 8 heads * 2 K/V
        return attention_params, kv_params

    def calculate_full_attention_params(self):
        attention_params = self.calculate_attention_params()
        kv_params = 2 * (self.kv_heads * (self.max_seq_len * self.head_dim)) 
        return attention_params, kv_params
    
    def calculate_moe_params(self):
        # router projection matrix for deriving routed experts
        router_proj = self.model_dim * self.num_local_experts

        # experts
        gate_proj = self.model_dim * self.ff_dim_moe
        up_proj = self.model_dim * self.ff_dim_moe
        down_proj = self.ff_dim_moe * self.model_dim
        expert_params = gate_proj + up_proj + down_proj

        # routed expert summary
        all_routed_expert_params = expert_params * self.num_local_experts
        activated_expert_params = expert_params * self.num_experts_per_tok

        # shared expert - Llama4 Implicit 1 shared expert
        shared_expert_params = expert_params

        return router_proj, all_routed_expert_params, activated_expert_params, shared_expert_params
    
    def calculate_mlp_params(self):
        gate_proj = self.model_dim * self.ff_dim_mlp
        up_proj = self.model_dim * self.ff_dim_mlp
        down_proj = self.ff_dim_mlp * self.model_dim
        mlp_params = gate_proj + up_proj + down_proj
        return mlp_params

    def calculate_capacity_moe(self):
        emb_params              = self.calculate_embedding_params()
        rotary_emb_params       = self.calculate_rotary_emb_params()
        lm_head_norm_params     = self.calculate_rms_norm_params()
        lm_head_params          = self.calculate_embedding_params()

        # decoder layer parameters
        self.attention_params   = 0
        self.kv_params          = 0
        self.moe_params         = 0
        self.moe_active_params  = 0
        self.mlp_params         = 0
        self.rms_params         = 0
        for layer in range(self.num_layers):
            if self.no_rope_layers[layer] == 0: # full attention
                params                  = self.calculate_full_attention_params()
                self.attention_params   += params[0]
                self.kv_params          += params[1]
            else:
                params                  = self.calculate_chunked_attention_params()
                self.attention_params   += params[0]
                self.kv_params          += params[1]

            if layer in self.moe_layers:
                params                  = self.calculate_moe_params()
                # router_proj, all_routed_expert_params, activated_expert_params, shared_expert_params
                self.moe_params         += params[0] + params[1] + params[3]
                self.moe_active_params  += params[0] + params[2] + params[3]
            else:
                self.mlp_params         += self.calculate_mlp_params()
            
            # input norm and post-attention norm
            self.rms_params             += 2 * self.calculate_rms_norm_params()
        
        self.emb_params                 = emb_params
        self.rotary_emb_params          = rotary_emb_params
        self.lm_head_norm_params        = lm_head_norm_params
        self.lm_head_params             = lm_head_params

        self.total_params               = self.emb_params + \
                                          self.rotary_emb_params + \
                                          self.lm_head_params + \
                                          self.lm_head_norm_params + \
                                          self.attention_params + \
                                          self.moe_params + \
                                          self.mlp_params + \
                                          self.rms_params
                                        #   self.kv_params + \
        
        self.active_params              = self.model_dim + \
                                          int(self.rotary_emb_params/self.max_seq_len) + \
                                          self.lm_head_params + \
                                          self.lm_head_norm_params + \
                                          self.attention_params + \
                                          self.moe_active_params + \
                                          self.mlp_params + \
                                          self.rms_params
                                          #   self.kv_params + \

        self.total_capacity             = self.emb_params * self.kv_dtype + \
                                          self.rotary_emb_params * self.norm_dtype + \
                                          self.lm_head_params * self.w_dtype + \
                                          self.lm_head_norm_params * self.norm_dtype + \
                                          self.attention_params * self.w_dtype + \
                                          self.moe_params * self.w_dtype + \
                                          self.mlp_params * self.w_dtype + \
                                          self.rms_params * self.norm_dtype
                                        #   self.kv_params * self.kv_dtype + \
        self.w_capacity                 = self.emb_params * self.kv_dtype + \
                                          self.rotary_emb_params * self.norm_dtype + \
                                          self.lm_head_params * self.w_dtype + \
                                          self.lm_head_norm_params * self.norm_dtype + \
                                          self.attention_params * self.w_dtype + \
                                          self.moe_params * self.w_dtype + \
                                          self.mlp_params * self.w_dtype + \
                                          self.rms_params * self.norm_dtype
                                        
        self.kv_capacity                = self.kv_params * self.kv_dtype
        
        self.active_capacity            = self.model_dim * self.kv_dtype + \
                                          int(self.rotary_emb_params/self.max_seq_len) * self.norm_dtype + \
                                          self.lm_head_params * self.w_dtype + \
                                          self.lm_head_norm_params * self.norm_dtype + \
                                          self.attention_params * self.w_dtype + \
                                          self.moe_active_params * self.w_dtype + \
                                          self.mlp_params * self.w_dtype + \
                                          self.rms_params * self.norm_dtype
                                          #   self.kv_params * self.kv_dtype + \
        self.print_summary_moe()

    def print_summary_moe(self):
        sep = "=" * 60
        print(sep)
        print("Model Configuration")
        print(sep)
        print(f"Model Name:               {self.model_name}")
        print(f"Hidden Layers:            {self.num_layers}")
        print(f"Vocabulary Size:          {self.vocab_size}")
        print(f"Hidden Size:              {self.model_dim}\n")

        # Attention configuration
        print("--- Attention Configuration ---")
        print(f"Head Dimension:           {self.head_dim}")
        print(f"Attention Heads:          {self.num_heads}")
        print(f"Key/Value Heads:          {self.kv_heads}")
        print(f"Max Sequence Length:      {self.max_seq_len}")
        print(f"Attention Chunk Size:     {self.attention_chunk_size}")
        print(f"RoPE Interleave Every:    {self.interleave_rope} layers (full attention)\n")

        # MoE configuration
        print("--- MoE/MLP Configuration ---")
        print(f"MoE Intermediate Size:    {self.ff_dim_moe}")
        print(f"MLP Intermediate Size:    {self.ff_dim_mlp}")
        print(f"Local Experts:            {self.num_local_experts}")
        print(f"Routed Experts per Token: {self.num_experts_per_tok}")
        print(f"MoE Interleave Every:     {self.interleave_moe} layers\n")

        # Capacity (in units of datatype size)
        print("--- Data Type Assumptions ---")
        print(f"Weight Dtype:              {self.w_dtype:,} B/Param")
        print(f"KV-Cache Dtype:            {self.kv_dtype:,} B/Param")
        print(f"Embedding Dtype:           {self.kv_dtype:,} B/Param")
        print(f"Normalize Dtype:           {self.norm_dtype:,} B/Param")
        print(f"Rotary Embedding Dtype:    {self.norm_dtype:,} B/Param\n")


        # Capacity summary
        print(sep)
        print(f"Model Summary Per Layer")
        print(sep)

        # Detailed Attention breakdown
        print("--- Attention Parameter Breakdown ---")
        attention_params, kv_chunked = self.calculate_chunked_attention_params()
        _, kv_full = self.calculate_full_attention_params()
        print(f"Weight Matrices:           {attention_params:,}")
        print(f"KV Cache Chunked:          {kv_chunked:,}")
        print(f"KV Cache Full Attention:   {kv_full:,}")
        # print(f"Chunked Attention Layers:  {self.chunked_attention_layers}")
        # print(f"Full Attention Layers:     {self.full_attention_layers}\n")
        


        # Detailed MoE breakdown
        print("--- Mixture-of-Experts (MoE) Parameter Breakdown ---")
        params = self.calculate_moe_params()
        router_proj = params[0]
        gate_proj = params[0]
        up_proj = self.model_dim * self.ff_dim_moe
        down_proj = self.ff_dim_moe * self.model_dim
        all_experts = params[1]
        active_experts = params[2]
        shared_expert = params[3] # shared expert size

        print(f"Router Projection Matrix:  {router_proj:,}")
        print(f"Expert Block (per expert): {shared_expert:,}")
        print(f"All Routed Experts:        {all_experts:,} ({self.num_local_experts} experts)")
        print(f"Active Routed Experts:     {active_experts:,} ({self.num_experts_per_tok}/token)")
        print(f"Shared Expert Parameters:  {shared_expert:,}")
        print(f"MoE Layers:                {self.moe_layers}\n")
        
        # Detailed MLP breakdown
        print("--- MLP Layers (Non-MoE) Parameter Breakdown ---")
        mlp_params = self.calculate_mlp_params()
        print(f"MLP Params:                {mlp_params:,}")
        # print(f"MLP Layers:                {self.mlp_layers}\n")
        
        print(sep)
        print("Model Summary")
        print(sep)

        # Parameter counts
        print(f"Token Embedding Params:    {self.emb_params:,}")
        print(f"Rotary Embeddings Params:  {self.rotary_emb_params:,}")
        print(f"LM Head Params:            {self.lm_head_params:,}")
        print(f"LM Head RMSNorm Params:    {self.lm_head_norm_params:,}\n")

        print(f"Attention Params:          {self.attention_params:,}")
        print(f"KV-Cache Params (max):     {self.kv_params:,} (per query)\n")

        print(f"MLP (Non-MoE) Params:      {self.mlp_params:,}")
        print(f"MoE Total Params:          {self.moe_params:,}")
        print(f"MoE Active Params:         {self.moe_active_params:,}\n")

        print(f"Total Parameter Count:     {self.total_params:,}")
        print(f"Active Parameter Count:    {self.active_params:,}\n")
        print(f"Total Capacity (Bytes):    {self.total_capacity:,}")
        print(f"Active Capacity (Bytes):   {self.active_capacity:,}")
        print(sep)

    # run a forward pass of the base model using pytorch framework and record the output into .pkl file
    def base_model_forward(self):
        if self.from_pretrained and (self.regenerate_output or not os.path.exists(self.saved_prefill_file)):
            attention_mask = torch.ones(self.prefill_input_ids.shape, dtype=torch.long)

            terminators = [
                self.base_tokenizer.eos_token_id,
                self.base_tokenizer.convert_tokens_to_ids("<|eot_id|>") 
            ]
            pad_token_id = self.base_tokenizer.eos_token_id

            if self.verbose > 0:
                print("Starting Model Generation for PyTorch LLM Implementation")
            
            t1 = time.perf_counter_ns()
            self.base_model.eval()
            with torch.no_grad():
                output_ids = self.base_model.generate(
                        self.prefill_input_ids,
                        max_new_tokens=self.max_new_tokens,
                        eos_token_id=terminators,
                        pad_token_id=pad_token_id,
                        attention_mask=attention_mask,
                        do_sample=False,
                        use_cache=True,
                        #temperature=0.6,
                        #top_p=0.9,
                    )
            decoded_outputs = self.base_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            t2 = time.perf_counter_ns()
            if self.verbose > 0:
                print("Generated Output:", decoded_outputs)
                print("Time to Generate %0.2fs" % ((t2-t1) / 10**9))

            self.decode_output_ids = output_ids[:, self.prefill_input_ids.size()[1]:]

            torch.save(self.prefill_input_ids, self.saved_prefill_file)
            torch.save(self.decode_output_ids, self.saved_decode_file)

    def my_model_forward(self, max_new_tokens=None):
        max_new_tokens = self.max_new_tokens if max_new_tokens == None else max_new_tokens
        self.my_model.reset()
        with torch.no_grad():
            self.my_model.forward(self.prefill_input_ids, max_new_tokens=max_new_tokens, valid_output_ids=(self.decode_output_ids if self.validate_output_ids else None), sim_num_layers=self.sim_num_layers)
