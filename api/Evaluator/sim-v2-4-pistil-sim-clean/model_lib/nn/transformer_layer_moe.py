import math
import torch

from model_lib.nn.linear import LinearLayer
from model_lib.nn.layer_norm import LayerNorm
from model_lib.nn.rotary_emb import RotaryEmbeddings
from model_lib.nn.kv_cache import KVCache


class TransformerLayerMoE:
    def __init__(self, my_model, layer_id, model_dim, ff_dim_moe, num_local_experts, num_experts_per_tok, head_dim, num_heads, kv_heads, max_seq_len, cos, sin, rms_norm_eps, act_fn, attn_softcap, q_pre_attn_scalar, dtype):
        self.my_model           = my_model
        self.save_state         = self.my_model.save_state
        self.layer_id           = layer_id
        self.batch_size         = 1
        self.model_dim          = model_dim 
        self.ff_dim             = ff_dim_moe     
        self.num_local_experts  = num_local_experts
        self.num_experts_per_tok= num_experts_per_tok

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

        self.wQ                 = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.head_dim * self.num_heads, self.input_dtype)
        self.wK                 = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.head_dim * self.kv_heads, self.input_dtype)
        self.wV                 = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.head_dim * self.kv_heads, self.input_dtype)
        self.wQKV               = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.head_dim * self.num_heads + 2 * self.head_dim * self.kv_heads, self.input_dtype)
        self.wO                 = LinearLayer(self.my_model, self.layer_id, self.head_dim * self.num_heads, self.model_dim, self.input_dtype)
        
        # shared expert
        self.gate_proj_s        = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.ff_dim, self.input_dtype)
        self.up_proj_s          = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.ff_dim, self.input_dtype)
        self.gate_up_proj_s     = LinearLayer(self.my_model, self.layer_id, self.model_dim, 2*self.ff_dim, self.input_dtype)
        self.down_proj_s        = LinearLayer(self.my_model, self.layer_id, self.ff_dim, self.model_dim, self.input_dtype)
        
        # routed expert - just sim 1 for speed, but model capacity as num_local_experts
        self.expert_proj        = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.num_local_experts, self.input_dtype)
        self.gate_proj_e        = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.ff_dim, self.input_dtype)
        self.up_proj_e          = LinearLayer(self.my_model, self.layer_id, self.model_dim, self.ff_dim, self.input_dtype)
        self.gate_up_proj_e     = LinearLayer(self.my_model, self.layer_id, self.model_dim, 2*self.ff_dim, self.input_dtype)
        self.down_proj_e        = LinearLayer(self.my_model, self.layer_id, self.ff_dim, self.model_dim, self.input_dtype)

        self.input_norm         = LayerNorm(self.my_model, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.post_atten_norm    = LayerNorm(self.my_model, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.pre_ff_norm        = LayerNorm(self.my_model, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.post_ff_norm       = LayerNorm(self.my_model, self.layer_id, self.model_dim, self.rms_norm_eps, self.input_dtype)
        self.kv_cache           = KVCache(self, self.layer_id, self.batch_size, self.max_seq_len, self.gqa, self.kv_heads, self.head_dim, self.q_pre_attn_scalar, self.input_dtype)
        self.rotary_emb         = RotaryEmbeddings(self, self.layer_id, self.cos, self.sin, self.kv_cache)

    def reset(self):
        self.kv_cache.reset()

    def init_random(self, use_additional_norms=False):
        self.wQ.init_random()
        self.wK.init_random()
        self.wV.init_random()
        self.wQKV.init_random()
        self.wO.init_random()
        
        # shared expert
        self.gate_proj_s.init_random()
        self.up_proj_s.init_random()
        self.gate_up_proj_s.init_random()
        self.down_proj_s.init_random()

        # routed expert
        self.expert_proj.init_random()
        self.gate_proj_e.init_random()
        self.up_proj_e.init_random()
        self.gate_up_proj_e.init_random()
        self.down_proj_e.init_random()

        self.input_norm.init_random()
        self.pre_ff_norm.init_random()
        if use_additional_norms:
            self.post_atten_norm.init_random()
            self.post_ff_norm.init_random()
        

    def load_q_from_pretrained(self, params):
        self.wQ.load_from_pretrained(params)
        if self.wK.weights != None and self.wK.weights != None and self.wV.weights != None:
            self.wQKV.load_from_pretrained(torch.cat((self.wQ.weights.T, self.wK.weights.T, self.wV.weights.T), dim=-1).T)

    def load_k_from_pretrained(self, params):
        self.wK.load_from_pretrained(params)
        if self.wK.weights != None and self.wK.weights != None and self.wV.weights != None:
            self.wQKV.load_from_pretrained(torch.cat((self.wQ.weights.T, self.wK.weights.T, self.wV.weights.T), dim=-1).T)

    def load_v_from_pretrained(self, params):
        self.wV.load_from_pretrained(params)
        if self.wK.weights != None and self.wK.weights != None and self.wV.weights != None:
            self.wQKV.load_from_pretrained(torch.cat((self.wQ.weights.T, self.wK.weights.T, self.wV.weights.T), dim=-1).T)
    
    def load_o_from_pretrained(self, params):
        self.wO.load_from_pretrained(params)
    
    def load_gate_proj_from_pretrained(self, params):
        self.gate_proj.load_from_pretrained(params)
        if self.gate_proj.weights != None and self.up_proj.weights != None:
            self.gate_up_proj.load_from_pretrained(torch.cat((self.gate_proj.weights.T, self.up_proj.weights.T), dim=-1).T)
    
    def load_up_proj_from_pretrained(self, params):
        self.up_proj.load_from_pretrained(params)
        if self.gate_proj.weights != None and self.up_proj.weights != None:
            self.gate_up_proj.load_from_pretrained(torch.cat((self.gate_proj.weights.T, self.up_proj.weights.T), dim=-1).T)
    
    def load_down_proj_from_pretrained(self, params):
        self.down_proj.load_from_pretrained(params)
    
    def load_input_norm_from_pretrained(self, params):
        self.input_norm.load_from_pretrained(params)
    
    def load_post_atten_norm_from_pretrained(self, params):
        self.post_atten_norm.load_from_pretrained(params)
    
    def load_pre_feedforward_norm_from_pretrained(self, params):
        self.pre_ff_norm.load_from_pretrained(params)
    
    def load_post_feedforward_norm_from_pretrained(self, params):
        self.post_ff_norm.load_from_pretrained(params)
    
    def validate_layer_size(self):
        validated = True
        if self.wQ.size()[0] != self.head_dim * self.num_heads and self.wQ.size()[1] != self.model_dim:
            print("Layer %i wQ size incorrect: %s != (%i, %i)" % (self.layer_id, str(self.wQ.size()), self.head_dim * self.num_heads, self.model_dim))
            validated = False
        if self.wK.size()[0] != self.head_dim * self.kv_heads and self.wK.size()[1] != self.model_dim:
            print("Layer %i wK size incorrect: %s != (%i, %i)" % (self.layer_id, str(self.wK.size()), self.head_dim * self.kv_heads, self.model_dim))
            validated = False
        if self.wV.size()[0] != self.head_dim * self.kv_heads and self.wV.size()[1] != self.model_dim:
            print("Layer %i wV size incorrect: %s != (%i, %i)" % (self.layer_id, str(self.wV.size()), self.head_dim * self.kv_heads, self.model_dim))
            validated = False
        if self.wO.size()[0] != self.model_dim and self.wO.size()[1] != self.head_dim * self.num_heads:
            print("Layer %i wO size incorrect: %s != (%i, %i)" % (self.layer_id, str(self.wO.size()), self.model_dim, self.head_dim * self.num_heads))
            validated = False

        if self.gate_proj.size()[0] != self.ff_dim and self.gate_proj.size()[1] != self.model_dim:
            print("Layer %i gate_proj size incorrect: %s != (%i, %i)" % (self.layer_id, str(self.gate_proj.size()), self.ff_dim, self.model_dim))
            validated = False
        if self.up_proj.size()[0] != self.ff_dim and self.up_proj.size()[1] != self.model_dim:
            print("Layer %i up_proj size incorrect: %s != (%i, %i)" % (self.layer_id, str(self.up_proj.size()), self.ff_dim, self.model_dim))
            validated = False
        if self.down_proj.size()[0] != self.model_dim and self.down_proj.size()[1] != self.ff_dim:
            print("Layer %i down_proj size incorrect: %s != (%i, %i)" % (self.layer_id, str(self.down_proj.size()), self.model_dim, self.ff_dim))
            validated = False
        
        if self.input_norm.size()[0] != self.model_dim:
            print("Layer %i input_norm size incorrect: %s != (%i)" % (self.layer_id, str(self.input_norm.size()), self.model_dim))
            validated = False
        if self.post_atten_norm.size()[0] != self.model_dim:
            print("Layer %i post_atten_norm size incorrect: %s != (%i)" % (self.layer_id, str(self.post_atten_norm.size()), self.model_dim))
            validated = False
        
        return validated


    def forward(self, layer_input, layer_id, use_base=False):
        seq_len = layer_input.size()[1]

        ##################################################################
        # normalize layer
        if use_base:
            hidden_states   = self.input_norm.forward(layer_input)
        else:
            hidden_states   = self.input_norm.forward_weights(layer_input)
        ##################################################################
        self.save_state.save_state(layer_id, name="input_norm_weights", param=hidden_states)

        ##################################################################
        # SDPA
        ##################################################################
        # wQKV
        if self.my_model.fused_wQKV == True:
            hidden_states = self.wQKV.forward(hidden_states)
            if not use_base:
                hidden_states = self.input_norm.forward_var(hidden_states)

            q_vector = hidden_states[:, :, 0:self.num_heads*self.head_dim]
            k_vector = hidden_states[:, :, self.num_heads*self.head_dim:self.num_heads*self.head_dim+self.kv_heads*self.head_dim]
            v_vector = hidden_states[:, :, self.num_heads*self.head_dim+self.kv_heads*self.head_dim:]
        else:
            q_vector = self.wQ.forward(hidden_states)
            k_vector = self.wK.forward(hidden_states)
            v_vector = self.wV.forward(hidden_states)  
            self.save_state.save_state(layer_id, name="q_vector", param=q_vector)
            self.save_state.save_state(layer_id, name="k_vector", param=k_vector)
            self.save_state.save_state(layer_id, name="v_vector", param=v_vector)
            
            if not use_base:
                q_vector = self.input_norm.forward_var(q_vector)
                k_vector = self.input_norm.forward_var(k_vector)
                v_vector = self.input_norm.forward_var(v_vector)
                self.save_state.save_state(layer_id, name="q_vector_var", param=q_vector)
                self.save_state.save_state(layer_id, name="k_vector_var", param=k_vector)
                self.save_state.save_state(layer_id, name="v_vector_var", param=v_vector)

        q_vector = q_vector.view(self.batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_vector = k_vector.view(self.batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        v_vector = v_vector.view(self.batch_size, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        ##################################################################


        ##################################################################
        q_vector, k_vector = self.rotary_emb.forward(seq_len, q_vector, k_vector)        
        ##################################################################

        self.save_state.save_state(layer_id, name="q_rot", param=q_vector)
        self.save_state.save_state(layer_id, name="k_rot", param=k_vector)
        
        ##################################################################
        # soft(QKT)V
        self.kv_cache.write_kv(seq_len, k_vector, v_vector)
        self.kv_cache.forward_repeat_kv(seq_len)
        scores = self.kv_cache.forward_QKT(seq_len, q_vector)
        if self.attn_softcap != None:
            scores = scores / self.attn_softcap
            scores = torch.tanh(scores)
            scores = scores * self.attn_softcap
        soft_scores = self.kv_cache.forward_soft(seq_len, scores)
        hidden_states = self.kv_cache.forward_V(seq_len, soft_scores)
        ##################################################################

        self.save_state.save_state(layer_id, name="scores", param=scores)
        self.save_state.save_state(layer_id, name="soft_scores", param=soft_scores)
        self.save_state.save_state(layer_id, name="sQKT_V", param=hidden_states)


        ##################################################################
        # wO        
        hidden_states        = self.wO.forward(hidden_states)
        self.save_state.save_state(layer_id, name="o_vector", param=hidden_states)
        ##################################################################

        # specific to gemma2
        if self.post_atten_norm.weights != None:
            if use_base:
                hidden_states   = self.post_atten_norm.forward(hidden_states)
            else:
                hidden_states   = self.post_atten_norm.forward_weights(hidden_states)
                hidden_states   = self.post_atten_norm.forward_var(hidden_states)
            self.save_state.save_state(layer_id, name="post_atten_norm", param=hidden_states)

        ##################################################################
        # add and layer norm
        layer_sum       = hidden_states + layer_input
        self.save_state.save_state(layer_id, name="layer_sum_attention", param=layer_sum)


        # layer norm
        hidden_states   = self.pre_ff_norm.forward(layer_sum)
        self.save_state.save_state(layer_id, name="pre_ff_norm_weights", param=hidden_states)
        
        #################################################################


        #################################################################
        # Shared Feed Forward Layers
        #################################################################
        # gate and up proj
        hidden_states_gate  = self.gate_proj_s.forward(hidden_states)
        hidden_states_up    = self.up_proj_s.forward(hidden_states)
        #################################################################

        self.save_state.save_state(layer_id, name="gate", param=hidden_states_gate)
        self.save_state.save_state(layer_id, name="up", param=hidden_states_up)

        self.save_state.save_state(layer_id, name="pre_ff_norm_var_gate", param=hidden_states_gate)
        self.save_state.save_state(layer_id, name="pre_ff_norm_var_up", param=hidden_states_up)

        #################################################################
        # activation
        if self.act_fn == "silu":
            hidden_states_gate  = torch.nn.SiLU()(hidden_states_gate)        
        elif self.act_fn == "gelu_pytorch_tanh":
            hidden_states_gate  = torch.nn.GELU(approximate="tanh")(hidden_states_gate)  
        else:
            print("Error: Unknown Activation Function: %s" % self.act_fn)
            exit()

        # multiply
        hidden_states_shared   = hidden_states_gate * hidden_states_up
        #################################################################

        self.save_state.save_state(layer_id, name="act", param=hidden_states_shared)


        #################################################################
        # mlp down proj
        hidden_states_shared   = self.down_proj_s.forward(hidden_states_shared)
        self.save_state.save_state(layer_id, name="down", param=hidden_states_shared)
        ##################################################################
        

        #################################################################
        # Routed Feed Forward Layers
        #################################################################
        # gate and up proj
        expert_to_use       = self.expert_proj.forward(hidden_states)
        # take softmax of the expert to use to scale result of the the expert

        hidden_states_gate  = self.gate_proj_e.forward(hidden_states)
        hidden_states_up    = self.up_proj_e.forward(hidden_states)
        #################################################################


        #################################################################
        # activation
        if self.act_fn == "silu":
            hidden_states_gate  = torch.nn.SiLU()(hidden_states_gate)        
        elif self.act_fn == "gelu_pytorch_tanh":
            hidden_states_gate  = torch.nn.GELU(approximate="tanh")(hidden_states_gate)  
        else:
            print("Error: Unknown Activation Function: %s" % self.act_fn)
            exit()

        # multiply
        hidden_states_expert   = hidden_states_gate * hidden_states_up
        #################################################################


        #################################################################
        # mlp down proj
        hidden_states_expert   = self.down_proj_s.forward(hidden_states_expert)
        ##################################################################

        hidden_states = hidden_states_shared + hidden_states_expert


        # specific to gemma2
        if self.post_ff_norm.weights != None:
            if use_base:
                hidden_states   = self.post_ff_norm.forward(hidden_states)
            else:
                hidden_states   = self.post_ff_norm.forward_weights(hidden_states)
                hidden_states   = self.post_ff_norm.forward_var(hidden_states)

        ##################################################################
        # add
        hidden_states += layer_sum
        self.save_state.save_state(layer_id, name="layer_sum_output", param=hidden_states)
        ##################################################################

        return hidden_states

    