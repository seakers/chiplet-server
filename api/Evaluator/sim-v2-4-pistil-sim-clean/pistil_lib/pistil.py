import time
import math
import torch
import numpy as np

from model_lib.my_model_shard_dense import MyLLMShard
from model_lib.my_model_shard_moe import MyLLMShardMoE
from pistil_lib.lib.pistil_loader import PistilLoader
from pistil_lib.lib.pistil_nic import PistilNIC

class Pistil:
    def __init__(self, 
            garden=None,
            chiplet_id=0,
            sys_manager=None,
            llm_manager=None,
            verbose=1
        ):
        self.args           = garden.args
        self.garden         = garden
        self.chiplet_id     = chiplet_id
        self.sys_manager    = sys_manager
        self.llm_manager    = llm_manager
        self.model_shard    = None
        self.pistil_loader  = PistilLoader(self)
        self.pistil_nic     = None 
        self.pistil_transactor = None               # transactional simulator

        self.verbose        = verbose

        self.e_chiplet_id   = (self.chiplet_id + 1) % self.garden.num_chiplets
        self.w_chiplet_id   = (self.chiplet_id - 1) % self.garden.num_chiplets
        
        # hardware buffers
        self.accum_shard    = None      # storage for accumulation buffer on chip
        self.act_shard      = None      # activation storage on chip
        self.act_shard_saved = None     # in case we need to reuse the act-shard later
        self.act_shard_tx   = None      # activation storage on chip from network - to send to next chiplet
        self.act_shard_rx   = None      # activation storage on chip from network - to recieve from next chiplet
        self.other_data     = {}        # data from network 
        self.other_data_tx  = {}        # data from network buffered
        self.other_data_rx  = {}        # data from network buffered

        # layer specific hardware buffers
        self.singleton_buf  = None      # accumulation buffer for accumulating layer norm variance, softscore max value, softmax exp sum, and final max 
        self.add_input_buf  = None      # buffer sized for activation input accumulation

        self.act_model_dim_sharding = None
        self.act_ff_dim_sharding = None

        self.head_receive_order         = None
        self.qkv_nic_config_algorithm   = None
        self.o_nic_config_algorithm     = None

    def print_row(self):
        print("########################################")

    def print_config(self):
        self.print_row()

        self.compute_ops    = self.sys_manager.get_core_compute() * self.sys_manager.get_cores_per_chiplet()
        self.mem_bw         = self.sys_manager.get_core_bandwidth() * self.sys_manager.get_cores_per_chiplet()
        self.capacity       = self.sys_manager.get_core_capacity() * self.sys_manager.get_cores_per_chiplet()
        self.c_to_c_bw      = self.sys_manager.get_network_core_to_core_bandwidth() * self.sys_manager.get_cores_per_chiplet()

        print("Pistil Compute Chiplet Config")
        print("\tCompute:\t%8.2f TFLOPs" % (self.compute_ops/(10**12)))
        print("\tMem BW:\t\t%8.2f GB/s" % (self.mem_bw/(10**9)))
        print("\tMem Cap:\t%8.2f GB" % (self.capacity/(1024**3)))
        print("\tC-to-C BW:\t%8.2f GB/s" % (self.c_to_c_bw/(10**9)))
        self.print_row()

    def create_model_shard(self):
        # garden/pistil is now the coupled llm manager
        if self.llm_manager.model_type == "dense":
            self.model_shard = MyLLMShard(  self, 
                                        self.llm_manager.model_dim, 
                                        self.llm_manager.num_layers, 
                                        self.llm_manager.ff_dim, 
                                        self.llm_manager.head_dim, 
                                        self.llm_manager.num_heads, 
                                        self.llm_manager.kv_heads, 
                                        self.llm_manager.max_seq_len, 
                                        self.llm_manager.rope_theta, 
                                        self.llm_manager.rms_norm_eps, 
                                        self.llm_manager.vocab_size, 
                                        self.llm_manager.act_fn,
                                        self.llm_manager.attn_softcap, 
                                        self.llm_manager.q_pre_attn_scalar, 
                                        self.llm_manager.my_model.dtype,
                                    )

        elif self.llm_manager.model_type == "moe":
            self.model_shard = MyLLMShardMoE(  self, 
                                        self.llm_manager.model_dim, 
                                        self.llm_manager.num_layers, 
                                        self.llm_manager.ff_dim_moe, 
                                        self.llm_manager.ff_dim_mlp, 
                                        self.llm_manager.num_local_experts, 
                                        self.llm_manager.num_experts_per_tok, 
                                        self.llm_manager.moe_layers, 
                                        self.llm_manager.no_rope_layers, 
                                        self.llm_manager.head_dim, 
                                        self.llm_manager.num_heads, 
                                        self.llm_manager.kv_heads, 
                                        self.llm_manager.max_seq_len, 
                                        self.llm_manager.attention_chunk_size, 
                                        self.llm_manager.rope_theta, 
                                        self.llm_manager.rms_norm_eps, 
                                        self.llm_manager.vocab_size, 
                                        self.llm_manager.act_fn,
                                        self.llm_manager.attn_softcap, 
                                        self.llm_manager.q_pre_attn_scalar, 
                                        self.llm_manager.my_model.dtype,
                                    )

        self.base_model = self.llm_manager.my_model 
        self.pistil_nic = PistilNIC(self) 
    

    #################################################################################
    # forward functions
    #################################################################################
    def emb_table_forward(self, input_ids):
        self.act_shard                  = self.model_shard.emb_table.forward(input_ids)
        self.add_input_buf              = self.act_shard.clone()

        if self.pistil_transactor != None:
            num_params = self.act_shard.size()[-1]
            instruction = {"name":"emb", "layer": -1, "bytes":num_params*self.pistil_transactor.mem_w_dtype}
            #self.pistil_transactor.mem_instr.append(instruction)

            # also fetch the rotary embeddings here
            wK_shards = self.model_shard.all_layers[0].wK.shards
            num_rot_params = 0
            for head, (ss, se), (es, ee) in wK_shards[1]:
                num_rot_params += (se-ss) + (ee-es)
            self.num_rot_mult = 2 * num_rot_params * (self.llm_manager.gqa+1) # 2 for sin and cos
            num_rot_params = self.llm_manager.head_dim if num_rot_params > self.llm_manager.head_dim else num_rot_params # means multiple heads are on-chip sharing same rot-emb
            num_rot_params = num_rot_params * 2 
            instruction = {"name": "rot_emb", "layer": -1, "bytes": num_rot_params*self.pistil_transactor.mem_kv_dtype}
            #self.pistil_transactor.mem_instr.append(instruction)

    def get_layer(self, layer_id, layer_name):
        if layer_id == -1:
            if layer_name == "model_norm":
                layer = self.model_shard.model_norm
            elif layer_name == "lm_head":
                layer = self.model_shard.lm_head
        else:
            layer = eval(f"self.model_shard.all_layers[{layer_id}].{layer_name}")
        return layer

    def norm_forward_weights(self, layer_id, layer_name):
        layer_norm                      = self.get_layer(layer_id, layer_name)
        variance_shard, hidden_shard    = layer_norm.forward_weights(self.act_shard)

        self.act_shard                  = hidden_shard
        self.singleton_buf              = variance_shard.clone() # singleton buf used for accumulating the variance

        if self.pistil_transactor != None:
            num_params = layer_norm.size()[-1]
            instruction = {"name": layer_name, "layer": layer_id, "bytes":num_params*self.pistil_transactor.mem_norm_dtype}
            self.pistil_transactor.mem_instr.append(instruction)
            instruction = {"name": layer_name, "layer": layer_id, "op": "act_fn", "batch_size": self.args.sim_batch_size, "params":num_params, "fn": "pow2"}
            self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": layer_name, "layer": layer_id, "op": "vvm", "batch_size": self.args.sim_batch_size, "params":num_params}
            self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": "flag-norm", "layer": layer_id, "op": "flag", "flag": "compute-norm-weights"}
            self.pistil_transactor.comp_instr.append(instruction)

    def quantize_accum_to_act_shard(self): # does not quantize yet, but will at some point
        self.act_shard                  = self.accum_shard
    
    # use the act_shard
    def norm_forward_var(self, layer_id, layer_name, fused=True):
        layer                           = self.get_layer(layer_id, layer_name)
        hidden_shard                    = layer.forward_var(self.act_shard, self.singleton_buf)
        self.act_shard                  = hidden_shard

        if not fused and self.pistil_transactor != None:
            num_params = self.act_shard.size()[-1]
            instruction = {"name": "flag-norm", "layer": layer_id, "op": "flag", "flag": "network-norm-var"}
            self.pistil_transactor.net_instr.append(instruction)
            instruction = {"name": "wait-norm", "layer": layer_id, "op": "wait", "flag": "network-norm-var"}
            self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": layer_name, "layer": layer_id, "op": "vmult", "batch_size": self.args.sim_batch_size, "params":num_params}
            self.pistil_transactor.comp_instr.append(instruction)
    
    def vmult(self, layer_id, layer_name):
        if self.pistil_transactor != None:
            num_params = self.act_shard.size()[-1]
            instruction = {"name": layer_name, "layer": layer_id, "op": "vmult", "batch_size": self.args.sim_batch_size, "params":num_params}
            self.pistil_transactor.comp_instr.append(instruction)

    def linear_forward(self, layer_id, layer_name, flag_swing=True):
        linear_layer = self.get_layer(layer_id, layer_name)
        self.accum_shard = linear_layer.forward(self.act_shard)

        if self.pistil_transactor != None:
            out_dim, in_dim = linear_layer.size()
            if layer_name == "lm_head":
                out_dim = math.ceil(out_dim * self.args.sim_vocab_size / self.llm_manager.vocab_size)
            num_params = in_dim * out_dim
            instruction = {"name": layer_name, "layer": layer_id, "bytes":num_params*self.pistil_transactor.mem_w_dtype}
            self.pistil_transactor.mem_instr.append(instruction)
            if flag_swing:
                instruction = {"name": "flag-linear", "layer": layer_id, "op": "flag", "flag": "compute-swing"}
                self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": layer_name, "layer": layer_id, "op": "vmm", "batch_size": self.args.sim_batch_size, "in_dim":in_dim, "out_dim":out_dim, "w_params": in_dim * out_dim}
            self.pistil_transactor.comp_instr.append(instruction)

            # reduction instruction for core-to-core reduction on-chip
            instruction = {"name": layer_name, "layer": layer_id, "op": "reduction", "batch_size": self.args.sim_batch_size, "out_dim":out_dim}
            self.pistil_transactor.comp_instr.append(instruction)

    def save_shard(self, layer_id):
        self.act_shard_saved = self.act_shard.clone()

        # if self.pistil_transactor != None:
            # num_params = self.act_shard.size()[-1]
            # instruction = {"name": "save", "layer": layer_id, "op": "save", "batch_size": self.args.sim_batch_size, "params":num_params}
            # self.pistil_transactor.comp_instr.append(instruction)
    
    def load_shard(self, layer_id):
        self.act_shard = self.act_shard_saved

        if self.pistil_transactor != None:
            # set the act-shard on buffer
            num_params = self.act_shard.size()[-1]
            instruction = {"name": "load", "layer": layer_id, "op": "load", "batch_size": self.args.sim_batch_size, "params":num_params, "dtype": 2} # on-chip-dtype
            self.pistil_transactor.comp_instr.append(instruction)

    def add_input(self, layer_id):
        self.act_shard += self.add_input_buf 
        self.add_input_buf = self.act_shard.clone()

        if self.pistil_transactor != None:
            num_params = self.act_shard.size()[-1]
            instruction = {"name": "add", "layer": layer_id, "op": "vva", "batch_size": self.args.sim_batch_size, "params":num_params}
            self.pistil_transactor.comp_instr.append(instruction)
    
    def gate_activation_sum(self, layer_id):
        half_id = int(self.act_shard.size()[-1]/2)
        
        if self.llm_manager.act_fn == "silu":
            self.act_shard = torch.nn.SiLU()(self.act_shard[:, :, 0:half_id]) * self.act_shard[:, :, half_id:]
        elif self.llm_manager.act_fn == "gelu_pytorch_tanh":
            self.act_shard  = torch.nn.GELU(approximate="tanh")(self.act_shard[:, :, 0:half_id]) * self.act_shard[:, :, half_id:]
        else:
            print("Error: Unknown Activation Function: %s" % self.llm_manager.act_fn)
            exit()

        if self.pistil_transactor != None:
            num_params = self.act_shard[:, :, 0:half_id].size()[-1]
            instruction = {"name": "act", "layer": layer_id, "op": "act_fn", "batch_size": self.args.sim_batch_size, "params":num_params, "fn": self.llm_manager.act_fn}
            self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": "mult-gate-up", "layer": layer_id, "op": "vmult", "batch_size": self.args.sim_batch_size, "params":num_params }
            self.pistil_transactor.comp_instr.append(instruction)
    
    def lm_head_forward_max(self):
        self.max_scores, self.max_indices = torch.max(self.act_shard, dim=-1)
        self.max_indices += self.model_shard.lm_head.shards[1][0] # add offset for correct vocab mapping

    def find_max_score(self, num_chiplets):
        for dest in range(num_chiplets):
            if dest != self.chiplet_id:
                self.max_scores = torch.cat((self.max_scores, self.garden.chiplet_garden[dest].max_scores), dim=-1)
                self.max_indices = torch.cat((self.max_indices, self.garden.chiplet_garden[dest].max_indices), dim=-1)
        
        max_scores, max_val_indices = torch.max(self.max_scores, dim=-1)
        real_indices = self.max_indices[:, max_val_indices]
        return real_indices

    ###################################################################################
    # attention specific kernels
    ###################################################################################

    def apply_rotary_embeddings(self, layer_id):
        wQ_shards = self.model_shard.all_layers[layer_id].wQ.shards
        wK_shards = self.model_shard.all_layers[layer_id].wK.shards
        self.shard_q, self.shard_k, self.shard_v, self.cached_ids = self.model_shard.all_layers[layer_id].rotary_emb.forward_shard(self.act_shard, wQ_shards, wK_shards)
        if self.pistil_transactor != None:
            instruction = {"name": "rot_emb_mult", "layer": layer_id, "op": "vvm", "batch_size": self.args.sim_batch_size, "params": self.num_rot_mult}
            self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": "rot_emb_add", "layer": layer_id, "op": "vva", "batch_size": self.args.sim_batch_size, "params": int(self.num_rot_mult/2)}
            self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": "flag-qkv", "layer": layer_id, "op": "flag", "flag": "compute-qkv-shuffle"}
            self.pistil_transactor.comp_instr.append(instruction)

    def write_kv_cache(self, layer_id):
        seq_len = 1
        for head_id in self.pistil_nic.k_cache_write_back:
            kv_dest_chiplet_id = self.model_shard.all_layers[layer_id].kv_cache.shards_inv[head_id][self.cached_ids]
            if kv_dest_chiplet_id == self.chiplet_id:
                self.model_shard.all_layers[layer_id].kv_cache.write_kv_shard(head_id, seq_len, self.pistil_nic.k_cache_write_back[head_id], self.pistil_nic.v_cache_write_back[head_id])
        
        # compute for QKT
        if self.args.sim_prefill_batch:
            real_batch_size = self.args.sim_batch_size - self.args.sim_prefill_chunk_size + 1
        else:
            real_batch_size = self.args.sim_batch_size 
        
        if self.pistil_transactor != None:
            for bs in range(real_batch_size):
                num_kv_heads = 0
                for head in self.pistil_nic.k_cache_write_back:
                    num_kv_heads += 1
                    pistil_id = 0
                    all_seq_len = np.array(self.garden.ws_kv_cache[pistil_id][1][head])
                    if bs == 0 and self.args.sim_prefill_batch:
                        wb_seq = len(np.where(all_seq_len < self.args.sim_prefill_chunk_size)[0])
                    else:
                        wb_seq = 1

                    num_params = 2 * self.llm_manager.head_dim * wb_seq
                    
                    instruction = {"name": "write-kv", "layer": layer_id, "bytes":num_params*self.pistil_transactor.mem_kv_dtype}
                    self.pistil_transactor.mem_instr.append(instruction)
                    
                    instruction = {"name": "flag-mem", "op": "flag"}
                    self.pistil_transactor.comp_instr.append(instruction)


    def calc_QKT(self, layer_id):
        self.scores, self.max_scores = self.model_shard.all_layers[layer_id].kv_cache.forward_QKT_shard(self.pistil_nic.q_input_broadcast)
        if self.llm_manager.attn_softcap != None:
            for kv_head in self.scores:
                self.scores[kv_head] = self.scores[kv_head] / self.llm_manager.attn_softcap
                self.scores[kv_head] = torch.tanh(self.scores[kv_head])
                self.scores[kv_head] = self.scores[kv_head] * self.llm_manager.attn_softcap
        
        if self.pistil_transactor != None:
            # memory access to kv$
            if self.args.sim_prefill_batch:
                real_batch_size = self.args.sim_batch_size - self.args.sim_prefill_chunk_size + 1
            else:
                real_batch_size = self.args.sim_batch_size 

            for bs in range(real_batch_size):
                all_kv_heads = list(self.scores.keys())
                for i, kv_head in enumerate(all_kv_heads):
                    _, _, q_heads, seq_len = self.scores[kv_head].size()
                    if bs == 0 and self.args.sim_prefill_batch:
                        seq_len_sim = math.ceil(seq_len / self.args.kv_cache * (self.args.sim_prefill_cached + self.args.sim_prefill_chunk_size)) # prefill has different cached size than decode potentially... 
                    else:
                        seq_len_sim = math.ceil(seq_len / self.args.kv_cache * self.args.sim_kv_cache)
                    num_params = self.llm_manager.head_dim * seq_len_sim
                    instruction = {"name": "QKT-" + str(kv_head), "layer": layer_id, "bytes":num_params*self.pistil_transactor.mem_kv_dtype}
                    self.pistil_transactor.mem_instr.append(instruction)
            
            # flags for shuffling qkv
            instruction = {"name": "flag-qkv", "layer": layer_id, "op": "flag", "flag": "network-qkv-shuffle"}
            self.pistil_transactor.net_instr.append(instruction)
            instruction = {"name": "wait-qkv", "layer": layer_id, "op": "wait", "flag": "network-qkv-shuffle"} 
            self.pistil_transactor.comp_instr.append(instruction)

            # compute for QKT
            if self.args.sim_prefill_batch:
                real_batch_size = self.args.sim_batch_size - self.args.sim_prefill_chunk_size + 1
            else:
                real_batch_size = self.args.sim_batch_size 

            for bs in range(real_batch_size):
                all_kv_heads = list(self.scores.keys())
                for i, kv_head in enumerate(all_kv_heads): 
                    _, _, q_heads, seq_len = self.scores[kv_head].size()
                    if bs == 0 and self.args.sim_prefill_batch:
                        seq_len_sim = math.ceil(seq_len / self.args.kv_cache * (self.args.sim_prefill_cached + self.args.sim_prefill_chunk_size)) # prefill has different cached size than decode potentially... 
                        num_tokens = self.args.sim_prefill_chunk_size
                    else:
                        seq_len_sim = math.ceil(seq_len / self.args.kv_cache * self.args.sim_kv_cache)
                        num_tokens = 1

                    instruction = {"name": "QKT-" + str(kv_head), "layer": layer_id, "op": "vmm", "q_heads":q_heads, "tokens": num_tokens, "in_dim":self.llm_manager.head_dim, "out_dim":seq_len_sim, "w_params": self.llm_manager.head_dim * seq_len_sim, "clear": True}
                    self.pistil_transactor.comp_instr.append(instruction)
                    if self.llm_manager.attn_softcap != None:
                        instruction = {"name": "softcap_down-" + str(kv_head), "layer": layer_id, "op": "vmult", "params":seq_len_sim*q_heads}
                        self.pistil_transactor.comp_instr.append(instruction)
                        instruction = {"name": "softcap_tanh-" + str(kv_head), "layer": layer_id, "op": "act_fn", "params":seq_len_sim*q_heads, "fn": "tanh"}
                        self.pistil_transactor.comp_instr.append(instruction)
                        instruction = {"name": "softcap_up-" + str(kv_head), "layer": layer_id, "op": "vmult", "params":seq_len_sim*q_heads}
                        self.pistil_transactor.comp_instr.append(instruction)


    def calc_partial_softmax(self, layer_id):
        self.soft_scores, self.exp_sum = self.model_shard.all_layers[layer_id].kv_cache.forward_softmax_shard(self.scores, self.max_scores)
        
        if self.pistil_transactor != None:
            instruction = {"name": "flag-scores-max", "layer": layer_id, "op": "flag", "flag": "compute-scores-max"}
            self.pistil_transactor.comp_instr.append(instruction)

            instruction = {"name": "flag-scores-max", "layer": layer_id, "op": "flag", "flag": "network-scores-max"}
            self.pistil_transactor.net_instr.append(instruction)
            instruction = {"name": "wait-scores-max", "layer": layer_id, "op": "wait", "flag": "network-scores-max"}
            self.pistil_transactor.comp_instr.append(instruction)


        if self.pistil_transactor != None:
            all_kv_heads = list(self.soft_scores.keys())
            for i, kv_head in enumerate(all_kv_heads):
                _, _, q_heads, seq_len = self.scores[kv_head].size()
                for qh in range(q_heads):
                    instruction = {"name": "sQKT_sub_max-" + str(kv_head), "layer": layer_id, "op": "vadd", "batch_size": self.args.sim_batch_size, "params": seq_len} # subtract max-scores
                    self.pistil_transactor.comp_instr.append(instruction)
                    instruction = {"name": "sQKT_exp-" + str(kv_head), "layer": layer_id, "op": "act_fn", "batch_size": self.args.sim_batch_size, "params": seq_len, "fn": "exp"}
                    self.pistil_transactor.comp_instr.append(instruction)

    def finish_softmax(self, layer_id):
        for head in self.soft_scores:
            self.soft_scores[head] /= self.exp_sum[head].unsqueeze(-1)
            self.soft_scores[head] = self.soft_scores[head].to(self.llm_manager.my_model.dtype)
            
            if self.pistil_transactor != None:
                instruction = {"name": "flag-exp-div", "layer": layer_id, "op": "flag", "flag": "compute-exp-sum"}
                self.pistil_transactor.comp_instr.append(instruction)

                instruction = {"name": "flag-exp-div", "layer": layer_id, "op": "flag", "flag": "network-exp-sum"}
                self.pistil_transactor.net_instr.append(instruction)
                instruction = {"name": "wait-exp-div", "layer": layer_id, "op": "wait", "flag": "network-exp-sum"}
                self.pistil_transactor.comp_instr.append(instruction)

            if self.pistil_transactor != None:
                _, _, q_heads, seq_len = self.soft_scores[head].size()
                for qh in range(q_heads):
                    instruction = {"name": "sQKT_exp_div-" + str(head), "layer": layer_id, "op": "vmult", "batch_size": self.args.sim_batch_size, "params":seq_len}
                    self.pistil_transactor.comp_instr.append(instruction)

    def calc_sQKT_V(self, layer_id):
        batch_size, seq_len, _ = self.shard_q.size()
        self.sdpa_p_sum = self.model_shard.all_layers[layer_id].kv_cache.forward_sQKT_V_shard(self.soft_scores)
        
        if self.pistil_transactor != None:
            if self.args.sim_prefill_batch:
                real_batch_size = self.args.sim_batch_size - self.args.sim_prefill_chunk_size + 1
            else:
                real_batch_size = self.args.sim_batch_size 

            for bs in range(real_batch_size):
                all_kv_heads = list(self.sdpa_p_sum.keys())
                for i, kv_head in enumerate(all_kv_heads):
                    _, _, q_heads, sl = self.soft_scores[kv_head].size()
                    if bs == 0 and self.args.sim_prefill_batch:
                        seq_len_sim = math.ceil(sl / self.args.kv_cache * (self.args.sim_prefill_cached + self.args.sim_prefill_chunk_size)) # prefill has different cached size than decode potentially... 
                        num_tokens = self.args.sim_prefill_chunk_size
                    else:
                        seq_len_sim = math.ceil(sl / self.args.kv_cache * self.args.sim_kv_cache)
                        num_tokens = 1

                    num_params = self.llm_manager.head_dim * seq_len_sim
                    instruction = {"name": "sQKT_V-" + str(kv_head), "layer": layer_id, "bytes":num_params*self.pistil_transactor.mem_kv_dtype}
                    self.pistil_transactor.mem_instr.append(instruction)
                    
                    instruction = {"name": "sQKT_V-" + str(kv_head), "layer": layer_id, "op": "vmm", "q_heads": q_heads, "tokens": num_tokens, "in_dim": seq_len_sim, "out_dim": self.llm_manager.head_dim, "w_params": seq_len_sim * self.llm_manager.head_dim, "clear": True}
                    self.pistil_transactor.comp_instr.append(instruction)
            
        if self.pistil_transactor != None:
            instruction = {"name": "flag-q-reduction", "layer": layer_id, "op": "flag", "flag": "compute-q-reduction"}
            self.pistil_transactor.comp_instr.append(instruction)

            instruction = {"name": "wait-q-reduction", "layer": layer_id, "op": "wait", "flag": "network-q"}
            self.pistil_transactor.comp_instr.append(instruction)
        
        # flatten tensor along head dimension
        spda_p_sum_flat = None
        for kv_head in self.sdpa_p_sum.keys():
            if spda_p_sum_flat == None:
                spda_p_sum_flat = self.sdpa_p_sum[kv_head].view(batch_size, seq_len, -1)
            else:
                spda_p_sum_flat = torch.cat((spda_p_sum_flat, self.sdpa_p_sum[kv_head].view(batch_size, seq_len, -1)), dim=-1)        
        self.spda_p_sum_flat = spda_p_sum_flat

    def cal_sigmoid(self, layer_id):
        num_params = self.act_shard.size()[-1]

        if self.pistil_transactor != None:
            # network is waiting for compute to signal the expert_proj VMM is complete to swing the input around
            instruction = {"name": "flag-linear", "layer": layer_id, "op": "flag", "flag": "compute-swing"}
            self.pistil_transactor.comp_instr.append(instruction)

            # once network swings all values around, signal that it's done
            instruction = {"name": "flag-exp-div", "layer": layer_id, "op": "flag", "flag": "network-exp-sum"}
            self.pistil_transactor.net_instr.append(instruction)

            # compute waits until all values swing
            instruction = {"name": "wait-exp-div", "layer": layer_id, "op": "wait", "flag": "network-exp-sum"}
            self.pistil_transactor.comp_instr.append(instruction)

            # compute can complete the sigmoid
            instruction = {"name": "exp_proj-sigmoid", "layer": layer_id, "op": "act_fn", "batch_size": self.args.sim_batch_size, "params": num_params, "fn": "exp"}
            self.pistil_transactor.comp_instr.append(instruction)
            instruction = {"name": "exp_proj-div", "layer": layer_id, "op": "vmult", "batch_size": self.args.sim_batch_size, "params":num_params}
            self.pistil_transactor.comp_instr.append(instruction)


    def compute_flag_memory_experts(self, layer_id):
        if self.pistil_transactor != None:
            instruction = {"name": "flag-experts-ready", "layer": layer_id, "op": "flag", "flag": "memory-experts-ready"}
            self.pistil_transactor.comp_instr.append(instruction)
        
    def memory_wait_experts(self, layer_id):
        if self.pistil_transactor != None:
            instruction = {"name": "wait-experts-ready", "layer": layer_id, "op": "wait", "flag": "memory-experts-ready"}
            self.pistil_transactor.mem_instr.append(instruction)


    