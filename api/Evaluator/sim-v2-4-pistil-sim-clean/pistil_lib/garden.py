import math
import time
import torch
import numpy as np

from pistil_lib.pistil import Pistil
from pistil_lib.shard_model import ShardModel
from pistil_lib.lib.moe_expert_dist import MoEDist


class Garden:
    def __init__(self, args, sys_manager=None, llm_manager=None, ignore_cap=True, verbose=1):
        self.args           = args
        self.sys_manager    = sys_manager
        
        self.num_chiplets   = self.sys_manager.get_num_chiplets()
        self.num_cores      = self.sys_manager.get_cores_per_chiplet() * self.num_chiplets

        self.llm_manager    = llm_manager
        self.save_state     = llm_manager.my_model.save_state
        self.sm             = ShardModel(garden=self, llm_manager=llm_manager)
        self.EXPERT_PROJ_CHIPLETS = 8 # Always shard the expert projection across 8 CUs - max guaranteed configuration... could be configurated to fewer...

        self.validate_pistil_intermediates = args.validate_pistil_intermediates
        self.verbose        = verbose
        
        if llm_manager.model_type == "moe":
            moe_dist = MoEDist(self.args.sim_batch_size, self.llm_manager.num_local_experts)
            self.avg_experts_routed         = moe_dist.avg_experts_routed
            self.avg_routed_distribution    = moe_dist.avg_routed_distribution

        self.chiplet_garden   = []
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden.append(Pistil(garden=self, chiplet_id=chiplet_id, sys_manager=sys_manager, llm_manager=llm_manager, verbose=verbose))

        if self.verbose > 0:
            self.chiplet_garden[0].print_config()

            # collect system parameters 
            self.sys_compute    = self.chiplet_garden[0].compute_ops * self.num_chiplets
            self.sys_mem_bw     = self.chiplet_garden[0].mem_bw * self.num_chiplets
            self.sys_capacity   = self.chiplet_garden[0].capacity * self.num_chiplets
            self.sys_bi_bw      = 2 * self.chiplet_garden[0].c_to_c_bw
        
            self.print_config()
                
        if self.sys_capacity <= self.llm_manager.total_capacity:
            self.print_row()
            print("Warning: Model does not fit in Sys Memory")
            print("\tSys Capacity:\t%8.2f GB" % (self.sys_capacity/1024**3))
            print("\tModel Capacity:\t%8.2f GB" % (self.llm_manager.total_capacity / (1024**3)))
            if ignore_cap:
                print("\tIgnoring Capacity Problem for Functional Simulation")
            else:
                exit()
            self.print_row()

    def print_row(self):
        print("########################################")
    
    def print_config(self):
        self.print_row()
        print("Pistil Garden Config")
        print("\tNum SiPs:\t%8.0f (Ring)" % (self.num_chiplets))
        print("\tSys Compute:\t%8.2f TOPs" % (self.sys_compute/(10**12)))
        print("\tSys Mem BW:\t%8.2f GB/s" % (self.sys_mem_bw/(10**9)))
        print("\tSys Mem Cap:\t%8.2f GB" % (self.sys_capacity/(10**9)))
        print("\tSys C-to-C BW:\t%8.2f MB/s" % (self.sys_bi_bw/(10**9)))
        self.print_row()
    
    def print_sharding(self, max_chiplets=4):
        for chiplet_id in range(min(self.num_chiplets, max_chiplets)):
            qkv_cols = 0
            for head, (ss, se), (es, ee) in self.ws_q_proj[chiplet_id][1]:
                qkv_cols += (se - ss) + (ee - es)
            for head, (ss, se), (es, ee) in self.ws_k_proj[chiplet_id][1]:
                qkv_cols += 2*((se - ss) + (ee - es))
            
            qk_scores = 0
            for head in self.ws_kv_cache[chiplet_id][1].keys():
                qk_scores += len(self.ws_kv_cache[chiplet_id][1][head])

            o_cols = self.ws_o_proj[chiplet_id][1][1] - self.ws_o_proj[chiplet_id][1][0]
            gate_up_cols = self.ws_gate_proj[chiplet_id][1][1] - self.ws_gate_proj[chiplet_id][1][0] + self.ws_up_proj[chiplet_id][1][1] - self.ws_up_proj[chiplet_id][1][0]
            down_cols = self.ws_down_proj[chiplet_id][1][1] - self.ws_down_proj[chiplet_id][1][0]

            act_size_model_dim = self.ws_input_model_dim[chiplet_id][1][1] - self.ws_input_model_dim[chiplet_id][1][0]
            act_size_ff_dim = self.ws_input_ff_dim[chiplet_id][1][1] - self.ws_input_ff_dim[chiplet_id][1][0]

            print("Chiplet ID: %i" % chiplet_id)
            print("\twQKV Cols:\t", qkv_cols)
            print("\tQK Scores:\t", qk_scores)
            print("\twO Cols:\t", o_cols)
            print("\twGate wUP Cols:\t", gate_up_cols)
            print("\twDown Cols:\t", down_cols)
            print("\taModel Dim:\t", act_size_model_dim)
            print("\taFF Dim:\t", act_size_ff_dim)

            print("\tQ-Head Shard:\t", self.ws_q_proj[chiplet_id])
            print("\tK-Head Shard:\t", self.ws_k_proj[chiplet_id])
            print("\tV-Head Shard:\t", self.ws_v_proj[chiplet_id])
            print("\tKV-Cache Heads:\t", self.ws_kv_cache[chiplet_id][1].keys())
            print("\tQKV-Alg:\t", self.qkv_nic_config_algorithm[chiplet_id])
            print("\tO-Alg:\t\t", self.o_nic_config_algorithm[chiplet_id])

    def shard_model_dense(self, input_cuts=1):
        print("Sharding Model")
        t1 = time.perf_counter_ns()

        # shard work for input cuts
        ws_embedding, _ = self.sm.shard_layer_input(self.llm_manager.vocab_size, self.llm_manager.model_dim, input_cuts=input_cuts)
        ws_input_model_dim, row_model_dim = self.sm.shard_layer_input(1, self.llm_manager.model_dim, input_cuts=input_cuts)
        ws_input_ff_dim, row_ff_dim = self.sm.shard_layer_input(1, self.llm_manager.ff_dim, input_cuts=input_cuts)

        # shard wQKV - head and rotary aware
        ws_q_proj, ws_k_proj, ws_v_proj = self.sm.shard_wqkv(self.llm_manager.model_dim, self.llm_manager.head_dim, self.llm_manager.num_heads, self.llm_manager.kv_heads, input_cuts=input_cuts)
        head_dest, head_receive_order, qkv_nic_config_algorithm, row_order_cache, o_nic_config_algorithm, ws_input_o_proj, row_order_wo = self.sm.find_dest_by_head(ws_q_proj, ws_k_proj, ws_input_model_dim)

        # shard KV$ - head and max_seq_len aware
        ws_kv_cache, work_kv_cache_inv = self.sm.shard_attention(self.llm_manager.head_dim, self.llm_manager.kv_heads, self.llm_manager.max_seq_len, ws_k_proj, input_cuts=1)

        # shard wO 
        ws_o_proj = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.model_dim, input_cuts=input_cuts)

        # shard combined gate/up proj
        ws_gate_proj = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.ff_dim, input_cuts=input_cuts)
        ws_up_proj = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.ff_dim, input_cuts=input_cuts)

        # shard down proj
        ws_down_proj = self.sm.shard_linear(self.llm_manager.ff_dim, self.llm_manager.model_dim, input_cuts=input_cuts)
    
        # shard language_model head layer
        ws_lm_head = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.vocab_size, input_cuts=1)

        # save sharding row/column breakdown by chiplet
        self.ws_embedding               = ws_embedding
        self.ws_q_proj                  = ws_q_proj
        self.ws_k_proj                  = ws_k_proj
        self.ws_v_proj                  = ws_v_proj
        self.ws_kv_cache                = ws_kv_cache

        self.work_kv_cache_inv          = work_kv_cache_inv
        self.ws_o_proj                  = ws_o_proj
        self.ws_input_model_dim         = ws_input_model_dim
        self.ws_input_ff_dim            = ws_input_ff_dim
        self.ws_gate_proj               = ws_gate_proj
        self.ws_up_proj                 = ws_up_proj
        self.ws_down_proj               = ws_down_proj
        self.ws_lm_head                 = ws_lm_head

        # row orderings after swinging input
        self.row_model_dim              = row_model_dim
        self.row_ff_dim                 = row_ff_dim
        self.row_order_cache            = row_order_cache
        self.row_order_wo               = row_order_wo

        self.head_receive_order         = head_receive_order
        self.qkv_nic_config_algorithm   = qkv_nic_config_algorithm
        self.o_nic_config_algorithm     = o_nic_config_algorithm
        self.ws_input_o_proj            = ws_input_o_proj

        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].create_model_shard()

        t2 = time.perf_counter_ns()
        print("Time to Shard Model: %0.2fs" % ((t2 - t1) / 10**9))

        # self.print_sharding()
    
    def shard_model_moe(self, input_cuts=1):
        print("Sharding Model")
        t1 = time.perf_counter_ns()

        # shard work for input cuts
        ws_embedding, _ = self.sm.shard_layer_input(self.llm_manager.vocab_size, self.llm_manager.model_dim, input_cuts=input_cuts)
        ws_input_model_dim, row_model_dim = self.sm.shard_layer_input(1, self.llm_manager.model_dim, input_cuts=input_cuts)
        ws_input_ff_dim_moe, row_ff_dim_moe = self.sm.shard_layer_input(1, self.llm_manager.ff_dim_moe, input_cuts=input_cuts)
        ws_input_ff_dim_mlp, row_ff_dim_mlp = self.sm.shard_layer_input(1, self.llm_manager.ff_dim_mlp, input_cuts=input_cuts)

        # shard wQKV - head and rotary aware
        ws_q_proj, ws_k_proj, ws_v_proj = self.sm.shard_wqkv(self.llm_manager.model_dim, self.llm_manager.head_dim, self.llm_manager.num_heads, self.llm_manager.kv_heads, input_cuts=input_cuts)
        head_dest, head_receive_order, qkv_nic_config_algorithm, row_order_cache, o_nic_config_algorithm, ws_input_o_proj, row_order_wo = self.sm.find_dest_by_head(ws_q_proj, ws_k_proj, ws_input_model_dim)

        # shard KV$ - head and max_seq_len aware
        ws_kv_cache, work_kv_cache_inv = self.sm.shard_attention(self.llm_manager.head_dim, self.llm_manager.kv_heads, self.llm_manager.max_seq_len, ws_k_proj, input_cuts=1)

        # shard wO 
        ws_o_proj = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.model_dim, input_cuts=input_cuts)

        ###################################################
        # moe
        # shard combined gate/up proj
        ws_gate_proj_moe = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.ff_dim_moe, input_cuts=input_cuts)
        ws_up_proj_moe = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.ff_dim_moe, input_cuts=input_cuts)
        ws_expert_proj = self.sm.shard_expert_projection(self.llm_manager.model_dim, self.llm_manager.num_local_experts, expert_proj_chiplets=self.EXPERT_PROJ_CHIPLETS, input_cuts=input_cuts)

        # shard down proj
        ws_down_proj_moe = self.sm.shard_linear(self.llm_manager.ff_dim_moe, self.llm_manager.model_dim, input_cuts=input_cuts)
        ###################################################
    
        ###################################################
        # mlp
        # shard combined gate/up proj
        ws_gate_proj_mlp = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.ff_dim_mlp, input_cuts=input_cuts)
        ws_up_proj_mlp = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.ff_dim_mlp, input_cuts=input_cuts)

        # shard down proj
        ws_down_proj_mlp = self.sm.shard_linear(self.llm_manager.ff_dim_mlp, self.llm_manager.model_dim, input_cuts=input_cuts)
        ###################################################

        # shard language_model head layer
        ws_lm_head = self.sm.shard_linear(self.llm_manager.model_dim, self.llm_manager.vocab_size, input_cuts=1)

        # save sharding row/column breakdown by chiplet
        self.ws_embedding               = ws_embedding
        self.ws_q_proj                  = ws_q_proj
        self.ws_k_proj                  = ws_k_proj
        self.ws_v_proj                  = ws_v_proj
        self.ws_kv_cache                = ws_kv_cache
        self.work_kv_cache_inv          = work_kv_cache_inv
        self.ws_o_proj                  = ws_o_proj
        self.ws_input_model_dim         = ws_input_model_dim
        
        self.ws_input_ff_dim_moe        = ws_input_ff_dim_moe
        self.ws_gate_proj_moe           = ws_gate_proj_moe
        self.ws_up_proj_moe             = ws_up_proj_moe
        self.ws_down_proj_moe           = ws_down_proj_moe
        self.ws_expert_proj             = ws_expert_proj

        self.ws_input_ff_dim_mlp        = ws_input_ff_dim_mlp
        self.ws_gate_proj_mlp           = ws_gate_proj_mlp
        self.ws_up_proj_mlp             = ws_up_proj_mlp
        self.ws_down_proj_mlp           = ws_down_proj_mlp
        
        self.ws_lm_head                 = ws_lm_head

        # row orderings after swinging input
        self.row_model_dim              = row_model_dim
        self.row_ff_dim_moe             = row_ff_dim_moe
        self.row_ff_dim_mlp             = row_ff_dim_mlp
        self.row_order_cache            = row_order_cache
        self.row_order_wo               = row_order_wo

        self.head_receive_order         = head_receive_order
        self.qkv_nic_config_algorithm   = qkv_nic_config_algorithm
        self.o_nic_config_algorithm     = o_nic_config_algorithm
        self.ws_input_o_proj            = ws_input_o_proj

        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].create_model_shard()

        t2 = time.perf_counter_ns()
        print("Time to Shard Model: %0.2fs" % ((t2 - t1) / 10**9))

        # self.print_sharding()

    def load_model_shards_dense(self):
        print("Loading Model Shards")
        t1 = time.perf_counter_ns()

        # shard model weights using distributions above
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_loader.load_embedding_shard(self.ws_embedding[chiplet_id])
            for layer_id in range(self.args.sim_num_layers):
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_wQKV_shard(layer_id, self.ws_q_proj[chiplet_id], "wQ", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_wQKV_shard(layer_id, self.ws_k_proj[chiplet_id], "wK", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_wQKV_shard(layer_id, self.ws_v_proj[chiplet_id], "wV", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_linear_shard(layer_id, self.ws_o_proj[chiplet_id], "wO", row_ordering=self.row_order_wo[chiplet_id]) # row_order is slightly different because of how qkv is sharded 
                
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_cache_shard(layer_id, self.ws_kv_cache[chiplet_id], self.work_kv_cache_inv, row_ordering=self.row_order_cache[chiplet_id])
                
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_gate_proj[chiplet_id], "gate_proj", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_up_proj[chiplet_id], "up_proj", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_linear_shard(layer_id, self.ws_down_proj[chiplet_id], "down_proj", row_ordering=self.row_ff_dim[chiplet_id])

                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "input_norm")
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "pre_ff_norm")
                if self.llm_manager.my_model.all_layers[0].post_atten_norm.weights != None:
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "post_atten_norm")
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "post_ff_norm")

            #if self.args.sim_num_layers == -1:   
            self.chiplet_garden[chiplet_id].pistil_loader.load_model_norm_shard(self.ws_input_model_dim[chiplet_id])
            self.chiplet_garden[chiplet_id].pistil_loader.load_lm_head_shard(self.ws_lm_head[chiplet_id], row_ordering=self.row_model_dim[chiplet_id])
        
        t2 = time.perf_counter_ns()
        print("Time to Load Shards: %0.2fs" % ((t2 - t1) / 10**9))
    
    def load_model_shards_moe(self):
        print("Loading Model Shards Mixture of Experts")
        t1 = time.perf_counter_ns()

        # shard model weights using distributions above
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_loader.load_embedding_shard(self.ws_embedding[chiplet_id])
            for layer_id in range(self.args.sim_num_layers):
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_wQKV_shard(layer_id, self.ws_q_proj[chiplet_id], "wQ", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_wQKV_shard(layer_id, self.ws_k_proj[chiplet_id], "wK", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_wQKV_shard(layer_id, self.ws_v_proj[chiplet_id], "wV", row_ordering=self.row_model_dim[chiplet_id])
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_linear_shard(layer_id, self.ws_o_proj[chiplet_id], "wO", row_ordering=self.row_order_wo[chiplet_id]) # row_order is slightly different because of how qkv is sharded 
                
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_cache_shard(layer_id, self.ws_kv_cache[chiplet_id], self.work_kv_cache_inv, row_ordering=self.row_order_cache[chiplet_id])
                
                if layer_id in self.llm_manager.moe_layers:
                    # shared expert
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_gate_proj_moe[chiplet_id], "gate_proj_s", row_ordering=self.row_model_dim[chiplet_id])
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_up_proj_moe[chiplet_id], "up_proj_s", row_ordering=self.row_model_dim[chiplet_id])
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_linear_shard(layer_id, self.ws_down_proj_moe[chiplet_id], "down_proj_s", row_ordering=self.row_ff_dim_moe[chiplet_id])
                    
                    # routed expert
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_gate_proj_moe[chiplet_id], "gate_proj_e", row_ordering=self.row_model_dim[chiplet_id])
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_up_proj_moe[chiplet_id], "up_proj_e", row_ordering=self.row_model_dim[chiplet_id])
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_linear_shard(layer_id, self.ws_down_proj_moe[chiplet_id], "down_proj_e", row_ordering=self.row_ff_dim_moe[chiplet_id])
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_linear_shard(layer_id, self.ws_expert_proj[chiplet_id%self.EXPERT_PROJ_CHIPLETS], "expert_proj", row_ordering=self.row_model_dim[chiplet_id])
                else:
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_gate_proj_mlp[chiplet_id], "gate_proj", row_ordering=self.row_model_dim[chiplet_id])
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_gate_up_shard(layer_id, self.ws_up_proj_mlp[chiplet_id], "up_proj", row_ordering=self.row_model_dim[chiplet_id])
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_linear_shard(layer_id, self.ws_down_proj_mlp[chiplet_id], "down_proj", row_ordering=self.row_ff_dim_mlp[chiplet_id])

                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "input_norm")
                self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "pre_ff_norm")
                if self.llm_manager.my_model.all_layers[0].post_atten_norm.weights != None:
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "post_atten_norm")
                    self.chiplet_garden[chiplet_id].pistil_loader.load_layer_norm_shard(layer_id, self.ws_input_model_dim[chiplet_id], "post_ff_norm")

            #if self.args.sim_num_layers == -1:   
            self.chiplet_garden[chiplet_id].pistil_loader.load_model_norm_shard(self.ws_input_model_dim[chiplet_id])
            self.chiplet_garden[chiplet_id].pistil_loader.load_lm_head_shard(self.ws_lm_head[chiplet_id], row_ordering=self.row_model_dim[chiplet_id])
        
        t2 = time.perf_counter_ns()
        print("Time to Load Shards: %0.2fs" % ((t2 - t1) / 10**9))



    def emb_table_forward(self, input_ids):
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].emb_table_forward(input_ids)


    def model_head_forward(self):
        # switch the lm_head and the normal transactional pipes
        if self.pistil_transactor != None:
            self.pistil_transactor.mem_instr_tmp      = self.pistil_transactor.mem_instr
            self.pistil_transactor.comp_instr_tmp     = self.pistil_transactor.comp_instr
            self.pistil_transactor.net_instr_tmp      = self.pistil_transactor.net_instr

            self.pistil_transactor.mem_instr          = []
            self.pistil_transactor.comp_instr         = []
            self.pistil_transactor.net_instr          = []

        ##################################################################
        # start model normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_weights(-1, "model_norm") 
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="fused-norm", layer_id=-1, layer_name="model_norm+lm_head") 
        ##################################################################

        ##################################################################
        # swing data - can ovelap compute, network and memory bw here
        for _ in range(self.num_chiplets-1): # swing data around all chiplets 
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("fused-norm")
        ##################################################################

        ##################################################################
        # lm_head
        for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
            self.chiplet_garden[chiplet_id].linear_forward(-1, "lm_head")
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
        ##################################################################

        ##################################################################
        # finish normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_var(-1, "model_norm") 
        ##################################################################

        ##################################################################
        # get max id for next layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].lm_head_forward_max() 
        ##################################################################

        max_ids = self.chiplet_garden[0].find_max_score(self.num_chiplets)

        # restore the instructions to be saved correctly
        if self.pistil_transactor != None:
            self.pistil_transactor.mem_instr_lm_head  = self.pistil_transactor.mem_instr
            self.pistil_transactor.comp_instr_lm_head = self.pistil_transactor.comp_instr
            self.pistil_transactor.net_instr_lm_head  = self.pistil_transactor.net_instr

            self.pistil_transactor.mem_instr          = self.pistil_transactor.mem_instr_tmp
            self.pistil_transactor.comp_instr         = self.pistil_transactor.comp_instr_tmp
            self.pistil_transactor.net_instr          = self.pistil_transactor.net_instr_tmp

        return max_ids
        

    def attention_layer(self, layer_id):
        ##################################################################
        # start input normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_weights(layer_id, "input_norm") 
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="fused-norm", layer_id=layer_id, layer_name="input_norm+wQKV") 
        ##################################################################

        self.save_state.compare_result(layer_id, name="input_norm_weights", param=self.save_state.collect_shards(self))

        ##################################################################
        # swing data - can ovelap compute, network and memory bw here
        for _ in range(self.num_chiplets-1): # swing data around all chiplets 
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("fused-norm")
        ##################################################################


        ##################################################################
        # wQKV
        for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
            self.chiplet_garden[chiplet_id].linear_forward(layer_id, "wQKV")
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
        ##################################################################

        self.save_state.compare_result(layer_id, name="q_vector", param=self.save_state.collect_q_accum_shards(self))
        self.save_state.compare_result(layer_id, name="k_vector", param=self.save_state.collect_k_accum_shards(self))
        self.save_state.compare_result(layer_id, name="v_vector", param=self.save_state.collect_v_accum_shards(self))
        
        
        ##################################################################
        # finish normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_var(layer_id, "input_norm") 
        ##################################################################

        self.save_state.compare_result(layer_id, name="q_vector_var", param=self.save_state.collect_q_act_shards(self))
        self.save_state.compare_result(layer_id, name="k_vector_var", param=self.save_state.collect_k_act_shards(self))
        self.save_state.compare_result(layer_id, name="v_vector_var", param=self.save_state.collect_v_act_shards(self))

        ##################################################################
        # TODO: sum if using a SiP sharding strategy 
        ##################################################################
        

        ##################################################################
        # apply rotary embeddings 
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].apply_rotary_embeddings(layer_id) 
        ##################################################################

        self.save_state.compare_result(layer_id, name="q_rot", param=self.save_state.collect_q_rot_shards(self))
        self.save_state.compare_result(layer_id, name="k_rot", param=self.save_state.collect_k_rot_shards(self))
        
        

        ##################################################################
        # send KV$ shard to correct chiplet
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="kv", layer_id=layer_id, layer_name="shuffle-kv")

        SHUFFLE_COMPLETE = False
        while SHUFFLE_COMPLETE == False:
            SHUFFLE_COMPLETE = True
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_input()
            
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_decode_network(operation="kv")
                SHUFFLE_COMPLETE = SHUFFLE_COMPLETE and self.chiplet_garden[chiplet_id].pistil_nic.shuffle_complete

        # write back KV$
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].write_kv_cache(layer_id)
        ##################################################################
        

        ##################################################################
        # broadcast Q shards 
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="q", layer_id=layer_id, layer_name="shuffle-q")

        # shuffle Q shards
        SHUFFLE_COMPLETE = False
        while SHUFFLE_COMPLETE == False:
            SHUFFLE_COMPLETE = True
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_input()
            
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_decode_network(operation="q")
                SHUFFLE_COMPLETE = SHUFFLE_COMPLETE and self.chiplet_garden[chiplet_id].pistil_nic.shuffle_complete

        ##################################################################
        # Compute soft(QKT)V
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].calc_QKT(layer_id)
            
        self.save_state.compare_result(layer_id, name="scores", param=self.save_state.collect_scores(self))
        

        #-------------------------------->
        # configure sending max value around
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="max", layer_id=layer_id, layer_name="scores_max")

        # shuffle scores max values
        SHUFFLE_COMPLETE = False
        while SHUFFLE_COMPLETE == False:
            SHUFFLE_COMPLETE = True
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_input()
            
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_decode_network(operation="max")
                SHUFFLE_COMPLETE = SHUFFLE_COMPLETE and self.chiplet_garden[chiplet_id].pistil_nic.shuffle_complete
        #-------------------------------->



        # calculate partial exp scores
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].calc_partial_softmax(layer_id)


        #-------------------------------->
        # configure sending max value around
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="exp_sum", layer_id=layer_id, layer_name="exp_sum")
        
        # shuffle exp sum values
        SHUFFLE_COMPLETE = False
        while SHUFFLE_COMPLETE == False:
            SHUFFLE_COMPLETE = True
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_input()
            
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_decode_network(operation="exp_sum")
                SHUFFLE_COMPLETE = SHUFFLE_COMPLETE and self.chiplet_garden[chiplet_id].pistil_nic.shuffle_complete
        #-------------------------------->
        
        # finish softmax
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].finish_softmax(layer_id)
        
        self.save_state.compare_result(layer_id, name="soft_scores", param=self.save_state.collect_soft_scores(self))
        
        
        
        # compute partial soft(QKT)V
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].calc_sQKT_V(layer_id)


        # reduce partial sums for softmax so it aligns with wO row shard
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="sdpa_reduction", layer_id=layer_id, layer_name="sdpa-reduction")
        
        # shuffle partial sums
        SHUFFLE_COMPLETE = False
        while SHUFFLE_COMPLETE == False:
            SHUFFLE_COMPLETE = True
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_input()
            
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.shuffle_decode_network(operation="sdpa_reduction")
                SHUFFLE_COMPLETE = SHUFFLE_COMPLETE and self.chiplet_garden[chiplet_id].pistil_nic.shuffle_complete

        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_nic.move_sdpa_reduction()

        self.save_state.compare_result(layer_id, name="sQKT_V", param=self.save_state.collect_shards(self))

        # swing data - can ovelap compute, network and memory bw here
        for _ in range(self.num_chiplets-1): # swing data around all chiplets 
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network(operation="cat")

        # wO
        for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
            self.chiplet_garden[chiplet_id].linear_forward(layer_id, "wO")
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
        
        self.save_state.compare_result(layer_id, name="o_vector", param=self.save_state.collect_accum_shards(self))
        ##################################################################
        

    def moe_layer(self, layer_id):
        ##################################################################
        # start post attention normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_weights(layer_id, "pre_ff_norm")
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="fused-norm", layer_id=layer_id, layer_name="pre_ff_norm+gate_up")
        ##################################################################

        self.save_state.compare_result(layer_id, name="pre_ff_norm_weights", param=self.save_state.collect_shards(self))

        ##################################################################
        # swing data - can ovelap compute, network and memory bw here
        for _ in range(self.num_chiplets-1): # swing data around all chiplets 
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("fused-norm")
        ##################################################################

        ##########################################################################################
        # expert projection
        # 1. save the input vector 
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].save_shard(layer_id)

        # 2. expert projection
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].linear_forward(layer_id, "expert_proj", flag_swing=True)
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()

        ##################################################################
        # 3. divide by variance on the expert output - finish normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_var(layer_id, "pre_ff_norm") 
        ##################################################################


        # 4. swing the accumulated result (subset of all chiplets)
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="act-shard", layer_id=layer_id, layer_name="expert_projection")

        for _ in range(self.EXPERT_PROJ_CHIPLETS-1): # swing data around shared expert projection groups
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("cat")

        # 5. calculate sigmoid across experts
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].cal_sigmoid(layer_id)

        # 6. compute flag memory is ready to read the experts from memory
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].compute_flag_memory_experts(layer_id)
            
        # 7. load the input vector and divide by variance
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].load_shard(layer_id)
            self.chiplet_garden[chiplet_id].vmult(layer_id, "vmult") # divide the vector by the variance

        batch_size = self.args.sim_batch_size
        if self.llm_manager.num_local_experts == 128:
            if batch_size == 1:
                NUM_EXPERTS = 1
            elif batch_size == 2:
                NUM_EXPERTS = 2

        ##########################################################################################
        # shared expert
        ##################################################################
        # gate_up_proj
        for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
            self.chiplet_garden[chiplet_id].linear_forward(layer_id, "gate_up_proj_s", flag_swing=False)
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
        ##################################################################


        ##################################################################
        # silu + up_proj
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].gate_activation_sum(layer_id) 
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="act-shard", layer_id=layer_id, layer_name="down_proj_s") 

        ##################################################################


        ##################################################################
        # swing data - can ovelap compute, network and memory bw here
        for _ in range(self.num_chiplets-1): # swing data around all chiplets 
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("cat")
        ##################################################################


        ##################################################################
        # down_proj
        for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
            self.chiplet_garden[chiplet_id].linear_forward(layer_id, "down_proj_s")
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
        ##################################################################


        ##########################################################################################
        # routed expert
        ##################################################################

        # add a flag to synchronize the memory and compute to make sure that we know which expert(s) to route
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].memory_wait_experts(layer_id)

        for i in range(self.avg_experts_routed):            
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].args.sim_batch_size = self.avg_routed_distribution[i]
            
            # load the input vector
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].load_shard(layer_id)

            # gate_up_proj
            for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
                self.chiplet_garden[chiplet_id].linear_forward(layer_id, "gate_up_proj_e", flag_swing=False) # don't swing, because assume vector is already on-chip
                self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
            ##################################################################


            ##################################################################
            # silu + up_proj
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].gate_activation_sum(layer_id) 
                self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="act-shard", layer_id=layer_id, layer_name="down_proj_e") 

            ##################################################################


            ##################################################################
            # swing data - can ovelap compute, network and memory bw here
            for _ in range(self.num_chiplets-1): # swing data around all chiplets 
                for chiplet_id in range(self.num_chiplets):
                    self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
                for chiplet_id in range(self.num_chiplets):
                    self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("cat")
            ##################################################################


            ##################################################################
            # down_proj
            for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
                self.chiplet_garden[chiplet_id].linear_forward(layer_id, "down_proj_e")
                self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
            ##################################################################

            # sum the shared and routed experts
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].add_input(layer_id)
    

        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].args.sim_batch_size = self.args.sim_batch_size


    def mlp_layer(self, layer_id):
        ##################################################################
        # start post attention normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_weights(layer_id, "pre_ff_norm")
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="fused-norm", layer_id=layer_id, layer_name="pre_ff_norm+gate_up")
        ##################################################################

        self.save_state.compare_result(layer_id, name="pre_ff_norm_weights", param=self.save_state.collect_shards(self))

        ##################################################################
        # swing data - can ovelap compute, network and memory bw here
        for _ in range(self.num_chiplets-1): # swing data around all chiplets 
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("fused-norm")
        ##################################################################


        ##################################################################
        # gate_up_proj
        for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
            self.chiplet_garden[chiplet_id].linear_forward(layer_id, "gate_up_proj")
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
        ##################################################################

        self.save_state.compare_result(layer_id, name="gate", param=self.save_state.collect_accum_shards_gate(self))
        self.save_state.compare_result(layer_id, name="up", param=self.save_state.collect_accum_shards_up(self))

    
        ##################################################################
        # finish normalize layer
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].norm_forward_var(layer_id, "pre_ff_norm") 
        
        self.save_state.compare_result(layer_id, name="pre_ff_norm_var_gate", param=self.save_state.collect_gate_shards(self))
        self.save_state.compare_result(layer_id, name="pre_ff_norm_var_up", param=self.save_state.collect_up_shards(self))
        ##################################################################



        ##################################################################
        # silu + up_proj
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].gate_activation_sum(layer_id) 
            self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="act-shard", layer_id=layer_id, layer_name="down_proj") 

        ##################################################################

        self.save_state.compare_result(layer_id, name="act", param=self.save_state.collect_shards(self))

        ##################################################################
        # swing data - can ovelap compute, network and memory bw here
        for _ in range(self.num_chiplets-1): # swing data around all chiplets 
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("cat")
        ##################################################################


        ##################################################################
        # down_proj
        for chiplet_id in range(self.num_chiplets): # forward pass of linear layer
            self.chiplet_garden[chiplet_id].linear_forward(layer_id, "down_proj")
            self.chiplet_garden[chiplet_id].quantize_accum_to_act_shard()
        ##################################################################


    def layer_forward(self, layer_id):
        self.attention_layer(layer_id)

        ##################################################################
        # Gemma2 specific layer norm
        if self.chiplet_garden[0].model_shard.all_layers[0].post_atten_norm.weights != None:
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].norm_forward_weights(layer_id, "post_atten_norm")
                self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="norm", layer_id=layer_id, layer_name="post_atten_norm")

            # swing data - not ovelapping compute, network and memory bw here
            for _ in range(self.num_chiplets-1): # swing data around all chiplets 
                for chiplet_id in range(self.num_chiplets):
                    self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
                for chiplet_id in range(self.num_chiplets):
                    self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("norm")
            
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].norm_forward_var(layer_id, "post_atten_norm", fused=False) 
            self.save_state.compare_result(layer_id, name="post_atten_norm", param=self.save_state.collect_shards(self))
        ##################################################################


        ##################################################################
        # add layer input
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].add_input(layer_id)
        ##################################################################
        self.save_state.compare_result(layer_id, name="layer_sum_attention", param=self.save_state.collect_shards(self))

        if self.llm_manager.model_type == "dense":
            self.mlp_layer(layer_id)
        elif self.llm_manager.model_type == "moe":
            if layer_id in self.llm_manager.moe_layers:
                self.moe_layer(layer_id)
            else:
                self.mlp_layer(layer_id)

        
        ##################################################################
        # Gemma2 specific layer norm
        if self.chiplet_garden[0].model_shard.all_layers[0].post_ff_norm.weights != None:
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].norm_forward_weights(layer_id, "post_ff_norm")
                self.chiplet_garden[chiplet_id].pistil_nic.configure_nic(operation="norm", layer_name="post_ff_norm")

            # swing data - not ovelapping compute, network and memory bw here
            for _ in range(self.num_chiplets-1): # swing data around all chiplets 
                for chiplet_id in range(self.num_chiplets):
                    self.chiplet_garden[chiplet_id].pistil_nic.swing_input()
                for chiplet_id in range(self.num_chiplets):
                    self.chiplet_garden[chiplet_id].pistil_nic.swing_decode_network("norm")
                
            for chiplet_id in range(self.num_chiplets):
                self.chiplet_garden[chiplet_id].norm_forward_var(layer_id, "post_ff_norm", fused=False) 
        ##################################################################


        ##################################################################
        # add layer input
        for chiplet_id in range(self.num_chiplets):
            self.chiplet_garden[chiplet_id].add_input(layer_id)
        ##################################################################
        self.save_state.compare_result(layer_id, name="layer_sum_output", param=self.save_state.collect_shards(self))
        #print("layer:", layer_id, self.chiplet_garden[0].act_shard[0, 0, 0:32])


    def sim_model(self):
        with torch.no_grad():
            if self.verbose > 0:
                print("Starting Model Generation for Pistil Implementation")
                
            #input_ids = self.llm_manager.prefill_input_ids
            if self.llm_manager.from_pretrained and self.args.sim_num_layers == -1:
                input_ids = torch.tensor([self.llm_manager.my_model.generated_token_ids[0]]).unsqueeze(0)
                generated_token_ids = [self.llm_manager.my_model.generated_token_ids[0]]
            else:
                # random input_ids
                input_ids = torch.randint(low=0, high=self.llm_manager.vocab_size, size=(1, 1))
                generated_token_ids = []

            for i in range(self.llm_manager.max_new_tokens-1):
                #input_ids = torch.tensor([self.llm_manager.my_model.generated_token_ids[i]]).unsqueeze(0)
                if self.validate_pistil_intermediates:
                    if i+1 == self.llm_manager.max_new_tokens-1:
                        self.save_state.enable_compare = True
                    
                t1 = time.perf_counter_ns()
                self.emb_table_forward(input_ids)

                for layer_id in range(self.args.sim_num_layers):
                    self.layer_forward(layer_id)
                
                output = self.model_head_forward()
                if self.args.sim_num_layers == -1:
                    input_ids = output
                    if self.llm_manager.from_pretrained:
                        print("Token Output %i - %0.3fs" % (i, (t2 - t1)/10**9))
                        print("\tGenerated ID:\t%i" % output[0].item())
                        generated_token_ids.append(output[0].item())
                
                t2 = time.perf_counter_ns()


            if self.llm_manager.from_pretrained == True and self.args.sim_num_layers == -1:
                generated_text = self.llm_manager.base_tokenizer.decode(torch.tensor(generated_token_ids), skip_special_tokens=True)
                print("My LLM Conversation:\n\tPrompt:\t\t%s\n\tGenerated Text:\t%s" % (self.llm_manager.prompt[0]["content"], generated_text))
