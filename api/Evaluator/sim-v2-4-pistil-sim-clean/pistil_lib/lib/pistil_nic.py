import math
import torch

class PistilNIC:
    def __init__(self, pistil):
        self.pistil                     = pistil
        self.args                       = pistil.args
        self.pistil_transactor          = pistil.pistil_transactor
        self.garden                     = self.pistil.garden
        self.chiplet_id                 = self.pistil.chiplet_id
        self.w_chiplet_id               = self.pistil.w_chiplet_id
        self.e_chiplet_id               = self.pistil.e_chiplet_id

        self.batch_size                 = 1
        self.seq_len                    = 1
        self.head_dim                   = self.pistil.model_shard.head_dim
        self.gqa                        = self.pistil.model_shard.gqa
        self.dtype                      = self.pistil.llm_manager.my_model.dtype
        
        self.ws_q_proj                  = self.garden.ws_q_proj[self.chiplet_id]
        self.ws_k_proj                  = self.garden.ws_k_proj[self.chiplet_id]
        self.ws_v_proj                  = self.garden.ws_v_proj[self.chiplet_id]
        
        self.head_receive_order         = self.garden.head_receive_order[self.chiplet_id]
        self.all_heads_on_chiplet       = sorted(self.head_receive_order.keys())
        self.qkv_nic_config_algorithm   = self.garden.qkv_nic_config_algorithm[self.chiplet_id]
        self.o_nic_config_algorithm     = self.garden.o_nic_config_algorithm[self.chiplet_id]

        self.east_tx = {}
        self.east_rx = {}
        self.west_tx = {}
        self.west_rx = {}

    # move data from network, to on chip, and forward data
    def swing_input(self):
        # swing chiplet data west - that way the memory prefetch is in order
        self.garden.chiplet_garden[self.w_chiplet_id].pistil_nic.east_rx = self.west_tx
        self.west_tx = {}

    def shuffle_input(self):
        # edge case with only 1 Chiplet
        if self.w_chiplet_id == self.chiplet_id or self.e_chiplet_id == self.chiplet_id:
            return     

        self.garden.chiplet_garden[self.w_chiplet_id].pistil_nic.east_rx = self.west_tx 
        self.garden.chiplet_garden[self.e_chiplet_id].pistil_nic.west_rx = self.east_tx 
        self.east_tx = {}
        self.west_tx = {}

    def configure_nic(self, operation, layer_id=-1, layer_name=None):        
        self.shuffle_complete = False
        self.east_tx = {}
        self.east_rx = {}
        self.west_tx = {}
        self.west_rx = {}
        
        if operation == "act-shard":
            self.west_tx["act_shard"]   = self.pistil.act_shard.clone()
            if self.pistil_transactor != None:
                instruction = {"name": "wait-compute-swing", "layer": layer_name, "op": "wait", "flag": "compute-swing"}
                self.pistil_transactor.net_instr.append(instruction)
        if operation == "norm":
            self.west_tx["variance"]    = self.pistil.singleton_buf.clone()
            if self.pistil_transactor != None:
                instruction = {"name": "wait-norm", "layer": layer_name, "op": "wait", "flag": "compute-norm-weights"}
                self.pistil_transactor.net_instr.append(instruction)
        if operation == "fused-norm":
            self.west_tx["act_shard"]   = self.pistil.act_shard.clone()
            self.west_tx["variance"]    = self.pistil.singleton_buf.clone()
            if self.pistil_transactor != None:
                instruction = {"name": "wait-norm", "layer": layer_name, "op": "wait", "flag": "compute-norm-weights"}
                self.pistil_transactor.net_instr.append(instruction)
                instruction = {"name": "wait-compute-swing", "layer": layer_name, "op": "wait", "flag": "compute-swing"}
                self.pistil_transactor.net_instr.append(instruction)
        
        if operation == "kv":
            # get data from compute chiplet
            shard_k = self.pistil.shard_k
            shard_v = self.pistil.shard_v

            self.kv_received_vals = {}
            self.k_cache_write_back = {}
            self.v_cache_write_back = {}

            offset = 0
            for head_id, (ss, se), (es, ee) in self.ws_k_proj[1]:       
                self.k_cache_write_back[head_id]                            = torch.zeros((self.batch_size, self.seq_len, self.head_dim)).to(self.dtype)
                cols_per_chiplet                                            = ee - es + se - ss
                self.k_cache_write_back[head_id][:, :, 0:cols_per_chiplet]  = shard_k[:, :, offset:offset+cols_per_chiplet].reshape((self.batch_size, self.seq_len, cols_per_chiplet))
                self.v_cache_write_back[head_id]                            = shard_v[:, :, offset:offset+cols_per_chiplet].reshape((self.batch_size, self.seq_len, cols_per_chiplet))
                self.kv_received_vals[head_id]                              = cols_per_chiplet
                offset += cols_per_chiplet

            if self.qkv_nic_config_algorithm["send_e_vals"] != None:
                send_e_vals             = self.qkv_nic_config_algorithm["send_e_vals"] 
                self.east_tx["k_vals"]  = shard_k[:, :, -send_e_vals:].clone()
                self.east_tx["v_vals"]  = shard_v[:, :, -send_e_vals:].clone()
            
            if self.qkv_nic_config_algorithm["send_w_vals"] != None:
                send_w_vals             = self.qkv_nic_config_algorithm["send_w_vals"] 
                self.west_tx["k_vals"]  = shard_k[:, :, 0:send_w_vals].clone()
                self.west_tx["v_vals"]  = shard_v[:, :, 0:send_w_vals].clone()
            
            if self.pistil_transactor != None:
                instruction = {"name": "wait-qkv", "layer": layer_name, "op": "wait", "flag": "compute-qkv-shuffle"}
                self.pistil_transactor.net_instr.append(instruction)
        
        if operation == "q":
            shard_q = self.pistil.shard_q

            self.q_received_vals = {}
            self.q_input_broadcast = {}

            offset = 0
            for head_id, (ss, se), (es, ee) in self.ws_k_proj[1]:       
                self.q_input_broadcast[head_id]                                 = torch.zeros((self.batch_size, self.seq_len, self.gqa, self.head_dim)).to(self.dtype)
                cols_per_chiplet                                                = ee - es + se - ss
                self.q_input_broadcast[head_id][:, :, :, 0:cols_per_chiplet]    = shard_q[:, :, offset:offset+cols_per_chiplet*self.gqa].reshape((self.batch_size, self.seq_len, self.gqa, cols_per_chiplet))
                self.q_received_vals[head_id]                                   = cols_per_chiplet
                offset += cols_per_chiplet*self.gqa

            if self.qkv_nic_config_algorithm["send_e_vals"] != None:
                send_e_vals = self.qkv_nic_config_algorithm["send_e_vals"] * self.gqa
                self.east_tx["q_vals"] = shard_q[:, :, -send_e_vals:].clone()
            
            if self.qkv_nic_config_algorithm["send_w_vals"] != None:
                send_w_vals = self.qkv_nic_config_algorithm["send_w_vals"] * self.gqa
                self.west_tx["q_vals"] = shard_q[:, :, 0:send_w_vals].clone()

        if operation == "max":
            max_scores = self.pistil.max_scores

            if self.qkv_nic_config_algorithm["send_e_vals"] != None:
                self.east_tx["max_scores"] = max_scores[self.all_heads_on_chiplet[-1]].clone()
            
            if self.qkv_nic_config_algorithm["send_w_vals"] != None:
                self.west_tx["max_scores"] = max_scores[self.all_heads_on_chiplet[0]].clone()
            
            if self.pistil_transactor != None:
                instruction = {"name": "wait-max", "layer": layer_name, "op": "wait", "flag": "compute-scores-max"}
                self.pistil_transactor.net_instr.append(instruction)

        if operation == "exp_sum":
            if self.qkv_nic_config_algorithm["send_e_vals"] != None:
                self.east_tx["exp_sum"] = self.pistil.exp_sum[self.all_heads_on_chiplet[-1]].clone()
            
            if self.qkv_nic_config_algorithm["send_w_vals"] != None:
                self.west_tx["exp_sum"] = self.pistil.exp_sum[self.all_heads_on_chiplet[0]].clone()

            if self.pistil_transactor != None:
                instruction = {"name": "wait-exp-div", "layer": layer_name, "op": "wait", "flag": "compute-exp-sum"}
                self.pistil_transactor.net_instr.append(instruction)

        if operation == "sdpa_reduction":
            self.o_nic_send_remaining = {"send_e_vals": 0, "send_w_vals": 0}
            
            if self.o_nic_config_algorithm["send_e_vals"] > 0:
                send_e_vals = self.o_nic_config_algorithm["send_e_vals"]
                if self.o_nic_config_algorithm["start_reduction"]:
                    self.east_tx["sdpa_p_sum"] = self.pistil.spda_p_sum_flat[:, :, -send_e_vals:].clone()
                    self.pistil.spda_p_sum_flat = self.pistil.spda_p_sum_flat[:, :, 0:-send_e_vals]
                else:
                    self.o_nic_send_remaining["send_e_vals"] = send_e_vals
            
            if self.o_nic_config_algorithm["send_w_vals"] > 0:
                send_w_vals = self.o_nic_config_algorithm["send_w_vals"]
                if self.o_nic_config_algorithm["start_reduction"]:
                    self.west_tx["sdpa_p_sum"] = self.pistil.spda_p_sum_flat[:, :, 0:send_w_vals].clone()
                    self.pistil.spda_p_sum_flat = self.pistil.spda_p_sum_flat[:, :, send_w_vals:]
                else:
                    self.o_nic_send_remaining["send_w_vals"] = send_w_vals
            if self.pistil_transactor != None:
                instruction = {"name": "wait-q-reduction", "layer": layer_name, "op": "wait", "flag": "compute-q-reduction"}
                self.pistil_transactor.net_instr.append(instruction)
                instruction = {"name": "receive-sdpa-p-sum", "op": "receive-sdpa-p-sum", "batch_size": self.args.sim_batch_size, "receive_vals": self.pistil.spda_p_sum_flat.size()[-1], "send_vals": self.o_nic_config_algorithm["send_w_vals"] + self.o_nic_config_algorithm["send_e_vals"]}
                self.pistil_transactor.net_instr.append(instruction)

        self.forward = not self.qkv_nic_config_algorithm["start_reduction"] 

    def swing_decode_network(self, operation=None):
        if operation == "fused-norm":
            act_shard                   = self.east_rx["act_shard"]
            variance_rx                 = self.east_rx["variance"] 
            self.pistil.act_shard       = torch.cat((self.pistil.act_shard, act_shard), dim=-1)     # write data to compute engine
            self.pistil.singleton_buf  += variance_rx                             # write data to sington buffer for variance sum
            if self.pistil_transactor != None:
                instruction = {"name": "receive-act", "op": "receive-fused-act-norm", "batch_size": self.args.sim_batch_size, "params": self.east_rx["act_shard"].size()[-1], "variance_vals": 1}
                self.pistil_transactor.net_instr.append(instruction)
        if operation == "norm":
            variance_rx                 = self.east_rx["variance"] 
            self.pistil.singleton_buf  += variance_rx                             # write data to sington buffer for variance sum
            if self.pistil_transactor != None:    
                instruction = {"name": "receive-norm", "op": "receive-norm", "batch_size": self.args.sim_batch_size, "variance_vals": 1}
                self.pistil_transactor.net_instr.append(instruction)
        if operation == "cat":
            act_shard                   = self.east_rx["act_shard"]
            self.pistil.act_shard       = torch.cat((self.pistil.act_shard, act_shard), dim=-1)     # write data to compute engine
            if self.pistil_transactor != None:
                instruction = {"name": "receive-act", "op": "receive-act", "batch_size": self.args.sim_batch_size, "params": self.east_rx["act_shard"].size()[-1]}
                self.pistil_transactor.net_instr.append(instruction)

        # forward for swing
        self.west_tx = self.east_rx
        self.east_rx = {}

    def shuffle_decode_network(self, operation):
        if operation == "kv":
            if "k_vals" in self.west_rx: # received from west means append to first head
                head            = self.all_heads_on_chiplet[0]
                receved_ids     = self.kv_received_vals[head]
                new_ids         = self.west_rx["k_vals"].size()[-1]
                
                self.k_cache_write_back[head][:, :, receved_ids:receved_ids+new_ids] = self.west_rx["k_vals"]
                self.v_cache_write_back[head] = torch.cat((self.west_rx["v_vals"], self.v_cache_write_back[head]), dim=-1)
                self.kv_received_vals[head] += new_ids  

            if "k_vals" in self.east_rx: # received from east means append to last head
                head            = self.all_heads_on_chiplet[-1]
                receved_ids     = self.kv_received_vals[head]
                new_ids         = self.east_rx["k_vals"].size()[-1]

                self.k_cache_write_back[head][:, :, receved_ids:receved_ids+new_ids] = self.east_rx["k_vals"]
                self.v_cache_write_back[head] = torch.cat((self.v_cache_write_back[head], self.east_rx["v_vals"]), dim=-1)
                self.kv_received_vals[head] += new_ids
            
            if self.pistil_transactor != None:    
                instruction = {"name": "receive-kv", "op": "receive-kv", "batch_size": self.args.sim_batch_size, "east_vals": (self.east_rx["k_vals"].size()[-1] if "k_vals" in self.east_rx else 0), "west_vals": (self.west_rx["k_vals"].size()[-1] if "k_vals" in self.west_rx else 0), "note": "take max of the east/west bc must be synchronized receives"}
                self.pistil_transactor.net_instr.append(instruction)
        
        if operation == "q":
            if "q_vals" in self.west_rx: # received from west means append to first head
                head            = self.all_heads_on_chiplet[0]
                receved_ids     = self.q_received_vals[head]
                new_ids         = int(self.west_rx["q_vals"].size()[-1]/self.gqa)
                
                self.q_input_broadcast[head][:, :, :, receved_ids:receved_ids+new_ids] = self.west_rx["q_vals"].reshape((self.batch_size, self.seq_len, self.gqa, new_ids))
                self.q_received_vals[head] += new_ids  

            if "q_vals" in self.east_rx: # received from east means append to last head
                head            = self.all_heads_on_chiplet[-1]
                receved_ids     = self.q_received_vals[head]
                new_ids         = int(self.east_rx["q_vals"].size()[-1]/self.gqa)

                self.q_input_broadcast[head][:, :, :, receved_ids:receved_ids+new_ids] = self.east_rx["q_vals"].reshape((self.batch_size, self.seq_len, self.gqa, new_ids))
                self.q_received_vals[head] += new_ids
            
            if self.pistil_transactor != None:    
                instruction = {"name": "receive-q", "op": "receive-q", "batch_size": self.args.sim_batch_size, "east_vals": (self.east_rx["q_vals"].size()[-1] if "q_vals" in self.east_rx else 0), "west_vals": (self.west_rx["q_vals"].size()[-1] if "q_vals" in self.west_rx else 0), "note": "take max of the east/west bc must be synchronized receives"}
                self.pistil_transactor.net_instr.append(instruction)

        if operation == "max":
            # always put values into q
            if "max_scores" in self.west_rx: 
                head            = self.all_heads_on_chiplet[0]
                max_scores_new  = self.west_rx["max_scores"]
                max_concat      = torch.cat((self.pistil.max_scores[head].unsqueeze(-1), max_scores_new.unsqueeze(-1)), dim=-1)
                
                self.pistil.max_scores[head] = torch.max(max_concat, dim=-1).values

            if "max_scores" in self.east_rx: 
                head            = self.all_heads_on_chiplet[-1]
                max_scores_new  = self.east_rx["max_scores"]
                max_concat      = torch.cat((self.pistil.max_scores[head].unsqueeze(-1), max_scores_new.unsqueeze(-1)), dim=-1)
                
                self.pistil.max_scores[head] = torch.max(max_concat, dim=-1).values # write values to pistil scratchpad

            if self.pistil_transactor != None:    
                instruction = {"name": "receive-max", "op": "receive-max", "batch_size": self.args.sim_batch_size, "east_vals": (self.east_rx["max_scores"].size()[-1] if "max_scores" in self.east_rx else 0), "west_vals": (self.west_rx["max_scores"].size()[-1] if "max_scores" in self.west_rx else 0)}
                self.pistil_transactor.net_instr.append(instruction)

        if operation == "exp_sum":
            if "exp_sum" in self.west_rx: 
                head            = self.all_heads_on_chiplet[0]
                exp_sum         = self.west_rx["exp_sum"]
                self.pistil.exp_sum[head] += exp_sum # write values to pistil scratchpad
                
            if "exp_sum" in self.east_rx: 
                head            = self.all_heads_on_chiplet[-1]
                exp_sum         = self.east_rx["exp_sum"]
                self.pistil.exp_sum[head] += exp_sum # write values to pistil scratchpad
            
            if self.pistil_transactor != None:    
                instruction = {"name": "receive-exp-sum", "op": "receive-exp-sum", "batch_size": self.args.sim_batch_size, "east_vals": (self.east_rx["exp_sum"].size()[-1] if "exp_sum" in self.east_rx else 0), "west_vals": (self.west_rx["exp_sum"].size()[-1] if "exp_sum" in self.west_rx else 0)}
                self.pistil_transactor.net_instr.append(instruction)

        if operation == "sdpa_reduction":
            # receive from west, send east if forwarding
            if "sdpa_p_sum" in self.west_rx: # received from west means append to first head
                received_vals = self.west_rx["sdpa_p_sum"].size()[-1]
                if self.forward:
                    self.pistil.spda_p_sum_flat[:, :, -received_vals:] += self.west_rx["sdpa_p_sum"]
                    send_e_vals                                         = self.o_nic_send_remaining["send_e_vals"]
                    self.east_tx["sdpa_p_sum"]                          = self.pistil.spda_p_sum_flat[:, :, -send_e_vals:].clone()
                    self.pistil.spda_p_sum_flat                         = self.pistil.spda_p_sum_flat[:, :, 0:-send_e_vals]
                    self.o_nic_send_remaining["send_e_vals"]            = 0
                else:
                    self.pistil.spda_p_sum_flat[:, :, 0:received_vals] += self.west_rx["sdpa_p_sum"]
            
            # receive from east, send west if forwarding
            if "sdpa_p_sum" in self.east_rx:
                received_vals = self.east_rx["sdpa_p_sum"].size()[-1]
                if self.forward:
                    self.pistil.spda_p_sum_flat[:, :, 0:received_vals] += self.east_rx["sdpa_p_sum"]
                    send_w_vals                                         = self.o_nic_send_remaining["send_w_vals"]
                    self.west_tx["sdpa_p_sum"]                          = self.pistil.spda_p_sum_flat[:, :, 0:send_w_vals].clone()
                    self.pistil.spda_p_sum_flat                         = self.pistil.spda_p_sum_flat[:, :, send_w_vals:]
                    self.o_nic_send_remaining["send_w_vals"]            = 0
                else:
                    self.pistil.spda_p_sum_flat[:, :, -received_vals:] += self.east_rx["sdpa_p_sum"]

        if operation != "sdpa_reduction":
            if self.forward:
                self.west_tx = self.east_rx
                self.east_tx = self.west_rx

                if len(self.east_tx.keys()) == 0 and len(self.west_tx.keys()) == 0:
                    self.shuffle_complete = True # no more data to send
                else:
                    self.shuffle_complete = False
            else:
                self.shuffle_complete = True

        else:
            if len(self.east_tx.keys()) == 0 and len(self.west_tx.keys()) == 0 and self.o_nic_send_remaining["send_w_vals"] == 0 and self.o_nic_send_remaining["send_e_vals"] == 0:
                self.shuffle_complete           = True # no more data to send
            else:
                self.shuffle_complete = False  

        self.west_rx = {}
        self.east_rx = {}

    def move_sdpa_reduction(self):
        self.pistil.act_shard = self.pistil.spda_p_sum_flat.to(self.dtype)
        
        self.west_tx = {}
        self.east_tx = {}
        self.west_tx["act_shard"]   = self.pistil.act_shard.clone()
        if self.pistil_transactor != None:
            instruction = {"name": "flag-q-reduction", "op": "flag", "flag": "network-q"}
            self.pistil_transactor.net_instr.append(instruction)
            instruction = {"name": "wait-compute-swing", "layer": "wO", "op": "wait", "flag": "compute-swing"}
            self.pistil_transactor.net_instr.append(instruction)
        

    

