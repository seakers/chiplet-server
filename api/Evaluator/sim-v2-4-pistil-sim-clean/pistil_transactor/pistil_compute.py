import math
import torch
import copy as cp


class PistilCompute:
    def __init__(self, sim, sys_manager, llm_manager, garden, pistil):
        self.sim                = sim
        self.sys_manager        = sys_manager
        self.llm_manager        = llm_manager
        self.garden             = garden
        self.pistil             = pistil

        ################################################################
        # system config
        clk_freq                = self.sys_manager.get_clk_freq()

        # compute 
        self.cores_per_chiplet  = self.sys_manager.get_cores_per_chiplet() 
        
        # modeling full compute chiplet so need enough pieces for all cores in this simulation... in practice could be a bit faster...
        self.tile_act_width     = self.cores_per_chiplet * self.sys_manager.get_tile_height()      # width of input vector
        self.tile_weight_width  = self.sys_manager.get_tile_width()       # broadcast reuse factor

        self.on_chip_dtype      = 2
        self.on_chip_accum_dtype = 4
        self.w_dtype            = self.llm_manager.w_dtype
        self.norm_dtype         = self.llm_manager.norm_dtype
        self.kv_dtype           = self.llm_manager.kv_dtype

        self.max_bytes_per_vvm  = self.tile_act_width * self.on_chip_dtype / 2  # Vector Vector Multiply Ops - 2*8 OPs / cycle
        self.max_bytes_per_vmm  = self.tile_act_width * self.tile_weight_width * self.on_chip_dtype # Vector Matrix Multiply Ops - 2*64 Ops / cycle
        
        self.ns_per_vvm_mult    = 1 / sys_manager.get_net_bus_freq() * (10**9) # network from activation scratchpad is slower
        self.ns_per_vmm_mult    = 1 / clk_freq * (10**9)

        self.hp_ops_latency     = self.sys_manager.get_hp_ops_latency()
        self.compute_perf_gops  = self.sys_manager.get_chiplet_compute() / (10**9)
        ################################################################


        self.all_instr          = None
        self.cur_instr          = None

        self.instr_start_time   = None
        self.next_time          = 0

        # flags for synchronization with network
        self.flag_qkv           = False
        self.flag_scores_max    = False
        self.flag_exp_sum       = False
        self.flag_q_reduction   = False
        self.flag_norm          = False
        self.flag_layer         = False
        self.instr_tmp_info     = {}

    def set_instructions(self, all_instr):
        self.all_instr = all_instr
        self.set_next_instr()
        self.next_time = self.start_instr(0)


    def set_next_instr(self):
        if len(self.all_instr) > 0:
            self.cur_instr = self.all_instr[0]
            self.all_instr = self.all_instr[1:]
            self.instr_start_time = self.next_time
            self.cur_instr["saved"] = cp.deepcopy(self.cur_instr)
            self.cur_instr["first-time"] = True
        else:
            self.cur_instr = None
            self.next_time = torch.inf
        
        self.instr_tmp_info = {}
    
    def start_instr(self, current_time):
        if self.cur_instr == None:
            return self.next_time

        name = self.cur_instr["name"]
        op = self.cur_instr["op"]

        # check to see if operands for data exist
        if name == "input_norm" and op == "act_fn":
            bytes_remaining = self.cur_instr["params"] * self.on_chip_dtype
            bytes_needed = min(self.max_bytes_per_vvm/2, bytes_remaining)  # divide by 2 because you typically need 2 vectors and this only models 1, plus we need to throttle SRAM reads

            buffered_bytes = self.sim.act_buffer["act-shard"]

            batch_size = self.cur_instr["batch_size"]
            
            # throttle number of hp ops in use if doesn't fit into disributed L1 buffer
            total_tmacs = self.sys_manager.get_tmacs_per_cores()
            DIST_L1_BUFFERS = self.sys_manager.sys_cfg["CORE"]["TMAC_BUFFER_B"]*total_tmacs*self.sys_manager.get_cores_per_chiplet()
            if self.cur_instr["batch_size"] * self.cur_instr["saved"]["params"] * self.on_chip_dtype > DIST_L1_BUFFERS:
                tile_reuse = math.ceil(batch_size / math.ceil(total_tmacs/2)) # half the tmacs
            else:
                tile_reuse = math.ceil(batch_size / total_tmacs)
            
            if buffered_bytes < bytes_needed:
                return torch.inf
            
            else:
                if "next-act-shard" not in self.sim.act_buffer:
                    self.sim.act_buffer["next-act-shard"] = 0
                
                additional_latency = (self.hp_ops_latency if self.cur_instr["first-time"] else 0)
                self.cur_instr["first-time"] = False
                return current_time + self.ns_per_vvm_mult + additional_latency
        
        elif op == "vvm" or op == "vva" or op == "vmult" or op == "act_fn" or op == "vadd":
            if name == "input_norm" and "input_norm" not in self.sim.weight_buffer:
                return torch.inf
            if name == "post_atten_norm" and "post_atten_norm" not in self.sim.weight_buffer:
                return torch.inf
            if name == "pre_ff_norm" and "pre_ff_norm" not in self.sim.weight_buffer:
                return torch.inf
            if name == "post_ff_norm" and "post_ff_norm" not in self.sim.weight_buffer:
                return torch.inf
            if name == "model_norm" and "model_norm" not in self.sim.weight_buffer:
                return torch.inf

            if "next-act-shard" not in self.sim.act_buffer:
                self.sim.act_buffer["next-act-shard"] = 0
                    
            bytes_remaining = self.cur_instr["params"] * self.on_chip_dtype
            bytes_needed = min(self.max_bytes_per_vvm/2, bytes_remaining) # divide by 2 because you typically need 2 vectors and this only models 1, plus we need to throttle SRAM reads

            act_buffered_bytes = self.sim.act_buffer["act-shard"]
            
            batch_size = self.cur_instr["batch_size"]
            
            total_tmacs = self.sys_manager.get_tmacs_per_cores()
            DIST_L1_BUFFERS = self.sys_manager.sys_cfg["CORE"]["TMAC_BUFFER_B"]*total_tmacs*self.sys_manager.get_cores_per_chiplet()
            if self.cur_instr["batch_size"] * self.cur_instr["saved"]["params"] * self.on_chip_dtype > DIST_L1_BUFFERS:
                tile_reuse = math.ceil(batch_size / math.ceil(total_tmacs/2)) # half the tmacs
            else:
                tile_reuse = math.ceil(batch_size / total_tmacs)
            
            # tile_reuse = math.ceil(batch_size / self.sys_manager.get_tmacs_per_cores())
            

            additional_latency = (self.hp_ops_latency if self.cur_instr["first-time"] else 0)
            self.cur_instr["first-time"] = False

            if name == "input_norm" and op == "vvm":
                weight_buffered_bytes = self.sim.weight_buffer["input_norm"] * self.on_chip_dtype / self.norm_dtype
            elif name == "post_atten_norm" and op == "vvm":
                weight_buffered_bytes = self.sim.weight_buffer["post_atten_norm"] * self.on_chip_dtype / self.norm_dtype
            elif name == "pre_ff_norm" and op == "vvm":
                weight_buffered_bytes = self.sim.weight_buffer["pre_ff_norm"] * self.on_chip_dtype / self.norm_dtype
            elif name == "post_ff_norm" and op == "vvm":
                weight_buffered_bytes = self.sim.weight_buffer["post_ff_norm"] * self.on_chip_dtype / self.norm_dtype
            elif name == "model_norm" and op == "vvm":
                weight_buffered_bytes = self.sim.weight_buffer["model_norm"] * self.on_chip_dtype / self.norm_dtype
            elif name == "rot_emb_mult" or name == "rot_emb_add" or "softcap_down" in name or "softcap_up" in name or op == "act_fn" or "sub" in name or "sQKT_exp_div" in name or "add" == name or op == "vmult":
                # everything should be on chip already
                if op == "act_fn":
                    if self.cur_instr["fn"] == "tanh":
                        self.cur_instr["saved"]["ops_per_param"] = 16
                        return current_time + self.ns_per_vvm_mult*tile_reuse * 16 + additional_latency
                    elif self.cur_instr["fn"] == "exp": 
                        self.cur_instr["saved"]["ops_per_param"] = 4
                        return current_time + self.ns_per_vvm_mult*tile_reuse * 4 + additional_latency
                    elif self.cur_instr["fn"] == "pow2":
                        self.cur_instr["saved"]["ops_per_param"] = 1
                        return current_time + self.ns_per_vvm_mult*tile_reuse + additional_latency
                    elif self.cur_instr["fn"] == "gelu_pytorch_tanh":
                        self.cur_instr["saved"]["ops_per_param"] = 10
                        return current_time + self.ns_per_vvm_mult*tile_reuse * 10 + additional_latency
                    elif self.cur_instr["fn"] == "silu":
                        self.cur_instr["saved"]["ops_per_param"] = 9
                        return current_time + self.ns_per_vvm_mult*tile_reuse * 9 + additional_latency
                else:
                    self.cur_instr["saved"]["ops_per_param"] = 1
                    return current_time + self.ns_per_vvm_mult*tile_reuse + additional_latency

            if act_buffered_bytes < bytes_needed or weight_buffered_bytes < bytes_needed:
                return torch.inf
            else:
                self.cur_instr["saved"]["ops_per_param"] = 1
                return current_time + self.ns_per_vvm_mult*tile_reuse + additional_latency

        elif op == "vmm" and name in ["wQKV", "wO", "gate_up_proj", "down_proj", "gate_up_proj_s", "down_proj_s", "gate_up_proj_e", "down_proj_e", "expert_proj", "lm_head"]:
            if name not in self.sim.weight_buffer:
                return torch.inf

            if "next-act-shard" not in self.sim.act_buffer: # first time here
                self.sim.act_buffer["next-act-shard"] = 0
                self.tiled_rows = math.ceil(self.cur_instr["in_dim"] / self.tile_act_width)
                self.tiled_cols = math.ceil(self.cur_instr["out_dim"] / self.tile_weight_width) 
                self.last_row_vals = self.cur_instr["in_dim"] % self.tile_act_width
                self.last_col_vals = self.cur_instr["out_dim"] % self.tile_weight_width

                self.last_row_vals = self.tile_act_width if self.last_row_vals == 0 else self.last_row_vals
                self.last_col_vals = self.tile_weight_width if self.last_col_vals == 0 else self.last_col_vals
                
                C_SIZE = self.cur_instr["batch_size"] * self.cur_instr["out_dim"] * self.on_chip_accum_dtype / (1024 * 1024) 
                
                if C_SIZE > self.sys_manager.sys_cfg["CORE"]["C_MATRIX_THRESHOLD"]*self.sys_manager.sys_cfg["CORE"]["GLOBAL_BUFFER_B"]/(1024*1024):
                    print("[Warning]: Not enough on-chip capacity for holding complete accumulation buffer for VMM... (layer %s)" % name)
                    print("\tNeed to model thrashing of 'C' Matrix to memory...")
                    print("\tTry increasing the network buffer size, or increasing the number of CUs...")
                    print("\tCould also break up the matmul into multiple column sharded matmuls with reductions between bc each core hold slice of partial sum...")
                    print(
                        f"\t{name} - BS={self.cur_instr['batch_size']} - "
                        f"In Dim={self.cur_instr['in_dim']} Out Dim={self.cur_instr['out_dim']}"
                    )
                    print("\tC_SIZE: %0.2fMB > %0.2fMB / Core * %0.2f" % (C_SIZE, (self.sys_manager.sys_cfg["CORE"]["GLOBAL_BUFFER_B"]/(1024*1024)), self.sys_manager.sys_cfg["CORE"]["C_MATRIX_THRESHOLD"]))
                    # exit()

                if name == "wO":
                    self.sim.act_buffer["act-shard"] *= self.cur_instr["batch_size"]
                self.cur_row = 0
                self.cur_col = 0
                # if name == "wO":
                #     print("Buffered Bytes: ", self.sim.act_buffer["act-shard"])
                #     exit()
                
            row_vals = self.last_row_vals if self.cur_row + 1 == self.tiled_rows else self.tile_act_width
            col_vals = self.last_col_vals if self.cur_col + 1 == self.tiled_cols else self.tile_weight_width

            act_bytes_needed = self.cur_instr["batch_size"] * row_vals * self.on_chip_dtype
            weight_bytes_needed = row_vals * col_vals * self.w_dtype

            act_buffered_bytes = self.sim.act_buffer["act-shard"]
            weight_buffered_bytes = self.sim.weight_buffer[name]

            batch_size = self.cur_instr["batch_size"]
            tile_reuse = math.ceil(batch_size / self.sys_manager.get_tmacs_per_cores())                
            if weight_buffered_bytes < weight_bytes_needed or act_buffered_bytes < act_bytes_needed:
                return torch.inf
            else:
                next_time = current_time + self.ns_per_vmm_mult * tile_reuse
                if "util" not in self.instr_tmp_info:
                    self.instr_tmp_info["util"] = []
                tile_util = (math.floor(batch_size / self.sys_manager.get_tmacs_per_cores()) + (batch_size % self.sys_manager.get_tmacs_per_cores())/self.sys_manager.get_tmacs_per_cores())/tile_reuse
                self.instr_tmp_info["util"].append([current_time, next_time, tile_util])
                return next_time

        elif op == "vmm" and "QKT" in name:
            if name not in self.sim.weight_buffer:
                return torch.inf
            
            if "next-act-shard" not in self.sim.act_buffer: # first time here
                self.sim.act_buffer["next-act-shard"] = 0
                self.tiled_rows = math.ceil(self.cur_instr["in_dim"] / self.tile_act_width)
                self.tiled_cols = math.ceil(self.cur_instr["out_dim"] / self.tile_weight_width) 
                self.last_row_vals = self.cur_instr["in_dim"] % self.tile_act_width
                self.last_col_vals = self.cur_instr["out_dim"] % self.tile_weight_width

                self.last_row_vals = self.tile_act_width if self.last_row_vals == 0 else self.last_row_vals
                self.last_col_vals = self.tile_weight_width if self.last_col_vals == 0 else self.last_col_vals

                self.cur_row = 0
                self.cur_col = 0

                # print(self.cur_instr)
                # s(QKT)V could be less than 100% utilization because theres not that many rows per core if the prefill chunk size is small...
                # e.g. if the cached prefill tokens is <256, then each K$ tile is <16 meaning each core only has <16 rows to work with... so last tile will be underutilized... 
                
            row_vals = self.last_row_vals if self.cur_row + 1 == self.tiled_rows else self.tile_act_width
            col_vals = self.last_col_vals if self.cur_col + 1 == self.tiled_cols else self.tile_weight_width

            # no activations needed, should already be resident
            weight_bytes_needed = row_vals * col_vals * self.kv_dtype #self.on_chip_dtype
            
            weight_buffered_bytes = self.sim.weight_buffer[name]
            
            batch_size = self.cur_instr["q_heads"] * self.cur_instr["tokens"] 
            tile_reuse = math.ceil(batch_size / self.sys_manager.get_tmacs_per_cores())

            if weight_buffered_bytes < weight_bytes_needed:
                return torch.inf
            else:
                next_time = current_time + self.ns_per_vmm_mult * tile_reuse
                if "util" not in self.instr_tmp_info:
                    self.instr_tmp_info["util"] = []
                tile_util = (math.floor(batch_size / self.sys_manager.get_tmacs_per_cores()) + (batch_size % self.sys_manager.get_tmacs_per_cores())/self.sys_manager.get_tmacs_per_cores())/tile_reuse
                self.instr_tmp_info["util"].append([current_time, next_time, tile_util])
                return next_time
        
        elif op == "reduction":
            # reductions can immediately happen    
            num_cores_per_chiplet = self.sys_manager.get_cores_per_chiplet()
            shard_size = (self.cur_instr["batch_size"] * self.cur_instr["out_dim"] * self.on_chip_accum_dtype) / num_cores_per_chiplet
            bw_bound_per_shard = shard_size / self.sys_manager.get_reduction_bandwidth() * self.sys_manager.get_clk_period_ns()
            if bw_bound_per_shard > self.sys_manager.get_reduction_latency():
                shard_latency = bw_bound_per_shard
            else:
                shard_latency = self.sys_manager.get_reduction_latency()
            return current_time + shard_latency * (num_cores_per_chiplet - 1) + self.sys_manager.get_reduction_latency()

        elif op == "load":
            batch_size = self.cur_instr["batch_size"]
            params = self.cur_instr["params"]
            dtype = self.cur_instr["dtype"]

            self.sim.act_buffer["act-shard"] = batch_size * params * dtype
            if "next-act-shard" in self.sim.act_buffer:
                del self.sim.act_buffer["next-act-shard"]
            return current_time + 1

        elif name == "flag-norm" and op == "flag":
            return current_time + 1
        elif name == "flag-mem" and op == "flag":
            return current_time + 1
        elif name == "flag-linear" and op == "flag":
            return current_time + 1
        elif name == "flag-qkv" and op == "flag":
            return current_time + 1
        elif name == "flag-scores-max" and op == "flag":
            return current_time + 1
        elif name == "flag-exp-div" and op == "flag":
            return current_time + 1
        elif name == "flag-q-reduction" and op == "flag":
            return current_time + 1
        elif name == "flag-layer" and op == "flag":
            return current_time + 1
        elif name == "flag-experts-ready" and op == "flag":
            return current_time + 1

        elif name == "wait-qkv" and op == "wait":
            if self.flag_qkv == False:
                return torch.inf
            else:
                return current_time + 1
        
        elif name == "wait-scores-max" and op == "wait":
            if self.flag_scores_max == False:
                return torch.inf
            else:
                return current_time + 1
        elif name == "wait-exp-div" and op == "wait":
            if self.flag_exp_sum == False:
                return torch.inf
            else:
                return current_time + 1
        elif name == "wait-q-reduction" and op == "wait":
            if self.flag_q_reduction == False:
                return torch.inf
            else:
                return current_time + 1
        elif name == "wait-norm" and op == "wait":
            if self.flag_norm == False:
                return torch.inf
            else:
                return current_time + 1
                
        elif name == "wait-layer" and op == "wait":
            if self.flag_layer == False:
                return torch.inf
            else:
                return current_time + 1

        else:
            print("comp", self.cur_instr)
            print("comp exitting")
            exit()

        return torch.inf

    def exe_instr(self, all_events):
        if self.cur_instr == None:
            return 
        name = self.cur_instr["name"]
        op = self.cur_instr["op"]

        # check to see if operands for data exist
        if name == "input_norm" and op == "act_fn":
            bytes_remaining = self.cur_instr["params"] * self.on_chip_dtype
            bytes_needed = min(self.max_bytes_per_vvm/2, bytes_remaining)

            self.sim.act_buffer["act-shard"] -= bytes_needed
            self.cur_instr["params"] -= bytes_needed / self.on_chip_dtype
            self.sim.act_buffer["next-act-shard"] += self.cur_instr["batch_size"] * bytes_needed
            if self.cur_instr["params"] == 0:
                instr_start = self.instr_start_time
                instr_end = self.next_time
                exe_time = instr_end - instr_start
                energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_compute_pipeline_act_weight(self.cur_instr["batch_size"], self.cur_instr["saved"]["params"], exe_time/(10**9), on_chip_dtype=self.on_chip_dtype)

                event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": 0.0, "energy": energy_stats, "power": power_stats}
                all_events["comp"].append(event)
                self.set_next_instr()
                self.sim.act_buffer["act-shard"] = self.sim.act_buffer["next-act-shard"]
                del self.sim.act_buffer["next-act-shard"]
                return 

        elif op == "vvm" or op == "vva" or op == "vmult" or op == "act_fn" or op == "vadd":               
            bytes_remaining = self.cur_instr["params"] * self.on_chip_dtype # self.cur_instr["batch_size"] * 
            bytes_needed = min(self.max_bytes_per_vvm/2, bytes_remaining)

            self.sim.act_buffer["act-shard"] -= bytes_needed
            self.cur_instr["params"] -= bytes_needed / self.on_chip_dtype # / self.cur_instr["batch_size"] 
            self.sim.act_buffer["next-act-shard"] += self.cur_instr["batch_size"] * bytes_needed
            
            bytes_needed = bytes_needed * self.norm_dtype / self.on_chip_dtype
            if name == "input_norm" and op == "vvm":
                self.sim.weight_buffer["input_norm"] -= bytes_needed 
                self.sim.weight_buffer_used -= bytes_needed
            if name == "post_atten_norm" and op == "vvm":
                self.sim.weight_buffer["post_atten_norm"] -= bytes_needed
                self.sim.weight_buffer_used -= bytes_needed
            if name == "pre_ff_norm" and op == "vvm":
                self.sim.weight_buffer["pre_ff_norm"] -= bytes_needed 
                self.sim.weight_buffer_used -= bytes_needed
            if name == "post_ff_norm" and op == "vvm":
                self.sim.weight_buffer["post_ff_norm"] -= bytes_needed
                self.sim.weight_buffer_used -= bytes_needed
            if name == "model_norm" and op == "vvm":
                self.sim.weight_buffer["model_norm"] -= bytes_needed 
                self.sim.weight_buffer_used -= bytes_needed

            if self.cur_instr["params"] == 0:
                instr_start = self.instr_start_time
                instr_end = self.next_time
                exe_time = instr_end - instr_start

                # print(name, exe_time, self.cur_instr["batch_size"], self.cur_instr["saved"]["params"])
                energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_compute_pipeline_activation(self.cur_instr["batch_size"], self.cur_instr["saved"]["params"], self.cur_instr["saved"]["ops_per_param"], exe_time/(10**9), on_chip_hp_dtype=self.on_chip_accum_dtype)

                event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": 0.0, "energy": energy_stats, "power": power_stats}
                all_events["comp"].append(event)
                self.set_next_instr()
                self.sim.act_buffer["act-shard"] = self.sim.act_buffer["next-act-shard"]
                del self.sim.act_buffer["next-act-shard"]
                return 

        elif op == "vmm" and name in ["wQKV", "wO", "gate_up_proj", "down_proj", "gate_up_proj_s", "down_proj_s", "gate_up_proj_e", "down_proj_e", "expert_proj", "lm_head"]:
            row_vals = self.last_row_vals if self.cur_row + 1 == self.tiled_rows else self.tile_act_width
            col_vals = self.last_col_vals if self.cur_col + 1 == self.tiled_cols else self.tile_weight_width

            act_bytes_needed = self.cur_instr["batch_size"] * row_vals * self.on_chip_dtype
            weight_bytes_needed = row_vals * col_vals * self.w_dtype

            #self.sim.act_buffer["act-shard"] -= act_bytes_needed / (self.tiled_cols)
            self.sim.weight_buffer[name] -= weight_bytes_needed
                
            self.sim.weight_buffer_used -= weight_bytes_needed
            #self.sim.act_buffer["next-act-shard"] += act_bytes_needed / self.tiled_rows # not needed since we can calculate directly using out_dim

            if self.cur_row + 1 == self.tiled_rows and self.cur_col + 1 == self.tiled_cols:
                total_flops = 2 * self.cur_instr["batch_size"] * self.cur_instr["in_dim"] * self.cur_instr["out_dim"]
                instr_start = self.instr_start_time
                instr_end = self.next_time
                exe_time = instr_end - instr_start
                gops = total_flops/exe_time # total_flops in ops and time in ns so results in gops
                util = 100 * gops / self.compute_perf_gops

                energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_compute_pipeline_vmm(self.cur_instr["batch_size"], self.cur_instr["in_dim"], self.cur_instr["out_dim"], exe_time/(10**9), wei_dtype=self.w_dtype, on_chip_dtype=self.on_chip_dtype)

                event = {"event": name, "op": "vmm", "start_time": instr_start, "end_time": instr_end, "total-flops": total_flops, "gops": gops, "util": util, "energy": energy_stats, "power": power_stats, "util-exact": self.instr_tmp_info["util"]}
                all_events["comp"].append(event)
                self.sim.act_buffer["act-shard"] = self.cur_instr["batch_size"] * self.cur_instr["out_dim"] * self.on_chip_dtype
                del self.sim.act_buffer["next-act-shard"]
                self.set_next_instr()
                return 
            elif self.cur_col + 1 == self.tiled_cols:
                self.cur_row += 1
                self.cur_col = 0
                self.sim.act_buffer["act-shard"] -= act_bytes_needed 
            else: 
                self.cur_col += 1
        
        elif op == "vmm" and "QKT" in name:
            row_vals = self.last_row_vals if self.cur_row + 1 == self.tiled_rows else self.tile_act_width
            col_vals = self.last_col_vals if self.cur_col + 1 == self.tiled_cols else self.tile_weight_width

            weight_bytes_needed = row_vals * col_vals * self.kv_dtype

            if self.cur_instr["clear"] == True:
                self.sim.weight_buffer[name] -= weight_bytes_needed
                self.sim.weight_buffer_used -= weight_bytes_needed

            if self.cur_row + 1 == self.tiled_rows and self.cur_col + 1 == self.tiled_cols:
                total_flops = 2 * self.cur_instr["q_heads"] * self.cur_instr["tokens"] * self.cur_instr["in_dim"] * self.cur_instr["out_dim"]
                instr_start = self.instr_start_time
                instr_end = self.next_time
                exe_time = instr_end - instr_start
                gops = total_flops/(instr_end - instr_start) # total_flops in ops and time in ns so results in gops
                util = 100 * gops / self.compute_perf_gops

                energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_compute_pipeline_vmm(self.cur_instr["q_heads"]*self.cur_instr["tokens"], self.cur_instr["in_dim"], self.cur_instr["out_dim"], exe_time/(10**9), wei_dtype=self.kv_dtype, on_chip_dtype=self.on_chip_dtype)

                event = {"event": name, "op": "vmm", "start_time": instr_start, "end_time": instr_end, "total-flops": total_flops, "gops": gops, "util": util, "energy": energy_stats, "power": power_stats, "util-exact": self.instr_tmp_info["util"]}
                all_events["comp"].append(event)
                self.sim.act_buffer["act-shard"] = self.cur_instr["out_dim"] * self.on_chip_dtype
                del self.sim.act_buffer["next-act-shard"]
                self.set_next_instr()
                return 
            elif self.cur_col + 1 == self.tiled_cols:
                self.cur_row += 1
                self.cur_col = 0
            else: 
                self.cur_col += 1

        elif name == "load":
            self.set_next_instr()

        elif op == "reduction":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start

            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_compute_reduction(self.cur_instr["batch_size"], self.cur_instr["out_dim"], exe_time/(10**9), on_chip_accum_dtype=self.on_chip_accum_dtype)

            event = {"event": "reduction", "op": "reduction", "start_time": instr_start, "end_time": instr_end, "util":0.0, "energy": energy_stats, "power": power_stats}
            all_events["comp"].append(event)
            self.set_next_instr()

        elif name == "flag-norm" and op == "flag":
            self.sim.pistil_net.flag_norm = True
            self.set_next_instr()
        elif name == "flag-mem" and op == "flag":
            self.sim.pistil_mem.wb_flags += 1
            self.set_next_instr()
        elif name == "flag-linear" and op == "flag":
            self.sim.pistil_net.flag_linear = True
            self.set_next_instr()                
        elif name == "flag-qkv" and op == "flag":
            self.sim.pistil_net.flag_qkv = True
            self.set_next_instr() 
        elif name == "flag-scores-max" and op == "flag":
            self.sim.pistil_net.flag_scores_max = True
            self.set_next_instr() 
        elif name == "flag-exp-div" and op == "flag":
            self.sim.pistil_net.flag_exp_sum = True
            self.set_next_instr() 
        elif name == "flag-q-reduction" and op == "flag":
            self.sim.pistil_net.flag_q_reduction = True
            self.set_next_instr() 
        elif name == "flag-layer" and op == "flag":
            self.sim.pistil_net.flag_layer = True
            self.set_next_instr() 
        elif name == "flag-experts-ready" and op == "flag":
            self.sim.pistil_mem.flag_experts_ready = True
            self.set_next_instr() 
    

        elif name == "wait-qkv" and op == "wait":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["comp"].append(event)
            self.set_next_instr()
            self.flag_qkv = False            
        elif name == "wait-scores-max" and op == "wait":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["comp"].append(event)
            self.set_next_instr()
            self.flag_qkv = False            
        elif name == "wait-exp-div" and op == "wait":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["comp"].append(event)
            self.set_next_instr()
            self.flag_exp_sum = False            
        elif name == "wait-q-reduction" and op == "wait":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["comp"].append(event)
            self.set_next_instr()
            self.flag_q_reduction = False
            start_wo_input_shard = self.garden.ws_input_o_proj[self.pistil.chiplet_id][1][1] - self.garden.ws_input_o_proj[self.pistil.chiplet_id][1][0]
            self.sim.act_buffer["act-shard"] = start_wo_input_shard * self.on_chip_dtype

            if "new-act-shard" in self.sim.act_buffer:
                del start_wo_input_shard["new-act-shard"]
        elif name == "wait-norm" and op == "wait":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["comp"].append(event)
            self.set_next_instr()
            self.flag_norm = False 

        elif name == "wait-layer" and op == "wait":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["comp"].append(event)
            self.set_next_instr()
            self.flag_layer = False 
        else:
            print("HERE")


    def run(self, current_time, all_events):
        if self.next_time == current_time:
            # then we can complete segment of current instruction and/or pull a new instruction
            self.exe_instr(all_events)
            self.next_time = self.start_instr(current_time)
            

        elif self.next_time == torch.inf and self.cur_instr != None:
            # there must be a stall... 
            # see if we unstalled because space freed up in buffer and set next_time if possible
            self.next_time = self.start_instr(current_time)
            
        
        return self.next_time