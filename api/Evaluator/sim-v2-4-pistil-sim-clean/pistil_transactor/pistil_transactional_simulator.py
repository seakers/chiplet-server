import math
import time
import json
import torch
import pickle 
import copy as cp

from tqdm import tqdm

from pistil_transactor.pistil_compute import PistilCompute
from pistil_transactor.pistil_mem import PistilMem
from pistil_transactor.pistil_network import PistilNetwork


class PistilTransactionalSimulator:
    def __init__(self, sys_manager, llm_manager, garden):
        self.sys_manager        = sys_manager
        self.args               = llm_manager.args
        self.llm_manager        = llm_manager
        self.garden             = garden
        self.my_pistil          = self.garden.chiplet_garden[0]

        ################################################################
        # system config
        # memory 
        self.on_chip_dtype      = 2
        self.mem_w_dtype        = self.llm_manager.w_dtype
        self.mem_norm_dtype     = self.llm_manager.norm_dtype
        self.mem_kv_dtype       = self.llm_manager.kv_dtype
        ################################################################

        self.weight_buffer      = {}
        self.weight_buffer_used = 0 
        self.act_buffer         = {"act-shard": 0}
        self.act_buffer_used    = 0

        self.pistil_comp        = PistilCompute(self, sys_manager, llm_manager, garden, self.my_pistil)
        self.pistil_mem         = PistilMem(self, sys_manager, llm_manager, garden, self.my_pistil)
        self.pistil_net         = PistilNetwork(self, sys_manager, llm_manager, garden, self.my_pistil)

        self.mem_instr          = []
        self.comp_instr         = []
        self.net_instr          = []
        
        # lm_head
        self.mem_instr_lm_head  = []
        self.comp_instr_lm_head = []
        self.net_instr_lm_head  = []

    def save_trace(self, trace_file):
        print("Saving trace to %s" % (trace_file + ".json"))
        trace_data = {
            'mem_instr': self.mem_instr,
            'comp_instr': self.comp_instr,
            'net_instr': self.net_instr,
            'mem_instr_lm_head': self.mem_instr_lm_head,
            'comp_instr_lm_head': self.comp_instr_lm_head,
            'net_instr_lm_head': self.net_instr_lm_head,
        }
        with open(trace_file + ".pkl", 'wb') as f:
            pickle.dump(trace_data, f)
    
        # Also save as plain text (JSON format)
        # with open(trace_file + '.json', 'w') as f:
        #     json.dump(trace_data, f, indent=2, default=str)
        

    def load_trace(self, trace_file):
        with open(trace_file + ".pkl", 'rb') as f:
            data = pickle.load(f)
            self.mem_instr_base = data.get('mem_instr', [])
            self.comp_instr_base = data.get('comp_instr', [])
            self.net_instr_base = data.get('net_instr', [])
            self.mem_instr_base_lm_head = data.get('mem_instr_lm_head', [])
            self.comp_instr_base_lm_head = data.get('comp_instr_lm_head', [])
            self.net_instr_base_lm_head = data.get('net_instr_lm_head', [])
        
        self.comp_instr_base.append({'name': 'flag-layer', 'op':'flag', 'flag': 'compute-layer-complete'})
        self.comp_instr_base.append({'name': 'wait-layer', 'op':'wait', 'flag': 'network-layer-complete'})
        self.net_instr_base.append({'name': 'wait-layer', 'op':'wait', 'flag': 'compute-layer-complete'})
        self.net_instr_base.append({'name': 'flag-layer', 'op':'flag', 'flag': 'network-layer-complete'})

        # step 1: duplicate trace for the number of layers we want to simulate
        self.mem_instr = []
        self.comp_instr = []
        self.net_instr = []

        # if simulating all layers
        if self.args.sim_num_layers == -1:
            self.args.sim_num_layers = self.llm_manager.num_layers
            if "llama4" in self.llm_manager.model_name:
                num_layers = int(self.llm_manager.num_layers / 2)
            else:
                num_layers = self.llm_manager.num_layers
            self.args.sim_num_layers = self.llm_manager.num_layers
        else:
            # if llama4 then trace multiplier needs to consider that trace is 2 layers
            if "llama4" in self.llm_manager.model_name:
                if self.args.sim_num_layers * 2 > self.llm_manager.num_layers:
                    print("WARNING: When simulating llama4, number of sim layers is x2 bc trace is first 2 layers.")
                    print(f"\tNumber of layers specified is larger than the number of layers in the model... 2*{self.args.sim_num_layers} > {self.llm_manager.num_layers}")
                    exit()
                num_layers = self.args.sim_num_layers
                self.args.sim_num_layers *= 2
            else: # dense model
                if self.args.sim_num_layers > self.llm_manager.num_layers:
                    print(f"WARNING: Number of layers specified is larger than the number of layers in the model... {self.args.sim_num_layers} > {self.llm_manager.num_layers}")
                    exit()
                num_layers = self.args.sim_num_layers

        for layer in range(num_layers):
            self.mem_instr += cp.deepcopy(self.mem_instr_base)
            self.comp_instr += cp.deepcopy(self.comp_instr_base)
            self.net_instr += cp.deepcopy(self.net_instr_base)

        # step 2: add embedding layer
        emb_instr = [{"name":"emb", "layer": -1, "bytes":self.mem_instr_base[0]["bytes"]}]
        self.mem_instr = emb_instr + self.mem_instr

        # step 3: add optional lm-head
        if self.args.sim_lm_head:
            self.mem_instr = self.mem_instr + self.mem_instr_base_lm_head
            
            C_SIZE = self.comp_instr_base_lm_head[-1]["batch_size"] * self.comp_instr_base_lm_head[-1]["out_dim"] * self.pistil_comp.on_chip_accum_dtype
            BUF_SIZE = self.sys_manager.sys_cfg["CORE"]["C_MATRIX_THRESHOLD"]*self.sys_manager.sys_cfg["CORE"]["GLOBAL_BUFFER_B"]
            if C_SIZE > BUF_SIZE:  # -1 is the reduction op but has same params
                # cut up the lm_head so it's multiple instructions
                vmm = self.comp_instr_base_lm_head[-2]
                reduction = self.comp_instr_base_lm_head[-1]
                DIFF = math.ceil(C_SIZE / BUF_SIZE)
                
                comp_flag = {"name": "flag-linear", "layer": -1, "op": "flag", "flag": "compute-swing"}
                net_wait = {"name": "wait-compute-swing", "layer": "lm_head", "op": "wait", "flag": "compute-swing"}

                net_swing = self.net_instr_base_lm_head[2:] + [self.net_instr_base_lm_head[2]]
                self.net_instr_base_lm_head = cp.deepcopy(self.net_instr_base_lm_head[:2])
                self.comp_instr_base_lm_head = cp.deepcopy(self.comp_instr_base_lm_head[:-2])

                for i in range(DIFF):
                    vmm_new = cp.deepcopy(vmm)
                    red_new = cp.deepcopy(reduction)
                    vmm_new["out_dim"] = math.floor(vmm_new["out_dim"] / DIFF)
                    red_new["out_dim"] = math.floor(red_new["out_dim"] / DIFF)
                    self.comp_instr_base_lm_head += [vmm_new] + [red_new] + [cp.deepcopy(comp_flag)]
                    self.net_instr_base_lm_head += cp.deepcopy(net_swing) + [cp.deepcopy(net_wait)]
            
            # print("##################")
            # for instr in self.comp_instr_base_lm_head:
            #     print(instr)
            # print("##################")
            # for instr in self.net_instr_base_lm_head:
            #     print(instr)

            self.comp_instr = self.comp_instr + self.comp_instr_base_lm_head
            self.net_instr = self.net_instr + self.net_instr_base_lm_head

    def run(self):
        USE_TQDM = False
        print("Running Transactional Simulator")
        '''
        - Simulation uses on a Compute-Chiplet granularity (not core)
        - Reductions are only within an Compute-Chiplet btw. cores
        '''

        self.pistil_comp.set_instructions(self.comp_instr)
        self.pistil_mem.set_instructions(self.mem_instr)
        self.pistil_net.set_instructions(self.net_instr)
        
        total_comp = len(self.pistil_comp.all_instr)
        total_mem = len(self.pistil_mem.all_instr)
        total_net = len(self.pistil_net.all_instr)

        prior_comp = total_comp
        prior_mem = total_mem
        prior_net = total_net

        # for instr in self.comp_instr_base:
            # print(instr)

        # Initialize tqdm bars with total, display in separate rows
        if USE_TQDM:
            pbar_comp = tqdm(total=total_comp, desc="Comp", position=0)
            pbar_mem = tqdm(total=total_mem, desc="Mem", position=1)
            pbar_net = tqdm(total=total_net, desc="Net", position=2)
            prev_done_comp = prev_done_mem = prev_done_net = 0

        STALLED = False
        stall_count = 0
        current_time = 0
        all_events = {"mem": [], "comp": [], "net": [], "cache": []}
        t1 = time.perf_counter()
        while self.pistil_comp.cur_instr != None or self.pistil_mem.cur_instr != None or self.pistil_net.cur_instr != None:
            dma_next_time = self.pistil_mem.run(current_time, all_events)
            comp_next_time = self.pistil_comp.run(current_time, all_events)
            net_next_time = self.pistil_net.run(current_time, all_events)

            dma_cache_next_time = self.pistil_mem.track_cache(current_time, all_events)

            next_time = min(comp_next_time, dma_next_time, net_next_time)

            # detect stalled simulation
            remaining_comp_instr = len(self.pistil_comp.all_instr)
            remaining_mem_instr = len(self.pistil_comp.all_instr)
            remaining_net_instr = len(self.pistil_comp.all_instr)
            if prior_comp == remaining_comp_instr and prior_mem == remaining_mem_instr and prior_net == remaining_net_instr and (next_time == current_time or next_time == torch.inf):
                stall_count += 1
                if stall_count == 10:
                    print("[Transactioanl Simulator] Error: Stalled Detected: Exiting")
                    if self.pistil_mem.cur_instr != None:
                        print("\tMem Stalled on Layer: ", self.pistil_mem.cur_instr["name"])
                    elif len(self.pistil_mem.all_instr) > 0:
                        print("\tMem Stalled on Layer: ", self.pistil_mem.all_instr[0]["name"])
                    
                    if self.pistil_comp.cur_instr != None:
                        print("\tCompute Stalled on Layer: ", self.pistil_comp.cur_instr["name"])
                    elif len(self.pistil_comp.all_instr) > 0:
                        print("\tCompute Stalled on Layer: ", self.pistil_comp.all_instr[0]["name"])
                        
                    if self.pistil_net.cur_instr != None:
                        print("\tNet Stalled on Layer: ", self.pistil_net.cur_instr["name"])
                    elif len(self.pistil_net.all_instr) > 0:
                        print("\tNet Stalled on Layer: ", self.pistil_net.all_instr[0]["name"])
                    STALLED = True
                    break
            else:
                stall_count = 0

            prior_comp = remaining_comp_instr
            prior_mem = remaining_mem_instr
            prior_net = remaining_net_instr


            if next_time == torch.inf:
                next_time = current_time
            else:
                current_time = next_time
                
            #if current_time == torch.inf:
            #    print("Inf Time Set...")
            #    exit()

            if USE_TQDM:
                # Compute how many instructions have been completed
                done_comp = total_comp - len(self.pistil_comp.all_instr)
                done_mem = total_mem - len(self.pistil_mem.all_instr)
                done_net = total_net - len(self.pistil_net.all_instr)

                # Update progress bars only by the delta
                pbar_comp.update(done_comp - prev_done_comp)
                pbar_mem.update(done_mem - prev_done_mem)
                pbar_net.update(done_net - prev_done_net)

                # Store current progress
                prev_done_comp = done_comp
                prev_done_mem = done_mem
                prev_done_net = done_net

        if USE_TQDM:
            pbar_comp.close()
            pbar_mem.close()
            pbar_net.close()

        t2 = time.perf_counter() 
        print("Finished Transactional Simulator")
        print("Total Time: %0.2fs" % (t2 - t1))

        return all_events
        
            

