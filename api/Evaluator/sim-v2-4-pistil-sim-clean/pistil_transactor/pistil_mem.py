import math
import torch


class PistilMem:
    def __init__(self, sim, sys_manager, llm_manager, garden, pistil):
        self.sim                = sim
        self.sys_manager        = sys_manager
        self.llm_manager        = llm_manager
        self.garden             = garden
        self.pistil             = pistil

        ################################################################
        # system config
        self.mem_bw             = self.sys_manager.get_chiplet_bandwidth() / (10**9) # convert to B/ns
        self.mem_bw_wb          = self.mem_bw
        
        cores_per_chiplet       = self.sys_manager.get_cores_per_chiplet() 
        vector_height           = self.sys_manager.get_tile_height() 
        vector_width            = self.sys_manager.get_tile_width()
        
        mem_w_dtype             = self.llm_manager.w_dtype
        mem_kv_dtype            = self.llm_manager.kv_dtype
        mem_w_tile_bytes        = cores_per_chiplet * vector_height * vector_width * mem_w_dtype
        mem_kv_tile_bytes       = cores_per_chiplet * vector_height * vector_width * mem_kv_dtype
        
        self.ns_per_w_tile      = mem_w_tile_bytes / self.mem_bw * (10**9)
        self.ns_per_kv_tile     = mem_kv_tile_bytes / self.mem_bw * (10**9)
        ################################################################

        self.on_chip_dtype      = 2
        self.w_dtype            = self.llm_manager.w_dtype
        self.norm_dtype         = self.llm_manager.norm_dtype
        self.kv_dtype           = self.llm_manager.kv_dtype

        self.cache_interval     = 8 # sample time in ns
        self.cache_read         = 0

        # simulation granularity - directly impacts how sram capacity is captured
        self.max_bytes_per_trans = 256
        self.sram_capacity      = self.sys_manager.get_core_on_chip_memory_buffer_capacity() * cores_per_chiplet # in bytes
        self.all_instr          = None
        self.cur_instr          = None

        self.instr_start_time   = None
        self.next_time          = 0

        self.instr_tmp_info     = {}

        self.wb_instructions    = []
        self.wb_flags           = 0

        # flags
        self.flag_experts_ready = False

    def set_instructions(self, all_instr):
        self.all_instr = all_instr
        self.set_next_instr()
        self.next_time = self.start_instr(0)
    
    def set_next_instr(self):
        while len(self.all_instr) > 0 and self.all_instr[0]["name"] == "write-kv":
            self.wb_instructions.append(self.all_instr[0])
            self.all_instr = self.all_instr[1:]
        
        if self.wb_flags:
            self.cur_instr = self.wb_instructions[0]
            self.wb_instructions = self.wb_instructions[1:]
            self.instr_start_time = self.next_time
        else:
            if len(self.all_instr) > 0: # check if all instructions complete
                self.cur_instr = self.all_instr[0]
                self.all_instr = self.all_instr[1:]
                self.instr_start_time = self.next_time
            elif len(self.wb_instructions) > 0:
                self.cur_instr = self.wb_instructions[0]
                self.wb_instructions = self.wb_instructions[1:]
                self.instr_start_time = torch.inf
            else:
                self.cur_instr = None
                self.next_time = torch.inf
            
        self.instr_tmp_info = {}
    
    def start_instr(self, current_time):
        if self.cur_instr == None:
            return self.next_time
        
        name = self.cur_instr["name"]
        layer_id = self.cur_instr["layer"]

        if name == "write-kv":
            if self.wb_flags > 0:
                if self.instr_start_time == torch.inf:
                    self.instr_start_time = current_time

                if "remaining_bytes" not in self.cur_instr:
                    self.cur_instr["remaining_bytes"] = self.cur_instr["bytes"]
                
                write_bytes = self.cur_instr["bytes"]
                if write_bytes > self.max_bytes_per_trans:
                    write_bytes = self.max_bytes_per_trans
                self.cur_instr["remaining_bytes"] -= write_bytes

                next_time = current_time + (write_bytes / (self.mem_bw_wb))
                if "util" not in self.instr_tmp_info:
                    self.instr_tmp_info["util"] = []
                self.instr_tmp_info["util"].append([current_time, next_time, 1.0])
                return next_time
            else:
                return torch.inf

        if name == "wait-experts-ready" and self.cur_instr["op"] == "wait":
            if self.flag_experts_ready == False:
                return torch.inf
            else:
                return current_time + 1

        if "remaining_bytes" not in self.cur_instr:
            self.cur_instr["remaining_bytes"] = self.cur_instr["bytes"]
            if "QKT" in self.cur_instr["name"] and self.kv_dtype == 2: # will bug in old simulations. 
                self.cur_instr["remaining_bytes"] *= 2
            self.bytes_read = 0
            
        read_bytes = self.cur_instr["remaining_bytes"]
        if read_bytes > self.max_bytes_per_trans:
            read_bytes = self.max_bytes_per_trans
        
        # convert bytes to on-chip bytes
        if "norm" in name: # 
            on_chip_bytes = read_bytes #* self.on_chip_dtype / self.norm_dtype
        elif "QKT" in name:
            on_chip_bytes = read_bytes #* self.on_chip_dtype / self.kv_dtype
        else:
            on_chip_bytes = read_bytes #* self.on_chip_dtype / self.w_dtype
            
        # only thing read into activation buffer from memory
        if name == "emb":
            next_time = current_time + (read_bytes / self.mem_bw)
            if "util" not in self.instr_tmp_info:
                self.instr_tmp_info["util"] = []
            self.instr_tmp_info["util"].append([current_time, next_time, 1.0])
            return next_time
        
        # stall if SRAM buffer will overflow... need to wait for compute to remove something
        if (on_chip_bytes + self.sim.weight_buffer_used) > self.sram_capacity:
            # print("SRAM Buffer Full!")
            #print(self.cur_instr)
            #print(self.sim.weight_buffer)
            #exit()
            return torch.inf
        else:
            next_time = current_time + (read_bytes / self.mem_bw)
            if "util" not in self.instr_tmp_info:
                self.instr_tmp_info["util"] = []
            self.instr_tmp_info["util"].append([current_time, next_time, 1.0])
            return next_time

    def exe_instr(self, all_events):
        if self.cur_instr == None:
            return 
        
        name = self.cur_instr["name"]
        layer_id = self.cur_instr["layer"]

        if name == "write-kv":
            if self.cur_instr["remaining_bytes"] == 0:
                total_bytes = self.cur_instr["bytes"]
                instr_start = self.instr_start_time
                instr_end = self.next_time
                exe_time = instr_end - instr_start
                mem_bw = (total_bytes/(1024**3)/((exe_time)/(10**9)))
                bw_util = 100*mem_bw / (self.mem_bw*(10**9)/(1024**3))

                energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_memory_writeback(total_bytes, exe_time/(10**9))
                event = {"event": name, "start_time": instr_start, "end_time": instr_end, "mem_bw": mem_bw, "util": bw_util, "energy": energy_stats, "power": power_stats, "util-exact": self.instr_tmp_info["util"]}
                all_events["mem"].append(event)
                self.wb_flags -= 1
                self.set_next_instr()
            return 

        if name == "wait-experts-ready" and self.cur_instr["op"] == "wait":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["mem"].append(event)
            self.set_next_instr()
            self.flag_experts_ready = False 
            return

        read_bytes = self.cur_instr["remaining_bytes"]
        if read_bytes > self.max_bytes_per_trans:
            read_bytes = self.max_bytes_per_trans

        # convert bytes to on-chip bytes
        if "norm" in name: # 
            on_chip_bytes = read_bytes #* self.on_chip_dtype / self.norm_dtype
        elif "QKT" in name:
            on_chip_bytes = read_bytes #* self.on_chip_dtype / self.kv_dtype
        else:
            on_chip_bytes = read_bytes #* self.on_chip_dtype / self.w_dtype

        if name == "emb":
            self.sim.act_buffer["act-shard"] += on_chip_bytes * self.on_chip_dtype / self.w_dtype
        elif name == "rot_emb":
            pass
        else:
            if name not in self.sim.weight_buffer.keys():
                self.sim.weight_buffer[name] = 0
            self.sim.weight_buffer[name] += on_chip_bytes
            self.sim.weight_buffer_used += on_chip_bytes

        self.bytes_read += read_bytes
        self.cur_instr["remaining_bytes"] -= read_bytes
        if self.cur_instr["remaining_bytes"] == 0:
            total_bytes = self.bytes_read
            expected_bytes = self.cur_instr["bytes"]
            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            mem_bw = (total_bytes/(1024**3)/((exe_time)/(10**9)))
            bw_util = 100*mem_bw / (self.mem_bw*(10**9)/(1024**3))

            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_memory_pipeline(total_bytes, exe_time/(10**9))

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "mem_bw": mem_bw, "util": bw_util, "energy": energy_stats, "power": power_stats, "util-exact": self.instr_tmp_info["util"]}
            all_events["mem"].append(event)
            self.set_next_instr()
        
    
    def track_cache(self, current_time, all_events):
        # keep track of cache size
        if current_time >= self.cache_read:
            cache_used = self.sim.weight_buffer_used
            all_events["cache"].append([current_time, cache_used/(1024)]) # KB 
            self.cache_read += self.cache_interval
        return self.cache_read


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
        