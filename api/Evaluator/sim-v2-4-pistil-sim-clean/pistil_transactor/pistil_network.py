import math
import torch


class PistilNetwork:
    def __init__(self, sim, sys_manager, llm_manager, garden, pistil):
        self.sim                = sim
        self.sys_manager        = sys_manager
        self.llm_manager        = llm_manager
        self.garden             = garden
        self.pistil             = pistil

        ################################################################
        # system config
        self.c_to_c_bw          = self.sys_manager.get_chiplet_network_bandwidth() / (10**9)
        self.c_to_c_latency     = self.sys_manager.get_network_latency() # in nanoseconds
        ################################################################

        self.on_chip_bytes      = 2

        self.all_instr          = None
        self.cur_instr          = None

        self.instr_start_time   = None
        self.next_time          = 0

        # flags for communcication
        self.flag_norm          = False
        self.flag_linear        = False
        self.flag_qkv           = False
        self.flag_scores_max    = False
        self.flag_exp_sum       = False
        self.flag_q_reduction   = False
        self.flag_layer         = False

    def set_instructions(self, all_instr):
        self.all_instr = all_instr
        self.set_next_instr()
        self.next_time = self.start_instr(0)
    
    def set_next_instr(self):
        if len(self.all_instr) > 0:
            self.cur_instr = self.all_instr[0]
            self.all_instr = self.all_instr[1:]
            self.instr_start_time = self.next_time
        else:
            self.cur_instr = None
            self.next_time = torch.inf

    def start_instr(self, current_time):
        if self.cur_instr == None:
            return self.next_time

        name = self.cur_instr["name"]
        op = self.cur_instr["op"]

        # print("network: ", self.cur_instr)

        flag = self.cur_instr["flag"] if "flag" in self.cur_instr else None
        if op == "wait" and flag == "compute-norm-weights":
            if self.flag_norm == False:
                return torch.inf
            else:
                return current_time + 1
        if op == "wait" and flag == "compute-swing":
            if self.flag_linear == False:
                return torch.inf
            else:
                return current_time + 1
        if op == "wait" and flag == "compute-qkv-shuffle":
            if self.flag_qkv == False:
                return torch.inf
            else:
                return current_time + 1
        if op == "wait" and flag == "compute-scores-max":
            if self.flag_scores_max == False:
                return torch.inf
            else:
                return current_time + 1
        if op == "wait" and flag == "compute-exp-sum":
            if self.flag_exp_sum == False:
                return torch.inf
            else:
                return current_time + 1
        if op == "wait" and flag == "compute-q-reduction":
            if self.flag_q_reduction == False:
                return torch.inf
            else:
                return current_time + 1
        if op == "wait" and flag == "compute-layer-complete":
            if self.flag_layer == False:
                return torch.inf
            else:
                return current_time + 1

        if op == "flag" and flag == "network-qkv-shuffle":
            return current_time + 1
        if op == "flag" and flag == "network-scores-max":
            return current_time + 1
        if op == "flag" and flag == "network-exp-sum":
            if self.sim.pistil_comp.flag_exp_sum:
                return torch.inf
            else:
                return current_time + 1
        if op == "flag" and flag == "network-q":
            return current_time + 1
        if op == "flag" and flag == "network-norm-var":
            return current_time + 1
        if op == "flag" and flag == "network-layer-complete":
            return current_time + 1

        if op == "receive-fused-act-norm":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["params"] + self.cur_instr["variance_vals"]) * self.on_chip_bytes
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        if op == "receive-act":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["params"]) * self.on_chip_bytes
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        if op == "receive-norm":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["variance_vals"]) * self.on_chip_bytes
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        if op == "receive-kv": 
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["east_vals"] + self.cur_instr["west_vals"]) * self.on_chip_bytes
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        if op == "receive-q": 
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["east_vals"] + self.cur_instr["west_vals"]) * self.on_chip_bytes
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        if op == "receive-max":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["east_vals"] + self.cur_instr["west_vals"]) * self.on_chip_bytes
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        if op == "receive-exp-sum":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["east_vals"] + self.cur_instr["west_vals"]) * self.on_chip_bytes
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        if op == "receive-sdpa-p-sum":
            total_vals_receive = self.cur_instr["batch_size"] * (self.cur_instr["receive_vals"]) * self.on_chip_bytes
            total_vals_send = self.cur_instr["batch_size"] * (self.cur_instr["send_vals"]) * self.on_chip_bytes
            total_vals = max(total_vals_send, total_vals_receive)
            if total_vals / self.c_to_c_bw > self.c_to_c_latency:
                if len(self.all_instr) > 0 and self.all_instr[0]["op"] != op:
                    network_time = total_vals / self.c_to_c_bw + self.c_to_c_latency
                else:
                    network_time = total_vals / self.c_to_c_bw
            else:
                network_time = self.c_to_c_latency
            return current_time + network_time
        return torch.inf

    def exe_instr(self, all_events):
        if self.cur_instr == None:
            return 
        name = self.cur_instr["name"]
        op = self.cur_instr["op"]
        flag = self.cur_instr["flag"] if "flag" in self.cur_instr else None

        if op == "wait" and flag == "compute-norm-weights":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["net"].append(event)
            self.set_next_instr()
            self.flag_norm = False
            return 
        if op == "wait" and flag == "compute-swing":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["net"].append(event)
            self.set_next_instr()
            self.flag_linear = False
            return 
        if op == "wait" and flag == "compute-qkv-shuffle":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["net"].append(event)
            self.set_next_instr()
            self.flag_qkv = False
            return 
        if op == "wait" and flag == "compute-scores-max":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["net"].append(event)
            self.set_next_instr()
            self.flag_scores_max = False
            return 
        if op == "wait" and flag == "compute-exp-sum":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["net"].append(event)
            self.set_next_instr()
            self.flag_exp_sum = False
            return 
        if op == "wait" and flag == "compute-q-reduction":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["net"].append(event)
            self.set_next_instr()
            self.flag_q_reduction = False
            return 
        if op == "wait" and flag == "compute-layer-complete":
            instr_start = self.instr_start_time
            instr_end = self.next_time
            event = {"event": name, "start_time": instr_start, "end_time": instr_end}
            all_events["net"].append(event)
            self.set_next_instr()
            self.flag_layer = False
            return 
        
        if op == "flag" and flag == "network-qkv-shuffle":
            self.sim.pistil_comp.flag_qkv = True
            self.set_next_instr()
        if op == "flag" and flag == "network-scores-max":
            self.sim.pistil_comp.flag_scores_max = True
            self.set_next_instr()
        if op == "flag" and flag == "network-exp-sum":
            self.sim.pistil_comp.flag_exp_sum = True
            self.set_next_instr()
        if op == "flag" and flag == "network-q":
            self.sim.pistil_comp.flag_q_reduction = True
            self.set_next_instr()
        if op == "flag" and flag == "network-norm-var":
            self.sim.pistil_comp.flag_norm = True
            self.set_next_instr()
        if op == "flag" and flag == "network-layer-complete":
            self.sim.pistil_comp.flag_layer = True
            self.set_next_instr()
        

        if op == "receive-fused-act-norm":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["params"] + self.cur_instr["variance_vals"]) * self.on_chip_bytes
            self.sim.act_buffer["act-shard"] += total_vals

            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            bw_b_p_s = total_vals / (exe_time/10**9)
            util = bw_b_p_s / (self.sys_manager.get_chiplet_network_bandwidth())

            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_network(total_vals, exe_time/10**9, dtype=self.on_chip_bytes)

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": util*100, "energy": energy_stats, "power": power_stats}
            all_events["net"].append(event)
            self.set_next_instr()
            return 
        if op == "receive-act":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["params"]) * self.on_chip_bytes
            self.sim.act_buffer["act-shard"] += total_vals

            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            bw_b_p_s = total_vals / (exe_time/10**9)
            util = bw_b_p_s / (self.sys_manager.get_chiplet_network_bandwidth())
            
            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_network(total_vals, exe_time/10**9, dtype=self.on_chip_bytes)

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": util*100, "energy": energy_stats, "power": power_stats}
            all_events["net"].append(event)
            self.set_next_instr()
            return 
        if op == "receive-norm":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["variance_vals"]) * self.on_chip_bytes
            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            bw_b_p_s = total_vals / (exe_time/10**9)
            util = bw_b_p_s / (self.sys_manager.get_chiplet_network_bandwidth())

            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_network(total_vals, exe_time/10**9, dtype=self.on_chip_bytes)

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": util*100, "energy": energy_stats, "power": power_stats}
            all_events["net"].append(event)
            self.set_next_instr()
            return 
        if op == "receive-kv" or op == "receive-q":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["east_vals"] + self.cur_instr["west_vals"]) * self.on_chip_bytes
            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            bw_b_p_s = total_vals / (exe_time/10**9)
            util = bw_b_p_s / (self.sys_manager.get_chiplet_network_bandwidth())
            
            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_network(total_vals, exe_time/10**9, dtype=self.on_chip_bytes)

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": util*100, "energy": energy_stats, "power": power_stats}
            all_events["net"].append(event)
            self.set_next_instr()
            return 
        if op == "receive-max":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["east_vals"] + self.cur_instr["west_vals"]) * self.on_chip_bytes
            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            bw_b_p_s = total_vals / (exe_time/10**9)
            util = bw_b_p_s / (self.sys_manager.get_chiplet_network_bandwidth())

            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_network(total_vals, exe_time/10**9, dtype=self.on_chip_bytes)

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": util*100, "energy": energy_stats, "power": power_stats}
            all_events["net"].append(event)
            self.set_next_instr()
            return 
        if op == "receive-exp-sum":
            total_vals = self.cur_instr["batch_size"] * (self.cur_instr["east_vals"] + self.cur_instr["west_vals"]) * self.on_chip_bytes
            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            bw_b_p_s = total_vals / (exe_time/10**9)
            util = bw_b_p_s / (self.sys_manager.get_chiplet_network_bandwidth())

            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_network(total_vals, exe_time/10**9, dtype=self.on_chip_bytes)

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": util*100, "energy": energy_stats, "power": power_stats}
            all_events["net"].append(event)
            self.set_next_instr()
            return 
        if op == "receive-sdpa-p-sum":
            total_vals_receive = self.cur_instr["batch_size"] * (self.cur_instr["receive_vals"]) * self.on_chip_bytes
            total_vals_send = self.cur_instr["batch_size"] * (self.cur_instr["send_vals"]) * self.on_chip_bytes
            total_vals = max(total_vals_send, total_vals_receive)

            instr_start = self.instr_start_time
            instr_end = self.next_time
            exe_time = instr_end - instr_start
            bw_b_p_s = total_vals / (exe_time/10**9)
            util = bw_b_p_s / (self.sys_manager.get_chiplet_network_bandwidth())

            energy_stats, power_stats = self.sys_manager.sys_energy.get_energy_network(total_vals_send + total_vals_receive, exe_time/10**9, dtype=self.on_chip_bytes)

            event = {"event": name, "start_time": instr_start, "end_time": instr_end, "util": util*100, "energy": energy_stats, "power": power_stats}
            all_events["net"].append(event)
            self.set_next_instr()
            return 

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
        
        