import math 

class SysEnergy:
    def __init__(self, sys_manager):
        self.sys_manager = sys_manager
        self.sys_cfg = sys_manager.sys_cfg
        self.hblc = sys_manager.hblc
    
    def scale_dict(self, d, scalar, op="*"):
        d_new = {}
        for key in d:
            if op == "*":
                d_new[key] = d[key]*scalar
            if op == "+":
                d_new[key] = d[key]+scalar
        return d_new

    def power_delivery(self, d):
        total = 0
        for key in d:
            total += d[key]
        d["pd"] = total * self.sys_cfg["ENERGY"]["POWER_DELIVERY"]
        return d

    def sum_dict(self, d):
        total = 0
        for key in d:
            total += d[key]
        return total

    def get_energy_memory_pipeline(self, total_bytes, exe_time_s):
        total_bits = total_bytes * 8
        energy_per_bit = self.hblc.calculate_energy()
        del energy_per_bit["total"]
        
        energy_per_mm_si = self.sys_cfg["ENERGY"]["PJ_PER_MM"]
        avg_dist_to_sram = self.sys_cfg["CORE"]["DISTANCE"]["MC_TO_SP"]

        energy_per_bit["mov-si"] = avg_dist_to_sram * energy_per_mm_si
        energy_per_bit["sram-w"] = self.sys_cfg["ENERGY"]["SRAM_W"]
        
        energy_total = self.scale_dict(energy_per_bit, total_bits, op="*")
        energy_total = self.power_delivery(energy_total)
        energy_total = self.scale_dict(energy_total, 10**-12, op="*") # to pJ

        power_total = self.scale_dict(energy_total, 1/exe_time_s)
        energy_total["total"] = self.sum_dict(energy_total)
        power_total["total"] = self.sum_dict(power_total)

        return energy_total, power_total
    
    def get_energy_memory_writeback(self, total_bytes, exe_time_s):
        total_bits = total_bytes * 8
        energy_per_bit = self.hblc.calculate_energy()
        del energy_per_bit["total"]
        
        # distance is from network/global buffer
        energy_per_mm_si = self.sys_cfg["ENERGY"]["PJ_PER_MM"]
        avg_dist_to_sram =  self.sys_cfg["CORE"]["DISTANCE"]["MC_TO_SP"] + \
                            self.sys_cfg["CORE"]["DISTANCE"]["SP_TO_DC"] + \
                            self.sys_cfg["CORE"]["DISTANCE"]["HP_HEIGHT"] + \
                            self.sys_cfg["CORE"]["DISTANCE"]["TMAC_HEIGHT"] + \
                            self.sys_cfg["CORE"]["DISTANCE"]["TMAC_BUFFER_HEIGHT"] + \
                            self.sys_cfg["CORE"]["DISTANCE"]["COMPUTE_BUS_HEIGHT"] + \
                            self.sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"]

        energy_per_bit["mov-si"] = avg_dist_to_sram * energy_per_mm_si
        energy_per_bit["sram-r"] = self.sys_cfg["ENERGY"]["SRAM_R"]
        
        energy_total = self.scale_dict(energy_per_bit, total_bits, op="*")
        energy_total = self.power_delivery(energy_total)
        energy_total = self.scale_dict(energy_total, 10**-12, op="*") # to pJ

        power_total = self.scale_dict(energy_total, 1/exe_time_s)
        energy_total["total"] = self.sum_dict(energy_total)
        power_total["total"] = self.sum_dict(power_total)

        return energy_total, power_total

    def get_energy_compute_pipeline_vmm(self, bs, in_dim, out_dim, exe_time_s, wei_dtype=0.5, on_chip_dtype=1.0):
        # three parts to this

        # part 1: weights
        # - read weights from memory-buffer                                                 (comp:wei-sram_r)
        # - move weights to stream-decoder                                                  (comp:wei-mov)
        # - stream decoder energy                                                           (comp:wei-dc)
        # - move weight tile to number of active vec-tiles (broadcast across batch size)    (comp:wei-mov)

        # part 2: activations
        # - read activations from network-buffer                                            (comp:net-sram_r)
        # - move activations from buffer to tile-mac (broadcast across batch size)          (comp:net-mov)

        # part 3: compute
        # - multiply tile vector                                                            (comp:tmac)


        # reused data
        energy_per_mm_si = self.sys_cfg["ENERGY"]["PJ_PER_MM"]
        max_tmacs = self.sys_cfg["CORE"]["TMAC_PER_CORE"]
        if bs < max_tmacs:
            cb_to_tmac = self.sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X{bs}"]
        else:
            cb_to_tmac = self.sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X{max_tmacs}"]


        # part 1: weights
        energy_per_bit_pre_sd = {
            "wei-sram_r": self.sys_cfg["ENERGY"]["SRAM_R"],
            "wei-mov": self.sys_cfg["CORE"]["DISTANCE"]["SP_TO_DC"] * energy_per_mm_si,
            "wei-dc": self.sys_cfg["ENERGY"]["SD_PJ_P_B_SCALAR"] * (on_chip_dtype/wei_dtype) # how much energy to convert a bit of quantized to dequantized scaled by num bits generated
        }
        energy_per_bit_post_sd = {
            "wei-mov": cb_to_tmac * energy_per_mm_si
        }
        total_bits_pre_dc = in_dim * out_dim * wei_dtype * 8
        total_bits_post_dc = in_dim * out_dim * on_chip_dtype * 8

        total_energy_wei = self.scale_dict(energy_per_bit_pre_sd, total_bits_pre_dc, op="*")
        energy_per_bit_post_sd = self.scale_dict(energy_per_bit_post_sd, total_bits_post_dc, op="*")

        total_energy_wei["wei-mov"] += energy_per_bit_post_sd["wei-mov"]

        
        # part 2: activations
        energy_per_bit_net = {
            "act-sram_r": 2*self.sys_cfg["ENERGY"]["SRAM_R"], # read from global buffer and act buffer
            "act-sram_w": self.sys_cfg["ENERGY"]["SRAM_W"], # write to act buffer
            "act-mov": (self.sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"] + cb_to_tmac) * energy_per_mm_si,
        }
        total_bits_act = (bs * in_dim * on_chip_dtype * 8)
        total_energy_net = self.scale_dict(energy_per_bit_net, total_bits_act, op="*")

        # part 3: compute
        num_tile_multiplies = bs * math.ceil(in_dim / (self.sys_cfg["CORE"]["TMAC_VECTOR_HEIGHT"])) * math.ceil(out_dim / (self.sys_cfg["CORE"]["TMAC_VECTOR_WIDTH"]))
        total_energy_comp = {
            "tmac": self.sys_cfg["ENERGY"]["TMAC"] * num_tile_multiplies
        }

        energy_total = {}
        for d in (total_energy_wei, total_energy_net, total_energy_comp):
            for k, v in d.items():
                energy_total[k] = energy_total.get(k, 0) + v
        
        # include power delivery and convert to pJ
        energy_total = self.power_delivery(energy_total)
        energy_total = self.scale_dict(energy_total, 10**-12, op="*") # to pJ

        power_total = self.scale_dict(energy_total, 1/exe_time_s)
        energy_total["total"] = self.sum_dict(energy_total)
        power_total["total"] = self.sum_dict(power_total)
        
        #print("#######################")
        #print("pJ/TFLOP: ", energy_total["total"] / (2 * bs * in_dim * out_dim / 10**12))
        #print("VMM Shape(", bs, in_dim, out_dim, ") TFLOPs/W: %0.2f" %(energy_total["total"] / (2 * bs * in_dim * out_dim / 10**12))**-1)
        #exit()
        
        return energy_total, power_total

    
    def get_energy_compute_pipeline_activation(self, bs, params, ops_per_param, exe_time_s, on_chip_hp_dtype):
        # part 1: activations - worst case is read from network buffer, probably read only from act-buffer

        # part 2: compute
        # - multiply tile hp op   (comp:hp-op)

        energy_per_mm_si = self.sys_cfg["ENERGY"]["PJ_PER_MM"]

        max_tmacs = self.sys_cfg["CORE"]["TMAC_PER_CORE"]

        # > 32kB act buffer size
        num_bits = bs * params * on_chip_hp_dtype * 8
        if bs * params * on_chip_hp_dtype > (self.sys_cfg["CORE"]["TMAC_BUFFER_B"])*max_tmacs*self.sys_manager.get_cores_per_chiplet():
            energy_total = {
                "act-mov": (self.sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"]/2 + self.sys_cfg["CORE"]["DISTANCE"]["BUS_TO_TMAC_C"]) * energy_per_mm_si * num_bits,
                "hp-op": self.sys_cfg["ENERGY"]["HP_VEC_OP"] * bs * params, # energy value is pJ/Op not pJ/VectorOp
                "act-sram_r": 2*self.sys_cfg["ENERGY"]["SRAM_R"] * num_bits,
                "act-sram_w": self.sys_cfg["ENERGY"]["SRAM_W"] * num_bits,
            }
        else:
            energy_total = {
                # 0.02670138888888889 mm
                "act-mov": self.sys_cfg["CORE"]["DISTANCE"]["ACT_TO_HP"] * energy_per_mm_si * num_bits,
                "hp-op": self.sys_cfg["ENERGY"]["HP_VEC_OP"] * bs * params, # energy value is pJ/Op not pJ/VectorOp
                "act-sram_r": 2*self.sys_cfg["ENERGY"]["SRAM_R"] * num_bits,
                "act-sram_w": self.sys_cfg["ENERGY"]["SRAM_W"] * num_bits,
            }

        energy_total = self.power_delivery(energy_total)
        energy_total = self.scale_dict(energy_total, 10**-12, op="*") # to pJ

        power_total = self.scale_dict(energy_total, 1/exe_time_s)
        energy_total["total"] = self.sum_dict(energy_total)
        power_total["total"] = self.sum_dict(power_total)
        return energy_total, power_total
    
    def get_energy_compute_reduction(self, bs, params, exe_time_s, on_chip_accum_dtype):
        energy_per_mm_si = self.sys_cfg["ENERGY"]["PJ_PER_MM"]

        energy_total = {}
        # part 1: reduction distance (mm) 
        # compute_path = (self.sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"] + self.sys_cfg["CORE"]["DISTANCE"]["CB_TO_TMAC_X1"] + self.sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"])* self.sys_manager.get_cores_per_chiplet()
        compute_path = (2*self.sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"] + self.sys_cfg["CORE"]["DISTANCE"]["CB_TO_TMAC_X1"]) * self.sys_manager.get_cores_per_chiplet()
        dist = compute_path + self.sys_cfg["CORE"]["WIDTH_MM"] * self.sys_manager.get_cores_per_chiplet() + 2*self.sys_cfg["CORE"]["DISTANCE"]["C_TO_C_REDUCTION_HEIGHT"] # *2 for both sides of reduction
        num_bits = bs * params * on_chip_accum_dtype * 8
        energy_total["act-mov"] = dist * energy_per_mm_si * num_bits # each core is moving that amount of data
        energy_total["act-sram_r"] = (self.sys_cfg["ENERGY"]["SRAM_R"]) * num_bits
        energy_total["act-sram_w"] = (2*self.sys_cfg["ENERGY"]["SRAM_W"]) * num_bits

        # part 2: reduction compute in hp
        num_adds_per_core = bs * params
        num_adds_per_chiplet = num_adds_per_core * self.sys_manager.get_cores_per_chiplet()
        energy_total["hp-op"] = self.sys_cfg["ENERGY"]["HP_VEC_OP"] * num_adds_per_chiplet # energy value is pJ/Op not pJ/VectorOp]

        # include power delivery and convert to pJ
        energy_total = self.power_delivery(energy_total)
        energy_total = self.scale_dict(energy_total, 10**-12, op="*") # to pJ

        power_total = self.scale_dict(energy_total, 1/exe_time_s)
        energy_total["total"] = self.sum_dict(energy_total)
        power_total["total"] = self.sum_dict(power_total)

        return energy_total, power_total
    

    def get_energy_compute_pipeline_act_weight(self, bs, params, exe_time_s, on_chip_dtype):
        energy_per_mm_si = self.sys_cfg["ENERGY"]["PJ_PER_MM"]
        
        max_tmacs = self.sys_cfg["CORE"]["TMAC_PER_CORE"]
        if bs < max_tmacs:
            cb_to_tmac = self.sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X{bs}"]
        else:
            cb_to_tmac = self.sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X{max_tmacs}"]


        energy_total = {}
        # part 1: reading weights from mem buffer
        energy_per_bit_wei = {
            "wei-sram_r": self.sys_cfg["ENERGY"]["SRAM_R"],
            "wei-mov": (self.sys_cfg["CORE"]["DISTANCE"]["SP_TO_DC"] + cb_to_tmac) * energy_per_mm_si,
        }

        total_bits = params * on_chip_dtype * 8
        total_energy_wei = self.scale_dict(energy_per_bit_wei, total_bits, op="*")

        # part 2: 
        # > 32kB act buffer size
        if bs * params * on_chip_dtype > (self.sys_cfg["CORE"]["TMAC_BUFFER_B"]*max_tmacs*self.sys_manager.get_cores_per_chiplet()):
            total_energy_act = {
                "act-mov": (self.sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"]/2 + self.sys_cfg["CORE"]["DISTANCE"]["BUS_TO_TMAC_C"]) * energy_per_mm_si * total_bits,
                "hp-op": self.sys_cfg["ENERGY"]["HP_VEC_OP"] * bs * params # energy value is pJ/Op not pJ/VectorOp
            }
        else:
            total_energy_act = {
                "act-mov": self.sys_cfg["CORE"]["DISTANCE"]["ACT_TO_HP"] * energy_per_mm_si * total_bits,
                "hp-op": self.sys_cfg["ENERGY"]["HP_VEC_OP"] * bs * params # energy value is pJ/Op not pJ/VectorOp
            }

        energy_total = {}
        for d in (total_energy_wei, total_energy_act):
            for k, v in d.items():
                energy_total[k] = energy_total.get(k, 0) + v

        # include power delivery and convert to pJ
        energy_total = self.power_delivery(energy_total)
        energy_total = self.scale_dict(energy_total, 10**-12, op="*") # to pJ

        power_total = self.scale_dict(energy_total, 1/exe_time_s)
        energy_total["total"] = self.sum_dict(energy_total)
        power_total["total"] = self.sum_dict(power_total)

        return energy_total, power_total
    
    def get_energy_network(self, total_vals, exe_time_s, dtype):
        # part 1: network IO - use avg. interconnect perf
        chiplets_per_pkg = self.sys_cfg["PKG"]["COMPUTE_CHIPLETS"]
        pkgs = self.sys_cfg["RING_STATION"]["PKGS"]
        strings = self.sys_cfg["RING_STATION"]["STRINGS"]
        pkgs_per_string = self.sys_cfg["RING_STATION"]["PKGS_PER_STRING"]

        sub_conn = (chiplets_per_pkg-1)*pkgs
        pcb_conn = (pkgs_per_string - 1)*strings
        pcb_long = int(strings / 2) + strings
        total_conn = sub_conn + pcb_conn + pcb_long

        avg_io_pj_p_bit = (sub_conn * self.sys_cfg["ENERGY"]["UCIE_S_SUB"] + pcb_conn * self.sys_cfg["ENERGY"]["UCIE_S_PCB_S"] + pcb_long * self.sys_cfg["ENERGY"]["UCIE_S_PCB_L"]) / total_conn


        # part 2: distance from network-IO to Global-Buffer
        energy_per_mm_si = self.sys_cfg["ENERGY"]["PJ_PER_MM"]
        dist_vert = (self.sys_cfg["COMPUTE_CHIPLET"]["UCIE_SHORELINE"] / 2) / 2 * 2 # first /2 is top and bottom, second /2 is avg distance, third *2 is both sides
        dist_hor = self.sys_cfg["COMPUTE_CHIPLET"]["WIDTH_MM"]
        dist_core = self.sys_cfg["CORE"]["DISTANCE"]["NB_TO_GB_AVG"] * 2 # *2 for going in and out of core
        dist = dist_vert + dist_hor + dist_core 

        # part 3: write to sram buffer
        sram_w_pj_per_b = self.sys_cfg["ENERGY"]["SRAM_W"]

        energy_total = {}
        energy_total["io"] = avg_io_pj_p_bit
        energy_total["act-mov"] = dist * energy_per_mm_si
        energy_total["sram_w"] = sram_w_pj_per_b
        
        total_bits = total_vals * dtype
        
        # include power delivery and convert to pJ
        energy_total = self.scale_dict(energy_total, total_bits, op="*")
        energy_total = self.power_delivery(energy_total)
        energy_total = self.scale_dict(energy_total, 10**-12, op="*") # to pJ

        power_total = self.scale_dict(energy_total, 1/exe_time_s)
        energy_total["total"] = self.sum_dict(energy_total)
        power_total["total"] = self.sum_dict(power_total)

        return energy_total, power_total

