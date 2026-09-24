import json

from pistil_transactor.lib.sys_energy import SysEnergy
from pistil_transactor.lib.hblc import HBLC


class PistilSystemManager:
    def __init__(self, args):
        self.args = args

        with open(args.sys_cfg, 'r') as f:
            sys_cfg = json.load(f)
        self.sys_cfg = sys_cfg

        self.override_params()

        self.hblc = HBLC(
            NAME="HBLC", 
            BANKS_PER_GROUP = self.sys_cfg["MEMORY_CHIPLET"]["BANK_GROUPS"], 
            CH_PER_LAYER    = self.sys_cfg["MEMORY_CHIPLET"]["CH_PER_LAYER"],
            RANKS           = self.sys_cfg["MEMORY_CHIPLET"]["RANKS"],
            FRAC_BANK_CAP   = self.sys_cfg["MEMORY_CHIPLET"]["FRAC_BANK_CAP"],
            COST_PER_GB_HBM = self.sys_cfg["COST"]["COST_PER_GB_HBM"]
        )
        self.print_summary()
        self.sys_energy = SysEnergy(self)
    
    def override_params(self):
        # set chiplet parameters
        sim_num_chiplets = self.args.sim_num_chiplets

        if sim_num_chiplets % 8 != 0:
            print("Error: num chiplets must be multiple of 8 for 2x packages on Ring Station")
            exit()
        num_pkgs = int(sim_num_chiplets / 4)

        if num_pkgs % 4 == 0:
            num_strings = 4
            pkgs_per_string = int(num_pkgs / 4)
        elif num_pkgs % 2 == 0:
            num_strings = 2
            pkgs_per_string = int(num_pkgs / 2)
        else:
            num_strings = 1
            pkgs_per_string = num_pkgs

        self.sys_cfg["NUM_CHIPLETS"] = sim_num_chiplets
        self.sys_cfg["RING_STATION"]["PKGS"] = num_pkgs
        self.sys_cfg["RING_STATION"]["STRINGS"] = num_strings
        self.sys_cfg["RING_STATION"]["PKGS_PER_STRING"] = pkgs_per_string
        self.sys_cfg["RING_STATION"]["WIDTH_MM"] = 1.1 * (self.sys_cfg["PKG"]["WIDTH_MM"] * num_strings)
        self.sys_cfg["RING_STATION"]["HEIGHT_MM"] = 1.1 * (self.sys_cfg["PKG"]["HEIGHT_MM"] * pkgs_per_string)

        # override memory parameters
        # sim_bank_groups     = self.args.sim_bank_groups
        # sim_ch_per_layer    = self.args.sim_ch_per_layer
        # sim_ranks           = self.args.sim_ranks
        # sim_frac_bank_cap   = self.args.sim_frac_bank_cap

        # self.sys_cfg["MEMORY_CHIPLET"]["BANK_GROUPS"]   = sim_bank_groups
        # self.sys_cfg["MEMORY_CHIPLET"]["CH_PER_LAYER"]  = sim_ch_per_layer
        # self.sys_cfg["MEMORY_CHIPLET"]["RANKS"]         = sim_ranks
        # self.sys_cfg["MEMORY_CHIPLET"]["FRAC_BANK_CAP"] = sim_frac_bank_cap
        
        # self.sys_cfg["CORE"]["MEMORY_BUFFER_B"] = self.args.sim_on_chip_buffer
        # self.sys_cfg["CORE"]["NET_LATENCY_CYCLES"] = self.args.sim_net_latency_cycles


    def print_row(self):
        print("########################################")
    
    # ------- Pretty Print -------
    def print_summary(self):
        self.print_row()
        print("=== CORE PERFORMANCE ===")
        print(f"Clock Frequency         : {self.get_clk_freq() / 1e9:.2f} GHz")
        print(f"Compute Throughput      : {self.get_core_compute() / 1e9:.2f} GOPS")
        print(f"Memory Bandwidth        : {self.get_core_bandwidth() / 1e9:.2f} GB/s")
        print(f"Memory Capacity         : {self.get_core_capacity() / (1024**2):.2f} MB")
        print(f"Core-to-Core Net BW     : {self.get_network_core_to_core_bandwidth() / 1e9:.2f} GB/s\n")

        print("=== CHIPLET PERFORMANCE ===")
        print(f"Cores per Chiplet       : {self.get_cores_per_chiplet()}")
        print(f"Chiplet Compute         : {self.get_chiplet_compute() / 1e12:.2f} TOPS")
        print(f"Chiplet Bandwidth       : {self.get_chiplet_bandwidth() / 1e9:.2f} GB/s")
        print(f"Chiplet Capacity        : {self.get_chiplet_capacity() / (1024**2):.2f} MB\n")

        print("=== PACKAGE PERFORMANCE ===")
        print(f"Chiplets per Package    : {self.get_chiplets_per_package()}")
        print(f"Package Compute         : {self.get_package_compute() / 1e12:.2f} TOPS")
        print(f"Package Bandwidth       : {self.get_package_bandwidth() / 1e9:.2f} GB/s")
        print(f"Package Capacity        : {self.get_package_capacity() / (1024**2):.2f} MB\n")

        print("=== SYSTEM PERFORMANCE ===")
        print(f"Packages in System      : {self.get_packages_in_system()}")
        print(f"System Compute          : {self.get_system_compute() / 1e12:.2f} TOPS")
        print(f"System Bandwidth        : {self.get_system_bandwidth() / 1e12:.2f} TB/s")
        print(f"System Capacity         : {self.get_system_capacity() / (1024**3):.2f} GB\n")

        print("=== ENERGY ===")
        print(f"Energy per TMAC         : {self.get_energy_per_tmac():.2f} pJ")
        print(f"Energy per HP Vec Op    : {self.get_energy_per_hp_vec_op():.2f} pJ\n")

        print("=== COST ===")
        print(f"Silicon Cost (Chiplet)  : $ {self.get_cost_per_chiplet():.2f}")
        print(f"Memory Cost  (2xHBLC)   : $ {self.hblc.calculate_cost()*2:.2f}")
        print(f"Package Cost            : $ {self.get_cost_per_pkg():.2f}")
        print(f"\tSilicon Cost    : $ {self.get_cost_per_chiplet()*self.get_chiplets_per_package():.2f}")
        print(f"\tMemory Cost     : $ {self.hblc.calculate_cost()*2*self.get_chiplets_per_package():.2f}")
        print(f"\tSubstrate Cost  : $ {self.get_cost_per_pkg_sub():.2f}")
        print(f"System Cost             : $ {self.get_cost_per_system():.2f}")
        print(f"\tSilicon Cost    : $ {self.get_cost_per_chiplet()*self.get_chiplets_per_package()*self.get_packages_in_system():.2f}")
        print(f"\tMemory Cost     : $ {self.hblc.calculate_cost()*2*self.get_chiplets_per_package()*self.get_packages_in_system():.2f}")
        print(f"\tSubstrate Cost  : $ {self.get_cost_per_pkg_sub()*self.get_packages_in_system():.2f}")
        print(f"\tPCB Cost        : $ {self.get_cost_per_pcb():.2f}")

        self.print_row()
        self.hblc.print_report()

    # ------- Core-Level Getters -------
    def get_num_chiplets(self):
        return self.sys_cfg["NUM_CHIPLETS"]

    def get_clk_period_ns(self):
        return 1 / self.sys_cfg["CORE"]["CLK_FREQ"]

    def get_clk_freq(self):
        return self.sys_cfg["CORE"]["CLK_FREQ"] * (10**9)
    
    def get_net_bus_freq(self):
        return self.sys_cfg["CORE"]["NET_BUS_FREQ"] * (10**9)

    def get_tmacs_per_cores(self):
        return self.sys_cfg["CORE"]["TMAC_PER_CORE"]

    def get_core_compute(self):
        # 2 OPs/MAC * 64 TMACs / cycle * 4 TMACS / Core * 2 GHz
        return 2 * self.sys_cfg["CORE"]["TMAC_OPS_PER_CYCLE"] * self.get_tmacs_per_cores() * self.get_clk_freq()

    def get_core_bandwidth(self):
        return self.hblc.mem_bytes_per_ns() * (10**9)
    
    def get_hp_ops_latency(self):
        return self.sys_cfg["CORE"]["HP_OPS_LATENCY"] * self.get_clk_period_ns()

    def get_core_on_chip_memory_buffer_capacity(self):
        return self.sys_cfg["CORE"]["MEMORY_BUFFER_B"] 
    
    def get_core_capacity(self):
        return self.hblc.calculate_capacity_per_ch() * (1024**3)
    
    def get_network_core_to_core_bandwidth(self):
        return self.sys_cfg["CORE"]["NET_BYTES_PER_CYCLE"] * self.get_net_bus_freq()

    def get_network_latency(self):
        return self.sys_cfg["CORE"]["NET_LATENCY_CYCLES"] * self.get_clk_period_ns()

    def get_tile_height(self):
        return self.sys_cfg["CORE"]["TMAC_VECTOR_HEIGHT"]

    def get_tile_width(self):
        return self.sys_cfg["CORE"]["TMAC_VECTOR_WIDTH"]
    
    def get_reduction_bandwidth(self):
        return self.sys_cfg["CORE"]["REDUCTION_BYTES_PER_CYCLE"] 
    
    def get_reduction_latency(self):
        return self.sys_cfg["CORE"]["REDUCTION_LATENCY_CYCLES"] * self.get_clk_period_ns()
    
    # ------- Chiplet-Level Getters -------
    def get_cores_per_chiplet(self):
        return self.sys_cfg["COMPUTE_CHIPLET"]["CORES"]

    def get_chiplet_compute(self):
        return self.get_cores_per_chiplet() * self.get_core_compute()

    def get_chiplet_bandwidth(self):
        return self.get_cores_per_chiplet() * self.get_core_bandwidth()

    def get_chiplet_capacity(self):
        return self.get_cores_per_chiplet() * self.get_core_capacity()

    def get_chiplet_network_bandwidth(self):
        return self.get_cores_per_chiplet() * self.get_network_core_to_core_bandwidth()

    # ------- Package-Level Getters -------
    def get_chiplets_per_package(self):
        return self.sys_cfg["PKG"]["COMPUTE_CHIPLETS"]

    def get_package_compute(self):
        return self.get_chiplets_per_package() * self.get_chiplet_compute()

    def get_package_bandwidth(self):
        return self.get_chiplets_per_package() * self.get_chiplet_bandwidth()

    def get_package_capacity(self):
        return self.get_chiplets_per_package() * 2 * self.hblc.calculate_capacity()
    

    # ------- System-Level Getters -------
    def get_packages_in_system(self):
        return self.sys_cfg["RING_STATION"]["PKGS"]

    def get_system_compute(self):
        return self.get_packages_in_system() * self.get_package_compute()

    def get_system_bandwidth(self):
        return self.get_packages_in_system() * self.get_package_bandwidth()

    def get_system_capacity(self):
        return self.get_packages_in_system() * self.get_package_capacity()
    

    # ------- Energy Metrics -------
    def get_energy_per_tmac(self):
        return self.sys_cfg["ENERGY"]["TMAC"]

    def get_energy_per_hp_vec_op(self):
        return self.sys_cfg["ENERGY"]["HP_VEC_OP"]

    

    # ------- Cost Metrics -------
    def get_cost_per_chiplet(self):
        return self.sys_cfg["COST"]["N2_COST_PER_MM"] * self.sys_cfg["COMPUTE_CHIPLET"]["WIDTH_MM"] * self.sys_cfg["COMPUTE_CHIPLET"]["HEIGHT_MM"] 
    
    def get_cost_per_pkg_sub(self):
        return self.sys_cfg["COST"]["SUBSTRATE_COST_PER_MM"] * self.sys_cfg["PKG"]["WIDTH_MM"] * self.sys_cfg["PKG"]["HEIGHT_MM"] 
    
    def get_cost_per_pkg(self):
        return self.get_cost_per_pkg_sub() + (self.get_cost_per_chiplet() + 2*self.hblc.calculate_cost()) * self.get_chiplets_per_package()

    def get_cost_per_pcb(self):
        return self.sys_cfg["COST"]["PCB_COST_PER_MM"] * self.sys_cfg["RING_STATION"]["WIDTH_MM"] * self.sys_cfg["RING_STATION"]["HEIGHT_MM"] 

    def get_cost_per_system(self):
        return self.get_cost_per_pkg() * self.sys_cfg["RING_STATION"]["PKGS"] + self.get_cost_per_pcb()
    
