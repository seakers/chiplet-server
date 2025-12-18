import os
import json
import math
import argparse

from pistil_transactor.lib.hblc import HBLC



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    # setup arguments
    parser.add_argument("--name",               type=str,   default=None)
    parser.add_argument("--base-sys-cfg",       type=str,   default=None)
    parser.add_argument("--output-config-dir",  type=str,   default=None)

    # Compute Knobs
    parser.add_argument("--tmacs",              type=int,   default=16,      choices=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    parser.add_argument("--tmac-buf-cap",       type=float, default=32,     choices=[32]) # 32 KB buffer
    parser.add_argument("--mem-buf-cap",        type=float, default=.5,     choices=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--net-buf-cap",        type=float, default=.25,     choices=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

    # Memory Knobs
    parser.add_argument("--mem-bank-groups",    type=int,   default=4,      choices=[1, 2, 3, 4]) 
    parser.add_argument("--mem-ch-per-layer",   type=int,   default=1,      choices=[1]) # needs to be 1 for shoreline assumption - otherwise num cores changes per chiplet
    parser.add_argument("--mem-ranks",          type=int,   default=4,      choices=[1, 2, 3, 4])
    parser.add_argument("--mem-frac-bank-cap",  type=float, default=1.0,    choices=[0.5, 0.75, 1.0]) # may be able to do 0.25? look into it

    # Scale Up
    parser.add_argument("--num-cus",            type=int,   default=64)

    # Sensitivity Studies
    parser.add_argument("--net-latency-cycles",         type=int,       default=None)
    parser.add_argument("--reduction-latency-cycles",   type=int,       default=None) # latency for inner ring reduction
    parser.add_argument("--reduction-bytes-per-cycle",  type=int,       default=None) # bw for inner ring reduction
    parser.add_argument("--sram-density",               type=int,       default=4.0) # MB/mm2
    parser.add_argument("--sram-read-energy",           type=float,     default=0.2) # pJ/bit
    parser.add_argument("--sram-write-energy",          type=float,     default=0.25) # pJ/bit
    
    parser.add_argument("--ucie-s-bandwidth-per-mm",    type=float,     default=None) # GB/s/mm
    parser.add_argument("--ucie-s-sub-energy",          type=float,     default=0.5) # energy for UCIe over substrate
    parser.add_argument("--ucie-s-pcb-energy",          type=float,     default=1.2) # energy for UCIe over PCB
    parser.add_argument("--power-delivery",             type=float,     default=0.15) # energy lost during power delivery

    args = parser.parse_args()


    if args.base_sys_cfg is None or not os.path.exists(args.base_sys_cfg):
        print(f"Error: Cannot find base system for template: --base-sys-config={args.base_sys_cfg}")
        exit()
    
    if args.output_config_dir is None or not os.path.exists(args.output_config_dir):
        print(f"Error: Cannot find base system for template: --output-config-dir={args.output_config_dir}")
        exit()
    
    if args.num_cus % 4 != 0:
        print(f"Error: num-cus should be a multiple of 4 to align with package requirements: --num-cus={args.num_cus}")
        exit()    

    if args.name == None:
        args.name = f"pistil-config-num_cus-{args.num_cus}-tmacs-{args.tmacs}-mem_buf_cap-{args.mem_buf_cap}-net_buf_cap-{args.net_buf_cap}-mem_bank_groups-{args.mem_bank_groups}-mem_ranks-{args.mem_ranks}- mem_frac_bank_cap-{args.mem_frac_bank_cap}"
    if ".json" not in args.name:
        args.name += ".json"

    with open(args.base_sys_cfg, 'r') as f:
        base_sys_cfg = json.load(f)
    
    fig_dir = ".".join(args.name.split(".")[:-1])
    fig_dir_path = os.path.join(args.output_config_dir, fig_dir)
    if not os.path.exists(fig_dir_path):
        os.mkdir(fig_dir_path)

    # overwrite base config
    base_sys_cfg["NUM_CHIPLETS"] = args.num_cus

    # Compute Knobs
    base_sys_cfg["CORE"]["TMAC_PER_CORE"] = args.tmacs
    base_sys_cfg["CORE"]["TMAC_BUFFER_B"] = args.tmac_buf_cap * 1024 # KB to B
    base_sys_cfg["CORE"]["MEMORY_BUFFER_B"] = args.mem_buf_cap * 1024 * 1024 # MB to B
    base_sys_cfg["CORE"]["GLOBAL_BUFFER_B"] = args.net_buf_cap * 1024 * 1024 # MB to B
    
    # Memory Knobs
    base_sys_cfg["MEMORY_CHIPLET"]["BANK_GROUPS"] = args.mem_bank_groups
    base_sys_cfg["MEMORY_CHIPLET"]["CH_PER_LAYER"] = args.mem_ch_per_layer
    base_sys_cfg["MEMORY_CHIPLET"]["RANKS"] = args.mem_ranks
    base_sys_cfg["MEMORY_CHIPLET"]["FRAC_BANK_CAP"] = args.mem_frac_bank_cap
    hblc = HBLC(NAME="HBM-CO", BANKS_PER_GROUP=args.mem_bank_groups, CH_PER_LAYER=args.mem_ch_per_layer, RANKS=args.mem_ranks, FRAC_BANK_CAP=args.mem_frac_bank_cap, COST_PER_GB_HBM=12)

    # Scale Up
    base_sys_cfg["RING_STATION"]["PKGS"] = int(args.num_cus / 4)

    # Sensitivity Studies
    if args.net_latency_cycles != None:
        base_sys_cfg["CORE"]["NET_LATENCY_CYCLES"] = args.net_latency_cycles
    if args.reduction_latency_cycles != None:
        base_sys_cfg["CORE"]["REDUCTION_LATENCY_CYCLES"] = args.reduction_latency_cycles
    if args.reduction_bytes_per_cycle != None:
        base_sys_cfg["CORE"]["REDUCTION_BYTES_PER_CYCLE"] = args.reduction_bytes_per_cycle

    base_sys_cfg["ENERGY"]["SRAM_R"] = args.sram_read_energy
    base_sys_cfg["ENERGY"]["SRAM_W"] = args.sram_write_energy

    if args.ucie_s_bandwidth_per_mm != None:
        base_sys_cfg["COMPUTE_CHIPLET"]["UCIE_BW_PER_MM"] = args.ucie_s_bandwidth_per_mm
    else:
        args.ucie_s_bandwidth_per_mm = base_sys_cfg["COMPUTE_CHIPLET"]["UCIE_BW_PER_MM"]
        
    base_sys_cfg["ENERGY"]["UCIE_S_SUB"] = args.ucie_s_sub_energy
    base_sys_cfg["ENERGY"]["UCIE_S_PCB_S"] = args.ucie_s_pcb_energy
    base_sys_cfg["ENERGY"]["UCIE_S_PCB_L"] = args.ucie_s_pcb_energy
    base_sys_cfg["ENERGY"]["POWER_DELIVERY"] = args.power_delivery

    # Defining parameters not in config
    buffer_width = 0.18 # mm - width of memory and network buffers
    FAN_OUT_BUFFER = 0.25 # mm - extra height on core for routing space for memory to core
    mem_shoreline = 11/4

    sram_density = args.sram_density
    mem_buffer_height = (args.mem_buf_cap / 2) / (buffer_width * sram_density)
    net_buffer_height = (args.net_buf_cap / 2) / (buffer_width * sram_density)
    tmac_buffer_height = base_sys_cfg["CORE"]["TMAC_BUFFER_B"] / (1024**2) / (buffer_width * sram_density)
    hp_ops_height = base_sys_cfg["CORE"]["DISTANCE"]["HP_HEIGHT"]

    tmac_height = base_sys_cfg["CORE"]["DISTANCE"]["TMAC_HEIGHT"] + hp_ops_height + tmac_buffer_height
    base_sys_cfg["CORE"]["DISTANCE"]["ACT_TO_HP"] = hp_ops_height / 2 + tmac_buffer_height / 2
    # distances which may need to be changed
    num_cores_per_shoreline = base_sys_cfg["COMPUTE_CHIPLET"]["CORES"]/2
    compute_shoreline = base_sys_cfg["CORE"]["WIDTH_MM"] * num_cores_per_shoreline
    
    # avg fanout assumes cores are center alligned (obviously)
    avg_fan_out = sum(range(1, int(num_cores_per_shoreline/2+1))) * (abs(mem_shoreline - compute_shoreline)/2 / (num_cores_per_shoreline/2)) / (num_cores_per_shoreline/2) 
    base_sys_cfg["CORE"]["DISTANCE"]["MC_TO_SP"] = avg_fan_out + mem_buffer_height/2 + FAN_OUT_BUFFER + buffer_width/2      # memory controller to scratchpad
    base_sys_cfg["CORE"]["DISTANCE"]["SP_TO_DC"] = mem_buffer_height/2 + tmac_height + buffer_width/2                     # scratchpad to bottom edge of stream dequantizer 
    base_sys_cfg["CORE"]["DISTANCE"]["TMAC_BUFFER_HEIGHT"] = tmac_buffer_height
    core_width = base_sys_cfg["CORE"]["WIDTH_MM"]
    compute_bus_height = base_sys_cfg["CORE"]["DISTANCE"]["COMPUTE_BUS_HEIGHT"] 
    center_to_tmac = ((core_width / 2) - buffer_width) / 2 + buffer_width / 2
    base_sys_cfg["CORE"]["DISTANCE"]["BUS_TO_TMAC_C"] = center_to_tmac
    if args.tmacs >= 1:
        base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X1"] = compute_bus_height/2 + center_to_tmac + 1*compute_bus_height/2 #+ 1*base_sys_cfg["CORE"]["DISTANCE"]["TMAC_HEIGHT"] / 2
    if args.tmacs >= 2:
        base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X2"] = compute_bus_height/2 + center_to_tmac + 2*compute_bus_height/2 #+ 2*base_sys_cfg["CORE"]["DISTANCE"]["TMAC_HEIGHT"] / 2
    if args.tmacs >= 3:
        base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X3"] = compute_bus_height/2 + 2*center_to_tmac + 3*compute_bus_height/2 #+ 3*base_sys_cfg["CORE"]["DISTANCE"]["TMAC_HEIGHT"] / 2
    if args.tmacs >= 4:
        base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X4"] = compute_bus_height/2 + 2*center_to_tmac + 4*compute_bus_height/2 #+ 4*base_sys_cfg["CORE"]["DISTANCE"]["TMAC_HEIGHT"]/2

    if args.tmacs > 4:
        for tmac in range(5, args.tmacs+1):
            if tmac % 2 == 1: # odd number of TMACs
                base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X{tmac}"] = base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X3"] + tmac_height * math.ceil((tmac-4)/2)
            else:
                base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X{tmac}"] = base_sys_cfg["CORE"]["DISTANCE"][f"CB_TO_TMAC_X4"] + tmac_height * math.ceil((tmac-4)/2)

    
    base_sys_cfg["CORE"]["DISTANCE"]["GB_TO_CB"] = tmac_height * math.ceil(max(0, args.tmacs-2)/2) + net_buffer_height/2 + buffer_width/2 # Global Buffer to Compute Bus
    base_sys_cfg["CORE"]["DISTANCE"]["NB_TO_GB_AVG"] =  net_buffer_height/2 + buffer_width/2 # Network Bus to Global Buffer

    base_sys_cfg["COMPUTE_CHIPLET"]["UCIE_BW_PER_MM"] = args.ucie_s_bandwidth_per_mm
    

    # height = mem_buffer + tmac + compute_bus + n*tmacs + net_buffer + net_bus + ctrl
    core_height =   mem_buffer_height + \
                    net_buffer_height + \
                    compute_bus_height + \
                    tmac_height * math.ceil(args.tmacs/2) + \
                    base_sys_cfg["CORE"]["DISTANCE"]["CTRL_HEIGHT"]
    
    # solve for network bus height based on core height
    ucie_bw_per_mm = base_sys_cfg["COMPUTE_CHIPLET"]["UCIE_BW_PER_MM"]
    clk_freq = base_sys_cfg["CORE"]["NET_BUS_FREQ"]
    wire_per_mm = base_sys_cfg["CORE"]["WIRES_PER_UM"] * 1000
    
    net_bus_height = (ucie_bw_per_mm * core_height) / ((clk_freq * wire_per_mm)/8 - ucie_bw_per_mm)  # bits to bytes
    
    base_sys_cfg["CORE"]["HEIGHT_MM"] = core_height + net_bus_height # add buffer on top and bottom
    base_sys_cfg["COMPUTE_CHIPLET"]["UCIE_SHORELINE"] = (core_height + net_bus_height)*2
    base_sys_cfg["COMPUTE_CHIPLET"]["HEIGHT_MM"] = 2*(core_height + net_bus_height + FAN_OUT_BUFFER) 
    
    base_sys_cfg["CORE"]["NET_BYTES_PER_CYCLE"] = ucie_bw_per_mm * base_sys_cfg["CORE"]["HEIGHT_MM"] / (base_sys_cfg["COMPUTE_CHIPLET"]["CORES"]/2) / base_sys_cfg["CORE"]["NET_BUS_FREQ"]
                                        
    # Distance from one core to the next vertically - energy takes care of horizontal
    # (net buffer + network bus + ctrl) *2
    vertical_reduction = 2 * (net_buffer_height/2 + buffer_width/2 + net_bus_height + base_sys_cfg["CORE"]["DISTANCE"]["CTRL_HEIGHT"])
    base_sys_cfg["CORE"]["DISTANCE"]["C_TO_C_REDUCTION_HEIGHT"] = vertical_reduction

    # PKG Area
    # core height + memory height
    PACKAGE_OVERHEAD = 1.1
    hblc.calculate_cost()
    MEM_HEIGHT = hblc.HEIGHT
    CORE_HEIGHT = base_sys_cfg["CORE"]["HEIGHT_MM"]
    base_sys_cfg["PKG"]["WIDTH_MM"] = (MEM_HEIGHT * 2 + CORE_HEIGHT) * PACKAGE_OVERHEAD

    # save the new config
    with open(os.path.join(args.output_config_dir, args.name), 'w') as f:
        json.dump(base_sys_cfg, f, indent=4, sort_keys=True)

    

    from visualize.visualize_system import RPU

    rpu = RPU(
        core_width          = base_sys_cfg["CORE"]["WIDTH_MM"],
        core_height         = base_sys_cfg["CORE"]["HEIGHT_MM"],
        buffer_width        = buffer_width,
        mem_buffer_height   = mem_buffer_height,
        compute_bus_height  = compute_bus_height,
        tmacs_per_core      = base_sys_cfg["CORE"]["TMAC_PER_CORE"],
        tmac_compute_height = base_sys_cfg["CORE"]["DISTANCE"]["TMAC_HEIGHT"],
        tmac_buffer_height  = tmac_buffer_height,
        tmac_hp_ops_height  = hp_ops_height,
        net_buffer_height   = net_buffer_height,
        net_bus_height      = net_bus_height,
        ctrl_height         = base_sys_cfg["CORE"]["DISTANCE"]["CTRL_HEIGHT"],
        num_cores_width     = int(base_sys_cfg["COMPUTE_CHIPLET"]["CORES"] / 2),
        num_cores_height    = 2,
        ucie_buffer         = (base_sys_cfg["COMPUTE_CHIPLET"]["WIDTH_MM"] - base_sys_cfg["CORE"]["WIDTH_MM"]*(base_sys_cfg["COMPUTE_CHIPLET"]["CORES"] / 2))/2,
        cu_width            = base_sys_cfg["COMPUTE_CHIPLET"]["WIDTH_MM"],
        cu_height           = base_sys_cfg["COMPUTE_CHIPLET"]["HEIGHT_MM"],
        mem_shoreline       = 11/4,
        mem_gap             = FAN_OUT_BUFFER,
        bank_height         = hblc.BANK_HEIGHT,
        bank_width          = hblc.BANK_WIDTH,
        banks_per_group     = hblc.BANKS_PER_GROUP,
        y_ctrl_height       = hblc.Y_CTRL_HEIGHT,
        tsv_height          = hblc.POWER_TSV_HEIGHT + hblc.IO_TSV_HEIGHT,
        hbm_co_height       = hblc.HEIGHT_MM,
        hbm_co_width        = hblc.WIDTH_MM,
    )

    import matplotlib.pyplot as plt

    ####################################################
    # Plot Core
    PLOT_WIDTH = 3
    CORE_WIDTH = base_sys_cfg["CORE"]["WIDTH_MM"]
    fix, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH*(CORE_HEIGHT/CORE_WIDTH)))
    rpu.plot_core(ax, x_origin=0, y_origin=0)

    ax.set_ylim(0, CORE_HEIGHT)
    ax.set_xlim(0, CORE_WIDTH)

    plt.tight_layout(pad=.1)
    plt.savefig(fig_dir_path + "/core-visual.pdf", dpi=100)
    plt.clf()
    ####################################################
    

    ####################################################
    # Plot Compute Chiplet
    CU_WIDTH = base_sys_cfg["COMPUTE_CHIPLET"]["WIDTH_MM"]
    CU_HEIGHT = base_sys_cfg["COMPUTE_CHIPLET"]["HEIGHT_MM"]
    PLOT_WIDTH = 3 * CU_WIDTH
    fix, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH*(CU_HEIGHT/CU_WIDTH)))
    rpu.plot_compute_chiplet(ax, x_origin=0, y_origin=CU_HEIGHT/2)

    ax.set_ylim(0, CU_HEIGHT)
    ax.set_xlim(0, CU_WIDTH)

    plt.tight_layout(pad=.1)
    plt.savefig(fig_dir_path + "/compute-chiplet-visual.pdf", dpi=100)
    plt.clf()
    ####################################################
    

    ####################################################
    # Plot HBM-CO
    PLOT_WIDTH = 2 * hblc.WIDTH_MM
    fix, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH*(hblc.HEIGHT_MM/hblc.WIDTH_MM)))
    rpu.plot_hbm_co(ax, x_origin=0, y_origin=0)

    ax.set_ylim(0, hblc.HEIGHT_MM)
    ax.set_xlim(0, hblc.WIDTH_MM)
    
    plt.tight_layout(pad=.1)
    plt.savefig(fig_dir_path + "/hbm-co-visual.pdf", dpi=100)
    plt.clf()
    ####################################################


    ####################################################
    # Plot Compute Unit
    CU_WIDTH = base_sys_cfg["COMPUTE_CHIPLET"]["WIDTH_MM"]
    CU_HEIGHT = base_sys_cfg["COMPUTE_CHIPLET"]["HEIGHT_MM"] + 2*hblc.HEIGHT_MM
    PLOT_WIDTH = 1.5 * CU_WIDTH
    fix, ax = plt.subplots(1, 1, figsize=(PLOT_WIDTH, PLOT_WIDTH*(CU_HEIGHT/CU_WIDTH)))
    rpu.plot_compute_unit(ax, x_origin=0, y_origin=CU_HEIGHT/2)

    ax.set_ylim(0, CU_HEIGHT)
    ax.set_xlim(0, CU_WIDTH)

    plt.tight_layout(pad=.1)
    plt.savefig(fig_dir_path + "/compute-unit-visual.pdf", dpi=100)
    plt.clf()
    ####################################################
