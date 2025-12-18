import csv
import math
import numpy as np
from collections import defaultdict

def compute_execution_stats(all_events, sys_manager, time_unit_div=1e3, cap_norm=1):
    stats = {}

    end_times = [event["end_time"] for hw in ["mem", "comp", "net"] for event in all_events[hw]]

    # Normalize time
    for hw in ["mem", "comp", "net"]:
        for event in all_events[hw]:
            event["start_time"] /= time_unit_div
            event["end_time"] /= time_unit_div

    # Total latency (in us)
    end_times = [event["end_time"] for hw in ["mem", "comp", "net"] for event in all_events[hw]]

    stats["total_latency_s"] = max(end_times) / 10**6

    # Peak on-chip buffer capacity in MB
    cache_events = [(ts / time_unit_div, kb) for ts, kb in all_events["cache"]]
    max_cache_kb = max([kb for _, kb in cache_events])
    avg_cache_kb = np.mean([kb for _, kb in cache_events]) if cache_events else 0.0
    stats["peak_buffer_MB"] = max_cache_kb / 1024.0
    stats["avg_cache_util_MB"] = avg_cache_kb / 1024.0

    # Compute total energy and average power from power trace
    '''
    power_trace = defaultdict(lambda: defaultdict(float))
    domain_energy = defaultdict(float)
    domain_total_power = defaultdict(float)
    print("Calculating power")
    for domain in ["mem", "comp", "net"]:
        for event in all_events[domain]:
            st_us = event["start_time"]
            et_us = event["end_time"]
            duration_us = et_us - st_us
            duration_ns = int(duration_us * 1000)
            if "power" not in event:
                continue
            for ns_offset in range(duration_ns):
                time_idx = int(st_us * 1000) + ns_offset
                for comp, val in event["power"].items():
                    if comp != "total":
                        power_trace[time_idx][(domain, comp)] += val
                        domain_total_power[domain] += val
                        domain_energy[domain] += val * 1e-9  # Joules

    all_time_indices = sorted(power_trace.keys())
    total_energy = 0.0
    total_power = 0.0
    for t in all_time_indices:
        total_power_t = sum(power_trace[t].values())
        total_energy += total_power_t * 1e-9  # Convert to Joules
        total_power += total_power_t
    avg_power = total_power / len(all_time_indices) if all_time_indices else 0

    stats["total_energy_J"] = total_energy
    stats["average_power_W"] = avg_power


    for domain in ["mem", "comp", "net"]:
        if all_time_indices:
            stats[f"average_power_{domain}_W"] = domain_total_power[domain] / len(all_time_indices)
        else:
            stats[f"average_power_{domain}_W"] = 0.0
    '''

    domain_energy = defaultdict(float)
    domain_total_power = defaultdict(float)
    domain_duration = 0.0
    for domain in ["mem", "comp", "net"]:
        domain_energy[domain] = 0
        for event in all_events[domain]:
            if "energy" in event:
                for comp, val in event["energy"].items():
                    if comp != "total":
                        domain_energy[domain] += val
            if "power" in event:
                st_s = event["start_time"] / 1e6
                et_s = event["end_time"] / 1e6
                duration_s = (et_s - st_s)  # us to s
                domain_duration = max(domain_duration, et_s)
                for comp, val in event["power"].items():
                    if comp != "total":
                        domain_total_power[domain] += val * duration_s  # W × time

    # Total energy (J)
    stats["total_energy_J"] = sum(domain_energy.values()) # pJ → J

    # Average power (W)
    stats["average_power_W"] = stats["total_energy_J"] / domain_duration if domain_duration > 0 else 0.0

    # Domain-specific power
    for domain in ["mem", "comp", "net"]:
        stats[f"average_power_{domain}_W"] = (domain_total_power[domain] / domain_duration) if domain_duration > 0 else 0.0

    # Compute average utilization
    def average_util(events):
        total_util = 0.0
        total_time = 0.0
        for event in events:
            if "util" in event:
                duration = event["end_time"] - event["start_time"]
                total_util += (event["util"] / 100.0) * duration
                total_time += duration
        return (total_util / total_time) if total_time else 0.0

    stats["avg_mem_util"] = average_util(all_events["mem"])
    stats["avg_comp_util"] = average_util(all_events["comp"])

    return stats

# Function to extract system + cost + HBM info into the stats dictionary
def add_system_cost_and_config_stats(stats, sys_manager):
    hblc = sys_manager.hblc
    hbm_cost = hblc.calculate_cost()
    hbm_energy = hblc.calculate_energy()
    hbm_capacity = hblc.calculate_capacity()
    hbm_bandwidth = hblc.calculate_bandwidth()

    # Cost metrics
    stats["chiplet_silicon_cost"] = sys_manager.get_cost_per_chiplet()
    stats["memory_cost_2xHBLC"] = hbm_cost * 2
    stats["package_cost"] = sys_manager.get_cost_per_pkg()
    stats["package_silicon_cost"] = sys_manager.get_cost_per_chiplet() * sys_manager.get_chiplets_per_package()
    stats["package_memory_cost"] = hbm_cost * 2 * sys_manager.get_chiplets_per_package()
    stats["package_substrate_cost"] = sys_manager.get_cost_per_pkg_sub()

    stats["system_cost"] = sys_manager.get_cost_per_system()
    stats["system_silicon_cost"] = sys_manager.get_cost_per_chiplet() * sys_manager.get_chiplets_per_package() * sys_manager.get_packages_in_system()
    stats["system_memory_cost"] = hbm_cost * 2 * sys_manager.get_chiplets_per_package() * sys_manager.get_packages_in_system()
    stats["system_substrate_cost"] = sys_manager.get_cost_per_pkg_sub() * sys_manager.get_packages_in_system()
    stats["system_pcb_cost"] = sys_manager.get_cost_per_pcb()

    # System scale metrics
    stats["num_packages"] = sys_manager.get_packages_in_system()
    stats["system_compute_TOPS"] = sys_manager.get_system_compute() / 1e12
    stats["system_bandwidth_TBps"] = sys_manager.get_system_bandwidth() / 1e12
    stats["system_capacity_GB"] = sys_manager.get_system_capacity() # / (1024**3)

    # HBLC-specific metrics
    stats["hblc_capacity_GB"] = hbm_capacity
    stats["hblc_bandwidth_GBps"] = hbm_bandwidth
    stats["hblc_bw_per_capacity"] = hbm_bandwidth / hbm_capacity
    stats["hblc_energy_total_pJ_per_bit"] = hbm_energy["total"]
    stats["hblc_energy_io_pJ_per_bit"] = hbm_energy["io"]
    stats["hblc_energy_tsv_pJ_per_bit"] = hbm_energy["tsvs"]
    stats["hblc_energy_movmem_pJ_per_bit"] = hbm_energy["mov-mem"]
    stats["hblc_energy_act_pJ_per_bit"] = hbm_energy["act"]
    stats["hblc_cost"] = hbm_cost
    stats["hblc_cost_per_GB"] = hbm_cost / hbm_capacity

    return stats

def save_stats_to_csv(stats, output_csv="execution_stats.csv"):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        for key, value in stats.items():
            writer.writerow([key, value])

def save_results(all_results, sys_manager, llm_manager, args, csv_filename, verbose=True):
    print(csv_filename)
    stats = compute_execution_stats(all_results, sys_manager)
    stats = add_system_cost_and_config_stats(stats, sys_manager)
    
    stats["model_name"] = llm_manager.model_name
    stats["num_layers"] = llm_manager.num_layers
    stats["sim_num_layers"] = args.sim_num_layers
    stats["batch_size"] = llm_manager.real_batch_size
    stats["prefill_batch"] = args.sim_prefill_batch
    stats["prefill_chunk_size"] = args.sim_prefill_chunk_size
    stats["prefill_cached"] = args.sim_prefill_cached
    stats["kv_cache"] = args.sim_kv_cache
    stats["w_dtype_B"] = args.w_dtype
    stats["kv_dtype_B"] = args.kv_dtype
    stats["norm_dtype_B"] = args.norm_dtype
    stats["sim_lm_head"] = args.sim_lm_head
    
    stats["num_chiplets"] = args.sim_num_chiplets
    stats["model_weight_capacity_GB"] = llm_manager.w_capacity / (1024**3)
    stats["kv_cache_capacity_per_batch_GB"] = llm_manager.kv_capacity / (1024**3)
    stats["total_model_capacity_GB"] = llm_manager.total_capacity / (1024**3)

    hblc = sys_manager.hblc
    stats["hblc_bank_groups"] = hblc.BANKS_PER_GROUP
    stats["hblc_ch_per_layer"] = hblc.CH_PER_LAYER
    stats["hblc_ranks"] = hblc.RANKS
    stats["hblc_frac_bank_cap"] = hblc.FRAC_BANK_CAP
    stats["cores_per_cu"] = sys_manager.sys_cfg["COMPUTE_CHIPLET"]["CORES"] 
    stats["mem_buffer_size"] = sys_manager.sys_cfg["CORE"]["MEMORY_BUFFER_B"] 
    stats["net_buffer_size"] = sys_manager.sys_cfg["CORE"]["GLOBAL_BUFFER_B"]
    stats["tmacs_per_core"] = sys_manager.sys_cfg["CORE"]["TMAC_PER_CORE"]

    # for key in stats:
    #     print(key, stats[key])
    
    # print results
    if verbose:
        num_cus = stats["num_chiplets"]
        if stats["sim_num_layers"] != stats["num_layers"] or stats["sim_num_layers"] != -1:
            latency_ms = stats["total_latency_s"] * stats["num_layers"] / stats["sim_num_layers"] * 1000
            epi = stats["total_energy_J"] * stats["num_layers"] / stats["sim_num_layers"]*1000
        else:
            latency_ms = stats["total_latency_s"] * 1000
            epi = stats["total_energy_J"]*1000

        print("###########################################")
        print("System Configuration")
        print(f"Num CUs:\t\t{num_cus}")
        print(f"System Compute:\t\t{stats["system_compute_TOPS"]:.2f} TOPs")
        print(f"System Memory BW:\t{stats["system_bandwidth_TBps"]:.2f} TB/s")
        print(f"System Memory Cap:\t{stats["system_capacity_GB"]:.2f} GB")
        print(f"System Avg. Power:\t{num_cus*stats["average_power_W"]:.2f} W [for this application]")
        print("")
        print("Core Configuration")
        print(f"Num TMACs/Core:\t\t{stats["tmacs_per_core"]}")
        print(f"Mem Buffer/Core:\t{(stats["mem_buffer_size"]/(1024*1024))} MB")
        print(f"Net Buffer/Core:\t{(stats["net_buffer_size"]/(1024*1024))} MB")
        print("")
        print("Memory Configuration")
        print(f"Num Ranks:\t\t{stats["hblc_ranks"]}")
        print(f"Num Banks / Bank Group:\t{(stats["hblc_bank_groups"])}")
        print(f"Fraction Bank Capacity:\t{(stats["hblc_frac_bank_cap"])}")
        print("###########################################")
        print("Application Configuration")
        print(f"Model:\t\t\t{stats["model_name"]}")
        print(f"Batch Size:\t\t{stats["batch_size"]}")
        print(f"Prefill Batch:\t\t{stats["prefill_batch"]}")
        if stats["prefill_batch"]:
            print(f"Prefill Chunk Size:\t{stats["prefill_chunk_size"]}")
            print(f"Prefill Cached Seq Len:\t{stats["prefill_cached"]}")
        print(f"KV Cache:\t\t{stats["kv_cache"]}")
        print(f"Sim Num Layers:\t\t{stats["sim_num_layers"]} / {stats["num_layers"]}")
        if stats["sim_num_layers"] != stats["num_layers"] or stats["sim_num_layers"] != -1:
            print("\t\t\t[WARNING] Sim layers not same as all layers, scaling latency and energy assuming same behavior")
        print(f"Sim LM Head:\t\t{stats["sim_lm_head"]} [Vocab projection layer at end?]")
        if stats["total_model_capacity_GB"] < stats["system_capacity_GB"]:
            print(f"\t\t\tCapacity Satisfied - Enough capacity for model and KV cache")
        else:
            print(f"[WARNING] Capacity Not Satisfied ({stats["total_model_capacity_GB"]} !< {stats["system_capacity_GB"]}) Not enough capacity for model and KV cache)")
        print(f"Bytes per Weight Param:\t{stats["w_dtype_B"]}")
        print(f"Bytes per KV$ Param:\t{stats["kv_dtype_B"]}")
        print("###########################################")
        print("General Performance Metrics")
        print(f"Latency per Token\t{latency_ms:.2f} ms")
        print(f"Energy per Inference\t{epi:.2f} mJ (BS={stats["batch_size"]})")
        print(f"Energy per Token\t{epi/stats["batch_size"]:.2f} mJ")
        print(f"Energy per Token\t{epi/args.sim_batch_size:.2f} mJ (w/ Prefill)")
        print(f"Average System Power\t{num_cus*stats["average_power_W"]:.2f} W")
        print(f"Estimated System Cost\t${stats["system_cost"]:.2f} (Area Based)")
        print("###########################################")
        print("Specific HW Metrics")
        print(f"Avg. Compute Util.\t{100*stats["avg_comp_util"]:.2f} %")
        print(f"Avg. Memory BW Util.\t{100*stats["avg_mem_util"]:.2f} %")
        print(f"Memory Capacity Util.\t{100*stats["total_model_capacity_GB"] / stats["system_capacity_GB"]:.2f} %")
        print(f"Avg. Mem. Buffer Util.\t{100*stats["avg_cache_util_MB"]/(stats["mem_buffer_size"]*stats["cores_per_cu"]/(1024*1024)):.2f} %")
        print(f"Peak Mem. Buffer Util.\t{100*stats["peak_buffer_MB"]/(stats["mem_buffer_size"]*stats["cores_per_cu"]/(1024*1024)):.2f} %")
        print(f"Avg. Sys Power Compute\t{num_cus*stats["average_power_comp_W"]:.2f} W")
        print(f"Avg. Sys Power Memory\t{num_cus*stats["average_power_mem_W"]:.2f} W")
        print(f"Avg. Sys Power Network\t{num_cus*stats["average_power_net_W"]:.2f} W")
        print("###########################################")
    

    save_stats_to_csv(stats, csv_filename)


