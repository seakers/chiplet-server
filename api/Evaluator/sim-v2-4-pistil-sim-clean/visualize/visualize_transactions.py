import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict
from tqdm import tqdm

PRINT_POWER = False

def get_event_color_map(event_names, REDUCTION_COLOR, cmap_name="Pastel1", unique_events=7):
    cmap = cm.get_cmap(cmap_name, 7)
    my_color_order = []
    for i in [1, 3, 0, 4, 8, 5, 6, 5, 6, 5, 6, 2,]:
        my_color_order.append(cmap(i)) 

    color_map = {name: my_color_order[i] for i, name in enumerate(event_names)}
    color_map["reduction"] = REDUCTION_COLOR
    color_map["write-kv"] = '#E9E7E3'

    return color_map

def normalize_cache(cache_events, sys_manager, time_unit_div=1e3, cap_norm=1):
    max_cache_kb = sys_manager.get_core_on_chip_memory_buffer_capacity() * sys_manager.get_cores_per_chiplet() / 1024
    max_cache_kb = 1.0
    return [(ts / time_unit_div, kb / max_cache_kb / cap_norm) for ts, kb in cache_events]

def compute_power_trace(all_events, time_step_ns=1):
    import collections
    power_trace = collections.defaultdict(lambda: collections.defaultdict(float))  # time_idx -> (comp -> power)
    total_power_trace = collections.defaultdict(float)  # time_idx -> (comp -> power)

    domain_color_map = {
        "mem": plt.cm.copper,
        "comp": plt.cm.Blues,
        "net": plt.cm.pink
    }

    domain_list = ["mem", "comp", "net"]

    component_color_map = {}
    power_components_by_domain = defaultdict(set)

    # 1. Gather component names and colors
    for domain in domain_list:
        for event in all_events[domain]:
            if "power" in event:
                for comp in event["power"]:
                    if comp == "total":
                        continue
                    power_components_by_domain[domain].add(comp)
                # Print Power Information 
                if PRINT_POWER:
                    for key, item in event["power"].items():
                        if item  > 30:
                            print(event["event"], event["power"])

    # 2. Assign each component a unique color per domain
    for domain in domain_list:
        components = sorted(power_components_by_domain[domain])
        cmap = domain_color_map[domain]
        for i, comp in enumerate(components):
            component_color_map[(domain, comp)] = cmap((i + 1) / (len(components) + 1))

    # 3. Accumulate power over time
    for domain in domain_list:
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
                    if comp == "total":
                        continue
                    key = (domain, comp)
                    power_trace[time_idx][key] += val  # constant power over ns slices
                    total_power_trace[time_idx] += val
    
    Q99 = np.percentile(list(total_power_trace.values()), 99.99)
    return power_trace, component_color_map, Q99


# all time is in ns converted to us
def plot_execution_timeline(all_events, sys_manager, llm_manager, output_path="exe-timeline.png", time_unit_div=1e3, SHOW_LEGEND=True, case=0):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.cm as cm
    from matplotlib.patches import Patch

    # Normalize time and cache
    for hw in ["mem", "comp", "net"]:
        for event in all_events[hw]:
            event["start_time"] /= time_unit_div
            event["end_time"] /= time_unit_div
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 4.5), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 2]})

    HEIGHT = 1.0
    Y_OFFSET = 0.0
    FONT_SIZE = 14
    LEGEND_FONT_SIZE = 6
    XLABEL_SIZE = 15
    Y_LABEL_FONTSIZE = 12
    LEGEND_TITLE_SIZE = 8
    XTICK_SIZE = 12
    YTICK_SIZE = 7
    RECT_LINE_WIDTH = 0.2
    FIG_TITLE_SIZE = 18
    ANNOTATION_SIZE = 16
    STALL_COLOR = "#C7C3B8"
    REDUCTION_COLOR = "#C49C93"
    UKNOWN_COLOR = "#8C564B"
    UTIL_COLOR = "red"
    UTIL_LINE_WIDTH = 1.0
    THRESHOLD_PERCENT = 3.0


    if "llama4" in llm_manager.model_name:
        event_labels = {
            "wQKV": "wQKV",
            "QKT-0": ["K$", "QKᵀ"],
            "sQKT_V-0": ["V$", "s(QKᵀ)V"],
            "wO": "wO",
            "gate_up_proj": "wUp / wGate",
            "down_proj": "wDown",
            "gate_up_proj_s": "MoE-S\nwUp / wGate",
            "down_proj_s": "MoE-S\nwDown",
            "gate_up_proj_e": "MoE-R\nwUp / wGate",
            "down_proj_e": "MoE-R\nwDown",
            "reduction": "reduction",
            "write-kv": "WB$",
        }
    else:
        event_labels = {
            "wQKV": "wQKV",
            "QKT-0": ["K$", "QKᵀ"],
            "sQKT_V-0": ["V$", "s(QKᵀ)V"],
            "wO": "wO",
            "gate_up_proj": "wUp / wGate",
            "down_proj": "wDown",
            "reduction": "reduction",
            "write-kv": "WB$",
        }

    color_map = get_event_color_map(event_labels.keys(), REDUCTION_COLOR)

    max_time = max(event["end_time"] for hw in ["mem", "comp", "net"] for event in all_events[hw])
    mem_time = max(event["end_time"] for hw in ["mem"] for event in all_events[hw])

    def smooth_utilization_from_exact(event, neighboring_segments=100):
        st = event["start_time"]
        et = event["end_time"]
        
        active_ranges = event.get("util-exact", [])

        for i in range(len(active_ranges)):
            active_ranges[i][0] /= 1000
            active_ranges[i][1] /= 1000

        if event["event"] == "sQKT_V-0":
            for a in active_ranges:
                print(a)

        segments = []
        num_samples = len(active_ranges)
        for i in range(num_samples-1):
            # continuation scenario - end of current isn't start of next
            if active_ranges[i][1] != active_ranges[i+1][0]:
                some_util = (active_ranges[i][1] - active_ranges[i][0])
                zero_util = (active_ranges[i+1][1] - active_ranges[i+1][0])
                avg_util = some_util * active_ranges[i][2]/(some_util + zero_util)
                segments.append([active_ranges[i][0], active_ranges[i+1][0], avg_util])
            else:
                segments.append([active_ranges[i][0], active_ranges[i][1], active_ranges[i][2]])
        if event["event"] == "sQKT_V-0":
            print(event["event"], event["util"])
            for s in segments:
                print(s)
        '''
        if len(segments) > 0:
            segments.append([active_ranges[-1][0], active_ranges[-1][1], active_ranges[i][2]])
        else:
            print(event)
            segments.append([active_ranges[-1][0], active_ranges[-1][1], active_ranges[-1][2]])
        '''

        num_samples = len(segments)
        times = []
        util = []
        for i in range(num_samples):
            times.append([segments[i][0], segments[i][1]])
            util.append([segments[i][2], segments[i][2]])

        return times, util


    def plot_hw_events(ax_id, ax, events, label, show_util=True, ylimit=(0, 1.1), SHOW_LABEL=False, LABELS=None):
        for event in events:
            st, et = event["start_time"], event["end_time"]
            duration = et - st
            if "wait" in event["event"]:
                color = STALL_COLOR
            elif event["event"] not in color_map:
                color = UKNOWN_COLOR
            else:
                color = color_map[event["event"]]

            # Draw bar
            ax.add_patch(patches.Rectangle((st, Y_OFFSET), duration, HEIGHT,
                                           linewidth=RECT_LINE_WIDTH, edgecolor='black',
                                           facecolor=color))

            # Draw label only for long events
            if SHOW_LABEL and duration > max_time * (THRESHOLD_PERCENT / 100.0) and event["event"] in LABELS and et < max_time and llm_manager.args.sim_num_layers < 3:
                if ax_id == 0 and type(LABELS[event["event"]]) == type([]):
                    ax.text(st + duration / 2, 0.5, LABELS[event["event"]][0], va='center', ha='center', fontsize=FONT_SIZE)
                elif ax_id == 1 and type(LABELS[event["event"]]) == type([]):
                    ax.text(st + duration / 2, 0.5, LABELS[event["event"]][1], va='center', ha='center', fontsize=FONT_SIZE)
                else:
                    ax.text(st + duration / 2, 0.5, LABELS[event["event"]], va='center', ha='center', fontsize=FONT_SIZE)

            # Red utilization line
            if show_util and "util" in event:
                if show_util:
                    if "util-exact" in event and USE_SMOOTH:
                        times, smooth_utils = smooth_utilization_from_exact(event)
                        ax.plot(times, smooth_utils, linewidth=UTIL_LINE_WIDTH, color=UTIL_COLOR)
                    elif "util" in event:
                        util_val = event["util"] / 100.0
                        ax.plot([st, et], [util_val, util_val], linewidth=UTIL_LINE_WIDTH, color=UTIL_COLOR)
        ax.set_ylabel(label, fontsize=Y_LABEL_FONTSIZE)
        ax.set_ylim(*ylimit)
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='x', labelsize=XTICK_SIZE)
        ax.tick_params(axis='y', labelsize=YTICK_SIZE)

    # Memory / Compute / Network
    USE_SMOOTH = False
    SHOW_LABEL = True
    
    plot_hw_events(0, axes[0], all_events["mem"], "Mem.", show_util=True, ylimit=(-.01, 1.1), SHOW_LABEL=SHOW_LABEL, LABELS=event_labels)
    plot_hw_events(1, axes[1], all_events["comp"], "Comp.", show_util=True, ylimit=(-.01, 1.1), SHOW_LABEL=SHOW_LABEL, LABELS=event_labels)
    plot_hw_events(2, axes[2], all_events["net"], "Net.", show_util=True, ylimit=(-.01, 1.1), SHOW_LABEL=False, LABELS=event_labels)

    CAP_NORM = 1.0
    if max(all_events["cache"])[0] > 100:
        CAP_NORM = 1024

    # Cache (Dynamic Y range)
    all_events["cache"] = normalize_cache(all_events["cache"], sys_manager, time_unit_div, cap_norm=CAP_NORM)
    x_vals = [x[0] for x in all_events["cache"]]
    y_vals = [x[1] for x in all_events["cache"]]
    axes[3].plot(x_vals, y_vals, linewidth=1.2)
    
    if CAP_NORM == 1.0:
        axes[3].set_ylabel("Buf. (KB)", fontsize=Y_LABEL_FONTSIZE)
    else:
        axes[3].set_ylabel("Buf. (MB)", fontsize=Y_LABEL_FONTSIZE)
        max_cache = sys_manager.get_core_on_chip_memory_buffer_capacity() * sys_manager.get_cores_per_chiplet() / 1024 / 1024
        axes[3].plot([0, max_time], [max_cache, max_cache], "--", linewidth=0.8, color="red")
        #axes[3].set_yticks([0, 8])
    
    axes[3].grid(True, axis='x', linestyle='--', linewidth=0.5)
    axes[3].tick_params(axis='x', labelsize=XTICK_SIZE)
    axes[3].tick_params(axis='y', labelsize=YTICK_SIZE)

    for i in range(3):
        axes[i].text(-0.01, 0.5, "Util.", rotation=90, va='center', ha='center', fontsize=6, transform=axes[i].transAxes)

    axes[0].set_xlim(0, max_time * 1.05)

    if SHOW_LEGEND:
        legend_patches = [Patch(facecolor=color_map[event], label=event_labels[event][0] + " / " + event_labels[event][1] if type(event_labels[event]) == type([]) else event_labels[event] ) for event in event_labels]
        axes[0].legend(
            title="Kernel Labels", 
            handles=legend_patches, 
            bbox_to_anchor=(1.0, 2.1), 
            loc='upper left', 
            fontsize=LEGEND_FONT_SIZE, 
            frameon=False, 
            title_fontsize=LEGEND_TITLE_SIZE+1,
            handletextpad=0.3,     # Space between handle and text
            borderaxespad=0.3,     # Padding between legend and axes
            labelspacing=0.3,      # Vertical space between labels
        )


    # Power stack-up plot (fifth axis)
    print("\tComputing Power Trace")
    power_trace, power_colors, max_power_val_q99 = compute_power_trace(all_events)
    print("\tFinished Computing Power Trace")

    all_time_indices = sorted(power_trace.keys())
    stacked_components = sorted({k for t in all_time_indices for k in power_trace[t].keys()})

    bottom = np.zeros(len(all_time_indices))
    x = np.array(all_time_indices) / 1000.0  # convert ns to us for plotting
    
    print("\tPlotting Stacked Components")
    for comp in tqdm(stacked_components):
        y = np.array([power_trace[t].get(comp, 0.0) for t in all_time_indices])
        color = power_colors[comp]
        axes[4].fill_between(x, bottom, bottom + y, color=color, linewidth=0)
        bottom += y
    print("\tFinished Plotting Stacked Components")

    axes[4].set_ylabel("Power (W)", fontsize=Y_LABEL_FONTSIZE, labelpad=-.8)
    axes[4].set_ylim(0, max_power_val_q99*1.1)
    axes[4].grid(True, axis='x', linestyle='--', linewidth=0.5)
    axes[4].tick_params(axis='x', labelsize=XTICK_SIZE)
    axes[4].tick_params(axis='y', labelsize=YTICK_SIZE)



    # power_legend = [Patch(facecolor=power_colors[c], label=f"{c[0]}:{c[1]}")
    #             for c in stacked_components]
        

    comp_legend = []
    mem_legend = []
    net_legend = []
    for c in reversed(stacked_components):
        if c[0] == "comp":
            new_patch = Patch(facecolor=power_colors[c], label=f"{c[1]}")
            comp_legend.append(new_patch)
        if c[0] == "mem":
            new_patch = Patch(facecolor=power_colors[c], label=f"{c[1]}")
            mem_legend.append(new_patch)
        if c[0] == "net":
            new_patch = Patch(facecolor=power_colors[c], label=f"{c[1]}")
            net_legend.append(new_patch)

    if SHOW_LEGEND:
        axes[-3].legend(
            title="Mem Power", 
            handles=mem_legend, 
            bbox_to_anchor=(1.0, 2.2), 
            loc='upper left', 
            fontsize=LEGEND_FONT_SIZE, 
            frameon=False, 
            title_fontsize=LEGEND_TITLE_SIZE,
            handletextpad=0.3,     # Space between handle and text
            borderaxespad=0.3,     # Padding between legend and axes
            labelspacing=0.3,      # Vertical space between labels
        )
        axes[-2].legend(
            title="Comp Power", 
            handles=comp_legend, 
            bbox_to_anchor=(1.0, 1.05), # (1.13, 3.5),
            loc='upper left', 
            fontsize=LEGEND_FONT_SIZE, 
            frameon=False, 
            title_fontsize=LEGEND_TITLE_SIZE,
            handletextpad=0.3,     # Space between handle and text
            borderaxespad=0.3,     # Padding between legend and axes
            labelspacing=0.3,      # Vertical space between labels
        )
        axes[-1].legend(
            title="Net Power", 
            handles=net_legend, 
            bbox_to_anchor=(1.0, 0.3), # (1.13, 1.8), 
            loc='upper left', 
            fontsize=LEGEND_FONT_SIZE, 
            frameon=False, 
            title_fontsize=LEGEND_TITLE_SIZE,
            handletextpad=0.3,     # Space between handle and text
            borderaxespad=0.3,     # Padding between legend and axes
            labelspacing=0.3,      # Vertical space between labels
        )
        
    axes[-1].set_xlabel("Exe Time (us)", fontsize=XLABEL_SIZE)

    RADIUS = 0.08

    if llm_manager.args.sim_prefill_batch:
        axes[0].set_title(f"Pistil Multi-Chiplet LLM Simulation\n{llm_manager.args.sim_num_chiplets}-CUs | {llm_manager.model_name} | BS {llm_manager.args.real_batch_size} | Prefill {llm_manager.args.sim_prefill_chunk_size}/{llm_manager.args.sim_prefill_cached} | Avg Decode Seq {llm_manager.args.sim_kv_cache}", fontsize=FIG_TITLE_SIZE, y=1.1)
    else:
        axes[0].set_title(f"Pistil Multi-Chiplet LLM Simulation\n{llm_manager.args.sim_num_chiplets}-CUs | {llm_manager.model_name} | BS {llm_manager.args.real_batch_size} | Seq {llm_manager.args.sim_kv_cache}", fontsize=FIG_TITLE_SIZE, y=1.1)
    if SHOW_LEGEND:
        fig.subplots_adjust(left=.04, right=0.9, top=.86, wspace=0)
    else:
        fig.subplots_adjust(left=.04, right=0.99, top=.86, wspace=0)
    
    print("\tSaving Figure")
    plt.savefig(output_path, dpi=300)
    plt.close()


