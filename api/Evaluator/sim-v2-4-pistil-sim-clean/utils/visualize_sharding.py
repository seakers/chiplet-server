import os
import numpy as np
import matplotlib.pyplot as plt

class VisualizeSharding:
    def __init__(self, results_dir):
        self.title_size = 12
        self.axtitle_size = 10
        self.xlabel_size = 10
        self.ylabel_size = 10
        self.xtick_size = 10
        self.ytick_size = 10
        self.legend_size = 10
        
        self.marker_size = 6
        self.marker_edge_width = .8

        self.large_marker_size = 8
        self.small_marker_size = 6

        self.dpi = 100
        
        self.results_dir = results_dir
        
        #self.my_color_map = ["#A51C30", "#3872B2", "#EC8F9C", "#FF6600", "#ECDD7B", "#808080"]
        self.my_color_map = ["#ACC3B1", "#B7D1E2", "#F1A151", "#F1D365", "#DAD7D0", "#668C87"]
        self.my_color_map_extended = ["#AED994", "#71CAB1", "#63CDC0", "#02A6AF", "#94C9DB", "#78ADD1", "#859FB8", "#9FADB8", "#E2EDF0"]
        self.my_markers = ["d", "*", "v", "o",]
    
    def plot_sharding(self, garden, llm_manager, chiplet_id=0):
        batch_size      = 1
        model_dim       = llm_manager.model_dim
        head_dim        = llm_manager.head_dim
        num_heads       = llm_manager.num_heads
        kv_heads        = llm_manager.kv_heads
        gqa             = llm_manager.gqa
        ff_dim          = llm_manager.ff_dim
        max_seq_len     = llm_manager.max_seq_len

        ws_q_proj       = garden.ws_q_proj[chiplet_id]        
        ws_k_proj       = garden.ws_k_proj[chiplet_id]
        ws_v_proj       = garden.ws_v_proj[chiplet_id]
        ws_kv_cache     = garden.ws_kv_cache[chiplet_id]
        ws_o_proj       = garden.ws_o_proj[chiplet_id]
        ws_input_model_dim = garden.ws_input_model_dim[chiplet_id]
        ws_input_ff_dim = garden.ws_input_ff_dim[chiplet_id]
        ws_gate_proj    = garden.ws_gate_proj[chiplet_id]
        ws_up_proj      = garden.ws_up_proj[chiplet_id]
        ws_down_proj    = garden.ws_down_proj[chiplet_id]

        def draw_rect(ax, x0, y0, x_width, y_height, color="black", linewidth=.5):
            ax.plot([x0, x0+x_width],           [y0, y0], color=color, linewidth=linewidth)                   # bottom -- 
            ax.plot([x0, x0+x_width],           [y0+y_height, y0+y_height], color=color, linewidth=linewidth) # top --
            ax.plot([x0, x0],                   [y0, y0+y_height], color=color, linewidth=linewidth)                  # left |
            ax.plot([x0+x_width, x0+x_width],   [y0, y0+y_height], color=color, linewidth=linewidth)          # right |
        
        # draw lines for matrices and get the staring index of each 
        fig, ax = plt.subplots(1, 1, figsize=(50, 10))
        
        SPACING = model_dim * .05
        BS_SCALING = 100

        input_x_start   = 0
        input_y_start   = 0
        input_x_width   = model_dim 
        input_y_height  = batch_size * BS_SCALING
        draw_rect(ax, input_x_start, input_y_start, input_x_width, input_y_height)

        qkv_x_start = input_x_start + input_x_width + SPACING
        qkv_y_start = batch_size * BS_SCALING  + BS_SCALING 
        qkv_x_width = head_dim * (num_heads + kv_heads*2)
        qkv_y_height = model_dim
        draw_rect(ax, qkv_x_start, qkv_y_start, qkv_x_width, qkv_y_height)
        
        qkv_res_x_start = input_x_start + input_x_width + SPACING
        qkv_res_y_start = 0
        qkv_res_x_width = head_dim * (num_heads + kv_heads*2)
        qkv_res_y_height = batch_size + BS_SCALING
        draw_rect(ax, qkv_res_x_start, qkv_res_y_start, qkv_res_x_width, qkv_res_y_height)
        
        for head, (ss, se), (es, ee) in ws_q_proj[1]:
            q_shard_x_start = qkv_res_x_start + head * head_dim + ss
            q_shard_x_width = se - ss
            q_shard_y_start = ws_q_proj[0][0] + batch_size * BS_SCALING + BS_SCALING
            q_shard_y_height = ws_q_proj[0][1] - ws_q_proj[0][0]
            draw_rect(ax, q_shard_x_start, q_shard_y_start, q_shard_x_width, q_shard_y_height, color="#DFC0F3")
            
            q_shard_x_start = qkv_res_x_start + head * head_dim + es
            q_shard_x_width = se - ss
            q_shard_y_start = ws_q_proj[0][0] + batch_size * BS_SCALING + BS_SCALING
            q_shard_y_height = ws_q_proj[0][1] - ws_q_proj[0][0]
            draw_rect(ax, q_shard_x_start, q_shard_y_start, q_shard_x_width, q_shard_y_height, color="#DFC0F3")
        
        for head, (ss, se), (es, ee) in ws_k_proj[1]:
            q_shard_x_start = qkv_res_x_start + num_heads * head_dim + head * head_dim + ss
            q_shard_x_width = se - ss
            q_shard_y_start = ws_q_proj[0][0] + batch_size * BS_SCALING + BS_SCALING
            q_shard_y_height = ws_q_proj[0][1] - ws_q_proj[0][0]
            draw_rect(ax, q_shard_x_start, q_shard_y_start, q_shard_x_width, q_shard_y_height, color="#FEE3A1")
            
            q_shard_x_start = qkv_res_x_start + num_heads * head_dim + head * head_dim + es
            q_shard_x_width = se - ss
            q_shard_y_start = ws_q_proj[0][0] + batch_size * BS_SCALING + BS_SCALING
            q_shard_y_height = ws_q_proj[0][1] - ws_q_proj[0][0]
            draw_rect(ax, q_shard_x_start, q_shard_y_start, q_shard_x_width, q_shard_y_height, color="#FEE3A1")
        
        for head, (ss, se), (es, ee) in ws_v_proj[1]:
            q_shard_x_start = qkv_res_x_start + (num_heads+kv_heads) * head_dim + head * head_dim + ss
            q_shard_x_width = se - ss
            q_shard_y_start = ws_q_proj[0][0] + batch_size * BS_SCALING + BS_SCALING
            q_shard_y_height = ws_q_proj[0][1] - ws_q_proj[0][0]
            draw_rect(ax, q_shard_x_start, q_shard_y_start, q_shard_x_width, q_shard_y_height, color="#85B7E8")
            
            q_shard_x_start = qkv_res_x_start + (num_heads+kv_heads) * head_dim + head * head_dim + es
            q_shard_x_width = se - ss
            q_shard_y_start = ws_q_proj[0][0] + batch_size * BS_SCALING + BS_SCALING
            q_shard_y_height = ws_q_proj[0][1] - ws_q_proj[0][0]
            draw_rect(ax, q_shard_x_start, q_shard_y_start, q_shard_x_width, q_shard_y_height, color="#85B7E8")


        q_x_start = qkv_res_x_start + qkv_res_x_width + SPACING
        for head_id in range(kv_heads):
            q_x_width = head_dim
            q_y_start = 0
            q_y_height = gqa * batch_size * BS_SCALING
            draw_rect(ax, q_x_start, q_y_start, q_x_width, q_y_height)

            k_cache_x_start = q_x_start + q_x_width + SPACING
            k_cache_y_start = q_y_height + BS_SCALING
            k_cache_width = max_seq_len
            k_cache_y_height = head_dim
            draw_rect(ax, k_cache_x_start, k_cache_y_start, k_cache_width, k_cache_y_height)
            
            qk_res_cache_x_start = q_x_start + q_x_width + SPACING
            qk_res_cache_y_start = 0
            qk_res_cache_width = max_seq_len
            qk_res_cache_y_height = q_y_height 
            draw_rect(ax, qk_res_cache_x_start, qk_res_cache_y_start, qk_res_cache_width, qk_res_cache_y_height)
            
            v_cache_x_start = q_x_start + q_x_width + SPACING + k_cache_width + SPACING
            v_cache_y_start = q_y_height + BS_SCALING
            v_cache_width = head_dim
            v_cache_y_height = max_seq_len
            draw_rect(ax, v_cache_x_start, v_cache_y_start, v_cache_width, v_cache_y_height)

            qkv_res_cache_x_start = q_x_start + q_x_width + SPACING + k_cache_width + SPACING
            qkv_res_cache_y_start = 0
            qkv_res_cache_width = head_dim
            qkv_res_cache_y_height = q_y_height 
            draw_rect(ax, qkv_res_cache_x_start, qkv_res_cache_y_start, qkv_res_cache_width, qkv_res_cache_y_height)

            if head_id in ws_kv_cache[1].keys():
                k_cache_s_x_start = k_cache_x_start
                k_cache_s_y_start = q_y_height + BS_SCALING + ws_kv_cache[0][0]
                k_cache_s_width = len(ws_kv_cache[1][head_id])
                k_cache_s_y_height = ws_kv_cache[0][1] - ws_kv_cache[0][0]
                draw_rect(ax, k_cache_s_x_start, k_cache_s_y_start, k_cache_s_width, k_cache_s_y_height, color="#FEE3A1")
            
            if head_id in ws_kv_cache[1].keys():
                v_cache_s_x_start = v_cache_x_start
                v_cache_s_y_start = q_y_height + BS_SCALING + ws_kv_cache[0][0]
                v_cache_s_width = ws_kv_cache[0][1] - ws_kv_cache[0][0]
                v_cache_s_y_height = len(ws_kv_cache[1][head_id])
                draw_rect(ax, v_cache_s_x_start, v_cache_s_y_start, v_cache_s_width, v_cache_s_y_height, color="#85B7E8")
            
            q_x_start += q_x_width + SPACING + k_cache_width + SPACING + v_cache_width + SPACING

        input_wo_x_start = q_x_start
        input_wo_y_start = 0
        input_wo_width = num_heads * head_dim
        input_wo_height = batch_size * BS_SCALING
        draw_rect(ax, input_wo_x_start, input_wo_y_start, input_wo_width, input_wo_height)

        wo_x_start = input_wo_x_start + input_wo_width + SPACING
        wo_y_start = batch_size * BS_SCALING + BS_SCALING
        wo_width = model_dim
        wo_height = num_heads * head_dim
        draw_rect(ax, wo_x_start, wo_y_start, wo_width, wo_height)

        wo_res_x_start = wo_x_start
        wo_res_y_start = 0
        wo_res_width = model_dim
        wo_res_height = batch_size * BS_SCALING
        draw_rect(ax, wo_res_x_start, wo_res_y_start, wo_res_width, wo_res_height)

        wo_s_x_start = wo_x_start + ws_o_proj[1][0]
        wo_s_y_start = batch_size * BS_SCALING + BS_SCALING + ws_o_proj[0][0]
        wo_s_width = ws_o_proj[1][1] - ws_o_proj[1][0]
        wo_s_height = ws_o_proj[0][1] - ws_o_proj[0][0]
        draw_rect(ax, wo_s_x_start, wo_s_y_start, wo_s_width, wo_s_height, color="#91B8D3")


        wgate_up_x_start = wo_res_x_start + wo_res_width + SPACING
        wgate_up_y_start = batch_size * BS_SCALING + BS_SCALING
        wgate_up_width = 2*ff_dim
        wgate_up_height = model_dim
        draw_rect(ax, wgate_up_x_start, wgate_up_y_start, wgate_up_width, wgate_up_height)
        
        wgate_up_res_x_start = wgate_up_x_start
        wgate_up_res_y_start = 0
        wgate_up_res_width = 2*ff_dim
        wgate_up_res_height = batch_size * BS_SCALING
        draw_rect(ax, wgate_up_res_x_start, wgate_up_res_y_start, wgate_up_res_width, wgate_up_res_height)
        

        wgate_up_s_x_start = wgate_up_x_start +  ws_gate_proj[1][0]
        wgate_up_s_y_start = batch_size * BS_SCALING + BS_SCALING + ws_gate_proj[0][0]
        wgate_up_s_width = ws_gate_proj[1][1] -  ws_gate_proj[1][0]
        wgate_up_s_height = ws_gate_proj[0][1] - ws_gate_proj[0][0]
        draw_rect(ax, wgate_up_s_x_start, wgate_up_s_y_start, wgate_up_s_width, wgate_up_s_height, color="#7A9E82")
        
        wgate_up_s_x_start = wgate_up_x_start + ff_dim + ws_up_proj[1][0]
        wgate_up_s_y_start = batch_size * BS_SCALING + BS_SCALING + ws_gate_proj[0][0]
        wgate_up_s_width = ws_up_proj[1][1] -  ws_up_proj[1][0]
        wgate_up_s_height = ws_up_proj[0][1] - ws_up_proj[0][0]
        draw_rect(ax, wgate_up_s_x_start, wgate_up_s_y_start, wgate_up_s_width, wgate_up_s_height, color="#7A9E82")
        
        
        wdown_x_start = wgate_up_x_start + wgate_up_width + SPACING
        wdown_y_start = batch_size * BS_SCALING + BS_SCALING
        wdown_width = model_dim
        wdown_height = ff_dim
        draw_rect(ax, wdown_x_start, wdown_y_start, wdown_width, wdown_height)
        
        wdown_res_x_start = wdown_x_start
        wdown_res_y_start = 0
        wdown_res_width = model_dim
        wdown_res_height = batch_size * BS_SCALING
        draw_rect(ax, wdown_res_x_start, wdown_res_y_start, wdown_res_width, wdown_res_height)
        
        wdown_s_x_start = wdown_x_start +  ws_down_proj[1][0]
        wdown_s_y_start = batch_size * BS_SCALING + BS_SCALING + ws_down_proj[0][0]
        wdown_s_width = ws_down_proj[1][1] -  ws_down_proj[1][0]
        wdown_s_height = ws_down_proj[0][1] - ws_down_proj[0][0]
        draw_rect(ax, wdown_s_x_start, wdown_s_y_start, wdown_s_width, wdown_s_height, color="#7A9E82")

        plt.tight_layout(pad=.1) 
        plt.savefig(os.path.join(self.results_dir, f"work_shard_{chiplet_id}.pdf"), dpi=self.dpi)
        plt.clf()
        plt.close()

        exit()