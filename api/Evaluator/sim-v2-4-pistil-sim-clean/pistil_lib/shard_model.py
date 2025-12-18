import math
import time
import heapq
import torch
import numpy as np


class ShardModel:
    def __init__(self, garden, llm_manager):
        self.garden         = garden
        self.llm_manager    = llm_manager
        self.num_chiplets   = garden.num_chiplets

    def get_distribution(self, items, places):
        quotient = items // places
        remainder = items % places
        distribution = [quotient + 1 if i < remainder else quotient for i in range(places)]
        return distribution

    def get_distribution_multiple_of_2(self, items, places):
        # we need to allocate columns of wQ to chiplets so rotary embeddings can be applied within a chiplet 
        base_count = (items // places) // 2 * 2  # base assigned per chiplet
        total_assigned = base_count * places # total collumns assigned
        remainder = items - total_assigned # missing columns 
        
        distribution = [base_count] * places  # Initialize with base even counts
        
        # Distribute the remaining items evenly in pairs to ensure all counts are even
        for i in range(0, int(remainder), 2):
            distribution[i // 2] += 2
        
        return distribution

    def check_input_cuts(self, input_cuts):
        if input_cuts > 1:
            if (self.num_chiplets % input_cuts) > 0:
                print("Warning: Rings-of-Rings sharding strategy cannot work with this example")
                print("\tNum Cuts must be multiple of num-chiplets (%i chiplets, %i cuts)" % (self.num_chiplets, input_cuts))
                exit()

    def shard_layer_input(self, input_rows, input_cols, input_cuts=1):
        self.check_input_cuts(input_cuts)
        num_sips = int(self.num_chiplets/input_cuts)

        cols_per_cut = int(input_cols/input_cuts)
        work_col_distribution = self.get_distribution(cols_per_cut, num_sips)

        work_shard = {}
        for cut in range(input_cuts):
            work_shard_start = 0
            for sip_id in range(num_sips):
                chiplet_col_work = work_col_distribution[sip_id]
                work_shard[num_sips*cut + sip_id] = ((0, input_rows), (cols_per_cut*cut + work_shard_start, cols_per_cut*cut + work_shard_start+chiplet_col_work))
                work_shard_start += chiplet_col_work
        
        # get expected row ordering based on input cuts
        row_ordering = {}
        for cut in range(input_cuts):
            offset = 0
            for sip_id in range(num_sips):
                chiplet_id = num_sips*cut + sip_id
                row_ordering[chiplet_id] = ((cols_per_cut*cut + offset, cols_per_cut*cut + cols_per_cut), (cols_per_cut*cut, cols_per_cut*cut + offset))
                offset += work_col_distribution[sip_id]
        return work_shard, row_ordering

    def shard_linear(self, weight_rows, weight_cols, input_cuts=1):
        self.check_input_cuts(input_cuts)
        
        num_sips = int(self.num_chiplets/input_cuts)

        work_row_distribution = self.get_distribution(weight_rows, input_cuts)
        work_col_distribution = self.get_distribution(weight_cols, num_sips)   # work columns divdied across pistils in SiPs
        
        work_shard = {}
        work_shard_start = 0
        for cut in range(input_cuts):
            work_shard_start = 0
            for sip_id in range(num_sips):
                chiplet_id = sip_id + cut*num_sips
                chiplet_row_work = work_row_distribution[cut]
                chiplet_col_work = work_col_distribution[sip_id]

                row_work_start = sum(work_row_distribution[:cut])
                work_shard[chiplet_id] = ((row_work_start, row_work_start+chiplet_row_work), (work_shard_start, work_shard_start+chiplet_col_work)) # row_ids (s,e), col_id (s,e)
                work_shard_start += chiplet_col_work
        
        return work_shard
    
    def shard_expert_projection(self, weight_rows, weight_cols, expert_proj_chiplets, input_cuts=1):
        self.check_input_cuts(input_cuts)
        
        num_sips = int(expert_proj_chiplets/input_cuts)

        work_row_distribution = self.get_distribution(weight_rows, input_cuts)
        work_col_distribution = self.get_distribution(weight_cols, num_sips)   # work columns divdied across pistils in SiPs
        
        work_shard = {}
        work_shard_start = 0
        for cut in range(input_cuts):
            work_shard_start = 0
            for sip_id in range(num_sips):
                chiplet_id = sip_id + cut*num_sips
                chiplet_row_work = work_row_distribution[cut]
                chiplet_col_work = work_col_distribution[sip_id]

                row_work_start = sum(work_row_distribution[:cut])
                work_shard[chiplet_id] = ((row_work_start, row_work_start+chiplet_row_work), (work_shard_start, work_shard_start+chiplet_col_work)) # row_ids (s,e), col_id (s,e)
                work_shard_start += chiplet_col_work
        
        return work_shard

    def shard_wqkv(self, weight_rows, head_dim, q_heads, kv_heads, input_cuts=1):
        gqa = self.llm_manager.gqa
        self.check_input_cuts(input_cuts)
        
        wKV_dim         = head_dim * kv_heads
        
        work_row_distribution = self.get_distribution(weight_rows, input_cuts)
        wKV_column_dist = self.get_distribution_multiple_of_2(wKV_dim, int(self.num_chiplets/input_cuts))
        
        def calc_ids_rotary_aware(dist, head_dim):
            breakdown = {}
            rotary_offset = head_dim // 2
            heads_completed = 0
            head_start = 0
            for sip_id in range(len(dist)):
                column_ids = []
                cols_per_chiplet = dist[sip_id]
                
                while cols_per_chiplet > 0:
                    if head_start != 0 and (2*head_start + cols_per_chiplet) >= head_dim:
                        # non-even starting position
                        # finish the heads
                        remaining_cols = 2 * (rotary_offset - head_start)
                        data = (heads_completed, 
                                    (head_start, rotary_offset), 
                                    (rotary_offset+head_start, head_dim))
                        column_ids.append(data)
                        cols_per_chiplet -= remaining_cols
                        head_start = 0
                        heads_completed += 1
                    elif cols_per_chiplet < head_dim:
                        # not enough to complete an entire head
                        # start next head and set the head_start dim 
                        rotary_wQ_cols = int(cols_per_chiplet / 2) # cols remaining
                        data = (heads_completed, 
                                    (head_start, head_start+rotary_wQ_cols), 
                                    (head_start + rotary_offset, head_start + rotary_offset + rotary_wQ_cols))
                        column_ids.append(data)
                        head_start += rotary_wQ_cols
                        cols_per_chiplet -= (2*rotary_wQ_cols)
                    else:
                        column_ids.append((heads_completed, (0, rotary_offset), (rotary_offset, head_dim)))
                        heads_completed += 1
                        cols_per_chiplet -= head_dim
                
                breakdown[sip_id] = column_ids
            
            return breakdown

        work_shard_k_cols = calc_ids_rotary_aware(wKV_column_dist, head_dim)

        # split the rows across SiPs
        work_shard_q = {}
        work_shard_k = {}
        work_shard_v = {}
        num_sips = int(self.num_chiplets / input_cuts)
        for cut in range(input_cuts):
            cur_head = 0
            head_offset = 0
            for sip_id in range(num_sips):
                chiplet_id = sip_id + cut*num_sips
                
                chiplet_row_work = work_row_distribution[cut]
                chiplet_col_work_k = work_shard_k_cols[sip_id]

                row_work_start = sum(work_row_distribution[:cut])

                work_shard_k[chiplet_id] = ((row_work_start, row_work_start+chiplet_row_work), chiplet_col_work_k)
                work_shard_v[chiplet_id] = ((row_work_start, row_work_start+chiplet_row_work), [])
                for (head, col_s, col_e) in chiplet_col_work_k:
                    if head != cur_head:
                        cur_head = head
                        head_offset = 0
                    cols_per_head = col_e[1] - col_e[0] + col_s[1] - col_s[0]
                    mid_cols_per_head = int(cols_per_head/2)
                    work_shard_v[chiplet_id][1].append((head, (head_offset, head_offset+mid_cols_per_head), (head_offset+mid_cols_per_head, head_offset+cols_per_head)))
                    head_offset += cols_per_head
                
                work_shard_q[chiplet_id] = ((row_work_start, row_work_start+chiplet_row_work), [])
                for (head, col_s, col_e) in chiplet_col_work_k:
                    for gqa_id in range(gqa):
                        work_shard_q[chiplet_id][1].append((head*gqa+gqa_id, col_s, col_e))
        #print("Warning - this could pop up again where the gpa id sharding is flipped with kv_heads - super hard debug... ") 
        #print("chiplet 0", work_shard_q[0])
        #print("chiplet 1", work_shard_q[1])
        #print("chiplet 2", work_shard_q[2])
        #print("chiplet 3", work_shard_q[3])
        #print("chiplet 4", work_shard_q[4])
        #exit()
        return work_shard_q, work_shard_k, work_shard_v
    
    def shard_attention(self, head_dim, kv_heads, max_seq_len, ws_k_proj, input_cuts=1):
        self.check_input_cuts(input_cuts)
        
        work_row_distribution_cache = self.get_distribution(head_dim, input_cuts)
        #cache_column_dist = self.get_distribution(max_seq_len*kv_heads, int(self.num_chiplets/input_cuts)) # most equal sharding
        scalar = max_seq_len * kv_heads / (head_dim * kv_heads)
        cache_column_dist = []
        for chiplet_id in ws_k_proj:
            cols_per_chiplet = 0
            for head, (ss, se), (es, ee) in ws_k_proj[chiplet_id][1]:
                cols_per_chiplet += (se - ss) + (ee - es)
            cache_column_dist.append(int(cols_per_chiplet*scalar))
        
        def calc_breakdown_seq(dist, max_seq_len):
            breakdown = {}
            heads_completed = 0
            head_start = 0
            for sip_id in range(len(dist)):
                column_ids = []
                cols_per_chiplet = dist[sip_id]
                
                while cols_per_chiplet > 0:
                    if head_start != 0 and (head_start + cols_per_chiplet) >= max_seq_len:
                        remaining_cols = (max_seq_len - head_start)
                        data = (heads_completed, max_seq_len-head_start)
                        column_ids.append(data)
                        cols_per_chiplet -= remaining_cols
                        head_start = 0
                        heads_completed += 1
                    elif cols_per_chiplet < max_seq_len:
                        data = (heads_completed, cols_per_chiplet)
                        column_ids.append(data)
                        head_start += cols_per_chiplet
                        cols_per_chiplet = 0
                    else:
                        column_ids.append((heads_completed, max_seq_len))
                        heads_completed += 1
                        cols_per_chiplet -= max_seq_len
                
                breakdown[sip_id] = column_ids
            return breakdown

        work_shard_cache_cols = calc_breakdown_seq(cache_column_dist, max_seq_len)

        def assign_ids(total_items, breakdown):
            num_groups = len(breakdown)
            groups = [[] for _ in range(num_groups)]
            assigned_count = [0] * num_groups
            
            # Initialize the priority queue with tuples (ratio, group_index)
            priority_queue = [(0, g_id) for g_id in range(num_groups)]
            heapq.heapify(priority_queue)
            
            # Assign IDs to each group in a round-robin fashion
            for item_id in range(total_items):
                min_ratio, min_group = heapq.heappop(priority_queue)
                assigned_count[min_group] += 1
                groups[min_group].append(item_id)
                
                # Calculate the new ratio and push back into the priority queue
                new_ratio = assigned_count[min_group] / breakdown[min_group]
                heapq.heappush(priority_queue, (new_ratio, min_group))
            
            return groups


        chiplet_seq_ids = {}
        total_items = max_seq_len
        for sip_id in range(len(work_shard_cache_cols)):
            cur_seq_breakdown = work_shard_cache_cols[sip_id]
            
            for head_id, num_seq_ids in cur_seq_breakdown:
                self_id = None
                head_seq_breakdown = []
                
                for sip_S in range(len(work_shard_cache_cols)):
                    cur_seq_breakdown_S = work_shard_cache_cols[sip_S]
                    if sip_id == sip_S:
                        self_id = len(head_seq_breakdown)

                    for head_id_S, num_seq_ids_S in cur_seq_breakdown_S:
                        if head_id_S == head_id:
                            head_seq_breakdown.append(num_seq_ids_S)
        
                groups = assign_ids(sum(head_seq_breakdown), head_seq_breakdown)
                if sip_id not in chiplet_seq_ids:
                    chiplet_seq_ids[sip_id] = {}
                
                if head_id not in chiplet_seq_ids[sip_id]:
                    chiplet_seq_ids[sip_id][head_id] = groups[self_id]

        work_shard_cache = {}
        for chiplet_id in range(self.num_chiplets):
            sip_id = chiplet_id % input_cuts
            group_id = int(chiplet_id/input_cuts)
            
            chiplet_row_work = work_row_distribution_cache[sip_id]
            chiplet_work_cache = chiplet_seq_ids[group_id]

            row_work_start = sum(work_row_distribution_cache[:sip_id])

            work_shard_cache[chiplet_id] = ((row_work_start, row_work_start+chiplet_row_work), chiplet_work_cache)


        # this implementation does not work with cutting the KV$ using an SiP approach... 
        work_shard_cache_inv = {}
        for chiplet_id in range(self.num_chiplets):
            for head in work_shard_cache[chiplet_id][1]:
                if head not in work_shard_cache_inv.keys():
                    work_shard_cache_inv[head] = {}
                for seq_id in work_shard_cache[chiplet_id][1][head]:
                    work_shard_cache_inv[head][seq_id] = chiplet_id

        return work_shard_cache, work_shard_cache_inv
    
    def find_dest_by_head(self, ws_q_proj, ws_k_proj, output_shard_cols):
        gqa = self.llm_manager.gqa
        head_dim = self.llm_manager.head_dim
        num_heads = self.llm_manager.num_heads
        model_dim = self.llm_manager.model_dim

        head_dest = {} 
        head_receive_order = {}
        qkv_nic_config_algorithm = {}
        row_order_cache = {}
        o_nic_config_algorithm = {} 
        row_order_wo = {}
        # head_dest = {head: [0, 1, 2, 3, 4]} # this applies for Q K and V as to what chiplet each chiplet needs to send QKV data too 
        # head_receive_order = {chiplet_id: {head: [0, 1, 2, 3, 4]} } # this is the order which a chiplet receives data from what chiplet_id
        # qkv_nic_config_algorithm: {chiplet_id: {forward: T/F, send_w:head_id/None, send_e:head_id/None}})  # send_left/send_right is for the inital send, forward true if send both east and west. Can be modified to use send bytes instead so more clear to chiplet
        # row_order_cache = {chiplet_id: {head: []} } # reorder rows of K$ / V$ to align with receive order
        # o_nic_config_algorithm: {chiplet_id: {send_w_vals: , send_e_vals: , start_reduction: T/F}} # is this a node which starts the reduction. send_w_vals / send_e_vals is the same 


        # ws_input_o_proj is based on gqa*kv$ sharding {chiplet_id: ((rs, re), (cs, ce)}
        ws_input_o_proj = {}
        
        head_dest = {}
        for chiplet_id in ws_k_proj.keys():
            for head, _, _ in ws_k_proj[chiplet_id][1]:
                if head not in head_dest.keys():
                    head_dest[head] = []
                head_dest[head].append(chiplet_id)
            
        # for figuring out the order for chiplets receiving data from chiplet_ids from a 2D torus 
        def reorder_heads(cur_chiplet, all_heads):
            pivot_index = all_heads.index(cur_chiplet)
            left_part = list(reversed(all_heads[:pivot_index]))  # elements before pivot_index
            right_part = all_heads[pivot_index + 1:]  # elements after pivot_index
            
            # Interleave: left_part first, then right_part - west then east
            interleaved_list = []
            for i in range(min(len(left_part), len(right_part))):
                interleaved_list.append(left_part[i])
                interleaved_list.append(right_part[i])
            
            # Add remaining elements from the longer part, if any
            if len(left_part) > len(right_part):
                interleaved_list.extend(left_part[len(right_part):])
            elif len(right_part) > len(left_part):
                interleaved_list.extend(right_part[len(left_part):])
            
            # Add the pivot element
            interleaved_list.insert(0, all_heads[pivot_index])
            #print(cur_chiplet, all_heads, interleaved_list)
            return interleaved_list

        #for head in head_dest:
        #    print(head, head_dest[head])
        
        for chiplet_id in ws_k_proj.keys():
            head_receive_order[chiplet_id] = {}
            for head, _, _ in ws_k_proj[chiplet_id][1]:
                head_receive_order[chiplet_id][head] = reorder_heads(chiplet_id, head_dest[head])

        #for chiplet_id in head_receive_order:
        #    for head in head_receive_order[chiplet_id].keys():
        #        print(chiplet_id, head, head_receive_order[chiplet_id][head])
        
        for chiplet_id in head_receive_order:
            qkv_nic_config_algorithm[chiplet_id] = {"send_w_vals": None, "send_e_vals": None, "start_reduction": True}
            for head in head_receive_order[chiplet_id].keys():
                cols_per_chiplet = 0
                for head_kv, (ss, se), (es, ee) in ws_k_proj[chiplet_id][1]:
                    if head == head_kv:
                        cols_per_chiplet += (se - ss) + (ee - es)
                e_chiplet_id = self.garden.chiplet_garden[chiplet_id].e_chiplet_id
                w_chiplet_id = self.garden.chiplet_garden[chiplet_id].w_chiplet_id
                send_w = True if head in head_receive_order[w_chiplet_id].keys() else False
                send_e = True if head in head_receive_order[e_chiplet_id].keys() else False
                forward = send_w and send_e

                if forward:
                    qkv_nic_config_algorithm[chiplet_id]["start_reduction"] = not forward
                if send_w:
                    qkv_nic_config_algorithm[chiplet_id]["send_w_vals"] = cols_per_chiplet
                if send_e:
                    qkv_nic_config_algorithm[chiplet_id]["send_e_vals"] = cols_per_chiplet
                
        #for chiplet_id in qkv_nic_config_algorithm:
        #    print(chiplet_id, qkv_nic_config_algorithm[chiplet_id])

        for chiplet_id in head_receive_order:
            row_order_cache[chiplet_id] = {}
            for head in head_receive_order[chiplet_id].keys():
                row_order_cache[chiplet_id][head] = torch.tensor([]).to(torch.long)
                for chiplet_id_order in head_receive_order[chiplet_id][head]:
                    for head_tmp, (ss, se), (es, ee) in ws_k_proj[chiplet_id_order][1]:
                        if head_tmp == head:
                            row_order_cache[chiplet_id][head] = torch.cat((row_order_cache[chiplet_id][head], torch.arange(ss, se).to(torch.long), torch.arange(es, ee).to(torch.long)))
        
        #for chiplet_id in row_order_cache:
        #    for head in row_order_cache[chiplet_id]:
        #        print(chiplet_id, head, row_order_cache[chiplet_id][head])
        

        offset = 0
        for chiplet_id in ws_k_proj:
            cols_per_chiplet = 0
            for head, (ss, se), (es, ee) in ws_k_proj[chiplet_id][1]:
                cols_per_chiplet += (ee-es + se-ss)*gqa
            ws_input_o_proj[chiplet_id] = ((0, 1), (offset, offset+cols_per_chiplet))
            offset += cols_per_chiplet
        
        # v is not rotated! use that to our advantage because v$ head_dim will be in order of reception so return in order for reduction
        offset = 0
        loop_vals = gqa*head_dim
        for chiplet_id in qkv_nic_config_algorithm:
            heads_on_chip = len(ws_k_proj[chiplet_id][1])
            total_vals_on_chip = heads_on_chip*gqa*head_dim
            chiplet_vals_keep = ws_input_o_proj[chiplet_id][1][1] - ws_input_o_proj[chiplet_id][1][0]

            send_w_vals = offset
            send_e_vals = total_vals_on_chip - offset - chiplet_vals_keep
            o_nic_config_algorithm[chiplet_id] = {"send_w_vals": send_w_vals, "send_e_vals": send_e_vals, "start_reduction": qkv_nic_config_algorithm[chiplet_id]["start_reduction"]}
            
            offset += chiplet_vals_keep
            while offset >= loop_vals:
                offset -= loop_vals
            
        for chiplet_id in ws_input_o_proj:
            row_order_wo[chiplet_id] = ((ws_input_o_proj[chiplet_id][1][0], num_heads*head_dim), (0, ws_input_o_proj[chiplet_id][1][0]))
            
        return head_dest, head_receive_order, qkv_nic_config_algorithm, row_order_cache, o_nic_config_algorithm, ws_input_o_proj, row_order_wo

