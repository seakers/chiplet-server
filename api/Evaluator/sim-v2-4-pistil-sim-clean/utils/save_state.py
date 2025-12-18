import torch

class SaveState:
    def __init__(self):
        self.saved_info = {}
        self.enabled = False
        self.enable_compare = False
        self.num_examples = 4 # half of total 

    def enable_save_state(self):
        self.enabled = True

    def save_state(self, layer_id, name, param):
        if self.enabled:
            if layer_id not in self.saved_info:
                self.saved_info[layer_id] = {}
            
            self.saved_info[layer_id][name] = param
    
    def tensor_to_string(self, tensor, precision=6):
        # Format each element in the tensor to the specified precision
        tensor_list = tensor.tolist()
        formatted_list = [f"{x:.{precision}f}" for x in tensor]
    
        # Convert the formatted list to a string
        tensor_str = ', '.join(formatted_list)
        return tensor_str

    def compare_result(self, layer_id, name, param):
        if self.enable_compare:
            baseline_res = self.saved_info[layer_id][name]
            baseline_res = baseline_res.flatten()
            param = param.flatten()

            if baseline_res.size() != param.size():
                print("Warning: Trying to compare results which are different sizes")
                print("\t", name, baseline_res.size(), param.size())
                return 

            diff = torch.sum(torch.abs(baseline_res - param))
            print("layer", layer_id, " - ", name, "diff=%0.8f" % (diff))
            print("\tbase:   ", self.tensor_to_string(baseline_res[0:self.num_examples]), "...", self.tensor_to_string(baseline_res[-self.num_examples:]))
            print("\tpistil: ", self.tensor_to_string(param[0:self.num_examples]), "...", self.tensor_to_string(param[-self.num_examples:]))
            #if diff > 0.001:
            #    print("Diff of this layer > 0")
            #    exit()

    def collect_shards(_, self):
        if _.enable_compare:
            collected_shard = torch.tensor([])
            for chiplet_id in range(self.num_chiplets):
                collected_shard = torch.cat((collected_shard, self.chiplet_garden[chiplet_id].act_shard), dim=-1)
            return collected_shard
    
    def collect_accum_shards(_, self):
        if _.enable_compare:
            collected_shard = torch.tensor([])
            for chiplet_id in range(self.num_chiplets):
                collected_shard = torch.cat((collected_shard, self.chiplet_garden[chiplet_id].act_shard), dim=-1)
            return collected_shard
    
    def collect_q_accum_shards(_, self):
        if _.enable_compare:
            q_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.num_heads))
            ws_q_proj = self.ws_q_proj
            head_dim = self.llm_manager.head_dim
            
            for chiplet_id in range(self.num_chiplets):
                act_shard = self.chiplet_garden[chiplet_id].act_shard
                q_shards = ws_q_proj[chiplet_id]
                
                offset = 0
                for head, (ss, se), (es, ee) in q_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    q_vector[head*head_dim+ss:head*head_dim+se] = act_shard[0, 0, offset:offset + (se-ss)]
                    q_vector[head*head_dim+es:head*head_dim+ee] = act_shard[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return q_vector
    
    def collect_k_accum_shards(_, self):
        if _.enable_compare:
            k_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.kv_heads))
            ws_k_proj = self.ws_k_proj
            head_dim = self.llm_manager.head_dim
            
            
            for chiplet_id in range(self.num_chiplets):
                q_shards = self.ws_q_proj[chiplet_id]
                chiplet_q_offset = 0
                for head, (ss, se), (es, ee) in q_shards[1]:
                    chiplet_q_offset += (se - ss) + (ee - es)

                act_shard = self.chiplet_garden[chiplet_id].act_shard
                k_shards = ws_k_proj[chiplet_id]
                
                offset = chiplet_q_offset
                for head, (ss, se), (es, ee) in k_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    k_vector[head*head_dim+ss:head*head_dim+se] = act_shard[0, 0, offset:offset + (se-ss)]
                    k_vector[head*head_dim+es:head*head_dim+ee] = act_shard[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return k_vector
    
    def collect_v_accum_shards(_, self):
        if _.enable_compare:
            v_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.kv_heads))
            ws_v_proj = self.ws_v_proj
            head_dim = self.llm_manager.head_dim
            
            
            for chiplet_id in range(self.num_chiplets):
                chiplet_q_offset = 0
                for head, (ss, se), (es, ee) in self.ws_q_proj[chiplet_id][1]:
                    chiplet_q_offset += (se - ss) + (ee - es)
                for head, (ss, se), (es, ee) in self.ws_v_proj[chiplet_id][1]:
                    chiplet_q_offset += (se - ss) + (ee - es)

                act_shard = self.chiplet_garden[chiplet_id].act_shard
                v_shards = ws_v_proj[chiplet_id]
                
                offset = chiplet_q_offset
                for head, (ss, se), (es, ee) in v_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    v_vector[head*head_dim+ss:head*head_dim+se] = act_shard[0, 0, offset:offset + (se-ss)]
                    v_vector[head*head_dim+es:head*head_dim+ee] = act_shard[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return v_vector
    
    def collect_q_act_shards(_, self):
        if _.enable_compare:
            q_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.num_heads))
            ws_q_proj = self.ws_q_proj
            head_dim = self.llm_manager.head_dim
            
            for chiplet_id in range(self.num_chiplets):
                act_shard = self.chiplet_garden[chiplet_id].act_shard
                q_shards = ws_q_proj[chiplet_id]
                
                offset = 0
                for head, (ss, se), (es, ee) in q_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    q_vector[head*head_dim+ss:head*head_dim+se] = act_shard[0, 0, offset:offset + (se-ss)]
                    q_vector[head*head_dim+es:head*head_dim+ee] = act_shard[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return q_vector

    def collect_k_act_shards(_, self):
        if _.enable_compare:
            k_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.kv_heads))
            ws_k_proj = self.ws_k_proj
            head_dim = self.llm_manager.head_dim
            
            
            for chiplet_id in range(self.num_chiplets):
                q_shards = self.ws_q_proj[chiplet_id]
                chiplet_q_offset = 0
                for head, (ss, se), (es, ee) in q_shards[1]:
                    chiplet_q_offset += (se - ss) + (ee - es)

                act_shard = self.chiplet_garden[chiplet_id].act_shard
                k_shards = ws_k_proj[chiplet_id]
                
                offset = chiplet_q_offset
                for head, (ss, se), (es, ee) in k_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    k_vector[head*head_dim+ss:head*head_dim+se] = act_shard[0, 0, offset:offset + (se-ss)]
                    k_vector[head*head_dim+es:head*head_dim+ee] = act_shard[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return k_vector
    
    def collect_v_act_shards(_, self):
        if _.enable_compare:
            v_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.kv_heads))
            ws_v_proj = self.ws_v_proj
            head_dim = self.llm_manager.head_dim
            
            
            for chiplet_id in range(self.num_chiplets):
                chiplet_q_offset = 0
                for head, (ss, se), (es, ee) in self.ws_q_proj[chiplet_id][1]:
                    chiplet_q_offset += (se - ss) + (ee - es)
                for head, (ss, se), (es, ee) in self.ws_v_proj[chiplet_id][1]:
                    chiplet_q_offset += (se - ss) + (ee - es)

                act_shard = self.chiplet_garden[chiplet_id].act_shard
                v_shards = ws_v_proj[chiplet_id]
                
                offset = chiplet_q_offset
                for head, (ss, se), (es, ee) in v_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    v_vector[head*head_dim+ss:head*head_dim+se] = act_shard[0, 0, offset:offset + (se-ss)]
                    v_vector[head*head_dim+es:head*head_dim+ee] = act_shard[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return v_vector

    
    def collect_q_rot_shards(_, self):
        if _.enable_compare:
            q_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.num_heads))
            ws_q_proj = self.ws_q_proj
            head_dim = self.llm_manager.head_dim
            
            for chiplet_id in range(self.num_chiplets):
                shard_q = self.chiplet_garden[chiplet_id].shard_q
                q_shards = ws_q_proj[chiplet_id]
                
                offset = 0
                for head, (ss, se), (es, ee) in q_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    q_vector[head*head_dim+ss:head*head_dim+se] = shard_q[0, 0, offset:offset + (se-ss)]
                    q_vector[head*head_dim+es:head*head_dim+ee] = shard_q[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return q_vector

    def collect_k_rot_shards(_, self):
        if _.enable_compare:
            k_vector = torch.zeros((self.llm_manager.head_dim * self.llm_manager.kv_heads))
            ws_k_proj = self.ws_k_proj
            head_dim = self.llm_manager.head_dim
            
            
            for chiplet_id in range(self.num_chiplets):
                shard_k = self.chiplet_garden[chiplet_id].shard_k
                k_shards = ws_k_proj[chiplet_id]
                
                offset = 0
                for head, (ss, se), (es, ee) in k_shards[1]:
                    cols_per_head = (se - ss) + (ee - es)
                    k_vector[head*head_dim+ss:head*head_dim+se] = shard_k[0, 0, offset:offset + (se-ss)]
                    k_vector[head*head_dim+es:head*head_dim+ee] = shard_k[0, 0, offset + (se-ss):offset+cols_per_head]
                    offset += cols_per_head
            return k_vector
    
    def collect_scores(_, self):
        if _.enable_compare:
            cur_seq_len = _.saved_info[0]["scores"].size()[-1]
            all_scores = torch.zeros(_.saved_info[0]["scores"].size()) # batch size, q_heads, cur_seq_len, cached_seq_len
            for chiplet_id in range(self.num_chiplets):
                scores = self.chiplet_garden[chiplet_id].scores
                for kv_head in scores:
                    shard = self.chiplet_garden[chiplet_id].model_shard.all_layers[0].kv_cache.shards
                    head_seq_ids = torch.tensor(shard[1][kv_head])
                    cached_ids = head_seq_ids[torch.where(head_seq_ids < cur_seq_len)[0]]
                    gqa = scores[kv_head].size()[2]
                    for gqa_id in range(gqa): 
                        all_scores[0, kv_head * gqa + gqa_id, :, :][:, cached_ids] += scores[kv_head][0, 0, gqa_id, :]
            return all_scores
    
    def collect_soft_scores(_, self):
        if _.enable_compare:
            cur_seq_len = _.saved_info[0]["scores"].size()[-1]
            all_scores = torch.zeros(_.saved_info[0]["soft_scores"].size()) # batch size, q_heads, cur_seq_len, cached_seq_len
            for chiplet_id in range(self.num_chiplets):
                scores = self.chiplet_garden[chiplet_id].soft_scores
                for kv_head in scores:
                    shard = self.chiplet_garden[chiplet_id].model_shard.all_layers[0].kv_cache.shards
                    head_seq_ids = torch.tensor(shard[1][kv_head])
                    cached_ids = head_seq_ids[torch.where(head_seq_ids < cur_seq_len)[0]]
                    gqa = scores[kv_head].size()[2]
                    for gqa_id in range(gqa): 
                        all_scores[0, kv_head * gqa + gqa_id, :, :][:, cached_ids] += scores[kv_head][0, 0, gqa_id, :]
                        
            return all_scores
    
    def collect_gate_shards(_, self):
        if _.enable_compare:
            collected_shard = torch.tensor([])
            for chiplet_id in range(self.num_chiplets):
                act_shard_half = int(self.chiplet_garden[chiplet_id].act_shard.size()[-1]/2)
                collected_shard = torch.cat((collected_shard, self.chiplet_garden[chiplet_id].act_shard[0, 0, :act_shard_half]), dim=-1)
            return collected_shard
    
    def collect_up_shards(_, self):
        if _.enable_compare:
            collected_shard = torch.tensor([])
            for chiplet_id in range(self.num_chiplets):
                act_shard_half = int(self.chiplet_garden[chiplet_id].act_shard.size()[-1]/2)
                collected_shard = torch.cat((collected_shard, self.chiplet_garden[chiplet_id].act_shard[0, 0, act_shard_half:]), dim=-1)
            return collected_shard
    
    def collect_accum_shards_gate(_, self):
        if _.enable_compare:
            collected_shard = torch.tensor([])
            for chiplet_id in range(self.num_chiplets):
                accum_shard_half = int(self.chiplet_garden[chiplet_id].act_shard.size()[-1]/2)
                collected_shard = torch.cat((collected_shard, self.chiplet_garden[chiplet_id].accum_shard[0, 0, :accum_shard_half]), dim=-1)
            return collected_shard
    
    def collect_accum_shards_up(_, self):
        if _.enable_compare:
            collected_shard = torch.tensor([])
            for chiplet_id in range(self.num_chiplets):
                accum_shard_half = int(self.chiplet_garden[chiplet_id].act_shard.size()[-1]/2)
                collected_shard = torch.cat((collected_shard, self.chiplet_garden[chiplet_id].accum_shard[0, 0, accum_shard_half:]), dim=-1)
            return collected_shard