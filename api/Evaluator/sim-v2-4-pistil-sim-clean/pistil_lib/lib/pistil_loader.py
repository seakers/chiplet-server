import math
import torch

class PistilLoader:
    def __init__(self, pistil):
        self.pistil = pistil

    def load_embedding_shard(self, shard):
        emb_table_weights = self.pistil.base_model.emb_table.weights
        emb_table_shard = emb_table_weights[shard[0][0]:shard[0][1], shard[1][0]:shard[1][1]]
        self.pistil.model_shard.emb_table.load_from_pretrained(emb_table_shard)
        self.pistil.model_shard.emb_table.shards = shard
        
    def load_model_norm_shard(self, shard):
        base_weights = self.pistil.base_model.model_norm.weights
        weight_shard = base_weights[shard[1][0]:shard[1][1]]
        self.pistil.model_shard.model_norm.load_from_pretrained(weight_shard)
        self.pistil.model_shard.model_norm.shards = shard

    def load_lm_head_shard(self, shard, row_ordering=None):
        base_weights = self.pistil.base_model.lm_head.weights
        # Safety check for row_ordering structure
        if row_ordering is None or len(row_ordering) < 2 or not all(len(r) >= 2 for r in row_ordering[:2]):
            # Use identity ordering if row_ordering is invalid
            num_rows = shard[1][1] - shard[1][0]
            row_ids = torch.arange(num_rows, dtype=torch.long)
        else:
            row_ids = torch.cat((torch.arange(row_ordering[0][0], row_ordering[0][1]), torch.arange(row_ordering[1][0], row_ordering[1][1])), dim=-1)
        weight_shard = base_weights[shard[1][0]:shard[1][1], :][:, row_ids]
        self.pistil.model_shard.lm_head.load_from_pretrained(weight_shard)
        self.pistil.model_shard.lm_head.shards = shard

    def load_layer_norm_shard(self, layer_id, shard, layer):
        base_weights = eval(f"self.pistil.base_model.all_layers[layer_id].{layer}.weights")
        weight_shard = base_weights[shard[1][0]:shard[1][1]]
        eval(f"self.pistil.model_shard.all_layers[layer_id].{layer}.load_from_pretrained(weight_shard)")
        exec(f"self.pistil.model_shard.all_layers[layer_id].{layer}.shards = shard")

    def load_layer_wQKV_shard(self, layer_id, shard, layer, row_ordering=None):
        head_dim = self.pistil.llm_manager.head_dim
        base_weights = eval(f"self.pistil.base_model.all_layers[layer_id].{layer}.weights")
        col_ids = torch.tensor([]).long()
        for head_id, (ss, se), (es, ee) in shard[1]:
            col_ids = torch.cat((col_ids, torch.arange(ss+head_id*head_dim, se+head_id*head_dim), torch.arange(es+head_id*head_dim, ee+head_id*head_dim))).long()
        # Safety check for row_ordering structure
        if row_ordering is None or len(row_ordering) < 2 or not all(len(r) >= 2 for r in row_ordering[:2]):
            # Use identity ordering if row_ordering is invalid
            num_rows = base_weights.shape[1]
            row_ids = torch.arange(num_rows, dtype=torch.long)
        else:
            row_ids = torch.cat((torch.arange(row_ordering[0][0], row_ordering[0][1]), torch.arange(row_ordering[1][0], row_ordering[1][1])), dim=-1)
        weight_shard = base_weights[col_ids, :][:, row_ids]

        eval(f"self.pistil.model_shard.all_layers[layer_id].{layer}.load_from_pretrained(weight_shard)")
        exec(f"self.pistil.model_shard.all_layers[layer_id].{layer}.shards = shard")
    
        if layer == "wV":
            weight_shard = torch.cat((self.pistil.model_shard.all_layers[layer_id].wQ.weights.T, self.pistil.model_shard.all_layers[layer_id].wK.weights.T, self.pistil.model_shard.all_layers[layer_id].wV.weights.T), dim=-1)
            # weight_shard = weight_shard[row_order]
            self.pistil.model_shard.all_layers[layer_id].wQKV.load_from_pretrained(weight_shard.T)
    
    def load_layer_gate_up_shard(self, layer_id, shard, layer, row_ordering=None):
        head_dim = self.pistil.llm_manager.head_dim
        base_weights = eval(f"self.pistil.base_model.all_layers[layer_id].{layer}.weights")    
        # Safety check for row_ordering structure
        if row_ordering is None or len(row_ordering) < 2 or not all(len(r) >= 2 for r in row_ordering[:2]):
            # Use identity ordering if row_ordering is invalid
            num_rows = base_weights.shape[1]
            row_ids = torch.arange(num_rows, dtype=torch.long)
        else:
            row_ids = torch.cat((torch.arange(row_ordering[0][0], row_ordering[0][1]), torch.arange(row_ordering[1][0], row_ordering[1][1])), dim=-1)
        weight_shard = base_weights[shard[1][0]:shard[1][1], :][:, row_ids]
        eval(f"self.pistil.model_shard.all_layers[layer_id].{layer}.load_from_pretrained(weight_shard)")
        exec(f"self.pistil.model_shard.all_layers[layer_id].{layer}.shards = shard")
    
        if layer == "up_proj":
            weight_shard = torch.cat((self.pistil.model_shard.all_layers[layer_id].gate_proj.weights.T, self.pistil.model_shard.all_layers[layer_id].up_proj.weights.T), dim=-1)
            self.pistil.model_shard.all_layers[layer_id].gate_up_proj.load_from_pretrained(weight_shard.T)
            new_shard = ((shard[0][0], shard[0][1]), (shard[1][0]*2, shard[1][1]*2)) 
            self.pistil.model_shard.all_layers[layer_id].gate_up_proj.shards = new_shard
        if layer == "up_proj_s":
            weight_shard = torch.cat((self.pistil.model_shard.all_layers[layer_id].gate_proj_s.weights.T, self.pistil.model_shard.all_layers[layer_id].up_proj_s.weights.T), dim=-1)
            self.pistil.model_shard.all_layers[layer_id].gate_up_proj_s.load_from_pretrained(weight_shard.T)
            new_shard = ((shard[0][0], shard[0][1]), (shard[1][0]*2, shard[1][1]*2)) 
            self.pistil.model_shard.all_layers[layer_id].gate_up_proj_s.shards = new_shard
        if layer == "up_proj_e":
            weight_shard = torch.cat((self.pistil.model_shard.all_layers[layer_id].gate_proj_e.weights.T, self.pistil.model_shard.all_layers[layer_id].up_proj_e.weights.T), dim=-1)
            self.pistil.model_shard.all_layers[layer_id].gate_up_proj_e.load_from_pretrained(weight_shard.T)
            new_shard = ((shard[0][0], shard[0][1]), (shard[1][0]*2, shard[1][1]*2)) 
            self.pistil.model_shard.all_layers[layer_id].gate_up_proj_e.shards = new_shard

    def load_layer_linear_shard(self, layer_id, shard, layer, row_ordering=None):
        base_weights = eval(f"self.pistil.base_model.all_layers[layer_id].{layer}.weights")
        # Safety check for row_ordering structure
        if row_ordering is None or len(row_ordering) < 2 or not all(len(r) >= 2 for r in row_ordering[:2]):
            # Use identity ordering if row_ordering is invalid
            num_rows = base_weights.shape[1]
            row_ids = torch.arange(num_rows, dtype=torch.long)
        else:
            row_ids = torch.cat((torch.arange(row_ordering[0][0], row_ordering[0][1]), torch.arange(row_ordering[1][0], row_ordering[1][1])), dim=-1)
        weight_shard = base_weights[shard[1][0]:shard[1][1], :][:, row_ids]
        eval(f"self.pistil.model_shard.all_layers[layer_id].{layer}.load_from_pretrained(weight_shard)")
        exec(f"self.pistil.model_shard.all_layers[layer_id].{layer}.shards = shard")
    
    def load_layer_cache_shard(self, layer_id, shard, shard_inv, row_ordering=None):
        self.pistil.model_shard.all_layers[layer_id].kv_cache.shards = shard
        self.pistil.model_shard.all_layers[layer_id].kv_cache.reset()

        cache_k = self.pistil.base_model.all_layers[layer_id].kv_cache.cache_k
        cache_v = self.pistil.base_model.all_layers[layer_id].kv_cache.cache_v
        prefill_seq_len = self.pistil.llm_manager.prefill_input_ids.size()[1]

        cache_k_shard = {}
        cache_v_shard = {}
        for head_id in shard[1].keys():
            head_seq_ids = torch.tensor(shard[1][head_id])
            subset_head_seq_ids = head_seq_ids[torch.where(head_seq_ids < prefill_seq_len)[0]]
            # Check if row_ordering exists and has this head_id, otherwise use identity ordering
            if row_ordering is not None and head_id in row_ordering:
                row_order = row_ordering[head_id]
            else:
                # Use identity ordering (no reordering) if row_ordering doesn't have this head_id
                head_dim = cache_k.shape[-1]  # Get head_dim from cache_k shape
                row_order = torch.arange(head_dim, dtype=torch.long)
            cache_k_shard[head_id] = cache_k[:, head_id, subset_head_seq_ids][:, :, row_order]
            cache_v_shard[head_id] = cache_v[:, head_id, subset_head_seq_ids][:, :, :] # row_ordering[head_id]

        self.pistil.model_shard.all_layers[layer_id].kv_cache.load_from_pretrained(cache_k_shard, cache_v_shard)
        self.pistil.model_shard.all_layers[layer_id].kv_cache.shards = shard
        self.pistil.model_shard.all_layers[layer_id].kv_cache.shards_inv = shard_inv
        self.pistil.model_shard.all_layers[layer_id].rotary_emb.cached_ids = prefill_seq_len

    def set_input_sharding(self, model_dim_sharding, ff_dim_sharding):
        self.pistil.act_model_dim_sharding = model_dim_sharding
        self.pistil.act_ff_dim_sharding = ff_dim_sharding

    def configure_attention_nic(self, head_receive_order, qkv_nic_config_algorithm, o_nic_config_algorithm):
        self.pistil.head_receive_order         = head_receive_order
        self.pistil.qkv_nic_config_algorithm   = qkv_nic_config_algorithm
        self.pistil.o_nic_config_algorithm     = o_nic_config_algorithm
