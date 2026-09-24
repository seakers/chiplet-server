import math
import torch

class RotaryEmbeddings:
    def __init__(self, parent, layer_id, cos, sin, kv_cache):
        self.parent             = parent
        self.layer_id           = layer_id
        self.cos                = cos
        self.sin                = sin
        self.kv_cache           = kv_cache
        
        self.cached_ids         = 0
    
    def forward_shard(self, shard, q_shard_info, k_shard_info):
        batch_size = shard.size()[0]
        seq_len = shard.size()[1]
        head_dim = self.parent.head_dim
        num_heads = self.parent.head_dim * self.parent.num_heads

        position_ids    = torch.arange(self.cached_ids, self.cached_ids + seq_len).unsqueeze(0)
        cos, sin        = self.cos[position_ids], self.sin[position_ids] 
        cos, sin        = cos.unsqueeze(1), sin.unsqueeze(1)

        offset = 0
        
        for head, (ss, se), (es, ee) in q_shard_info[1]:
            cols_per_chiplet = ee - es + se - ss
            head_pos_ids = torch.cat((torch.arange(ss, se), torch.arange(es, ee)))
            rotated_shard = torch.cat((-1 * shard[:, :, offset+se-ss : offset+cols_per_chiplet], shard[:, :, offset : offset+se-ss]), dim=-1)
            
            shard[:, :, offset : offset+cols_per_chiplet] = (shard[:, :, offset : offset+cols_per_chiplet] * cos[:,:,:, head_pos_ids]) + (rotated_shard * sin[:,:,:, head_pos_ids])
            offset += cols_per_chiplet
        
        qk_split = offset

        for head, (ss, se), (es, ee) in k_shard_info[1]:
            cols_per_chiplet = ee - es + se - ss
            head_pos_ids = torch.cat((torch.arange(ss, se), torch.arange(es, ee)))
            rotated_shard = torch.cat((-1 * shard[:, :, offset+se-ss : offset+cols_per_chiplet], shard[:, :, offset : offset+se-ss]), dim=-1)

            shard[:, :, offset : offset+cols_per_chiplet] = (shard[:, :, offset : offset+cols_per_chiplet] * cos[:,:,:, head_pos_ids]) + (rotated_shard * sin[:,:,:, head_pos_ids])
            offset += cols_per_chiplet

        self.shard_q = shard[:, :, 0:qk_split]
        self.shard_k = shard[:, :, qk_split:offset]
        self.shard_v = shard[:, :, offset:]
        
        cached_ids = self.cached_ids
        self.cached_ids += seq_len

        return self.shard_q, self.shard_k, self.shard_v, cached_ids
        
