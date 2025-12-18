import math
import torch

class KVCache:
    def __init__(self, parent, layer_id, batch_size, gqa, head_dim, q_pre_attn_scalar, dtype):
        self.parent             = parent
        self.layer_id           = layer_id
        self.batch_size         = batch_size
        self.max_seq_len        = None
        self.gqa                = gqa
        self.kv_heads           = None
        self.head_dim           = head_dim
        self.q_pre_attn_scalar  = q_pre_attn_scalar
        self.dtype              = dtype

        self.cached_ids         = {}
        self.cache_k            = {}
        self.cache_v            = {}
        self.shards             = None
        self.shards_inv         = None

        
    
    def load_from_pretrained(self, k_params, v_params):
        for head_id in k_params.keys():
            self.cached_ids[head_id] = k_params[head_id].size()[1]
            self.cache_k[head_id][:, 0:k_params[head_id].size()[1], :]    = k_params[head_id]
            self.cache_v[head_id][:, 0:v_params[head_id].size()[1], :]    = v_params[head_id]            

    def reset(self):
        if self.shards == None:
            print("Warning: Did not initialize KV$ shard...")
            exit()

        for head_id in self.shards[1].keys():
            self.cached_ids[head_id] = 0
            self.cache_k[head_id] = torch.zeros((self.batch_size, len(self.shards[1][head_id]), self.shards[0][1] - self.shards[0][0])).to(self.dtype)
            self.cache_v[head_id] = torch.zeros((self.batch_size, len(self.shards[1][head_id]), self.shards[0][1] - self.shards[0][0])).to(self.dtype)

    def write_kv_shard(self, head_id, seq_len, k_vector, v_vector):
        self.cache_k[head_id][:self.batch_size, self.cached_ids[head_id] + seq_len - 1, :] = k_vector
        self.cache_v[head_id][:self.batch_size, self.cached_ids[head_id] + seq_len - 1, :] = v_vector
        self.cached_ids[head_id] += seq_len
    
    def forward_QKT_shard(self, gqa_q):
        scores = {}
        max_scores = {}
        for head_id in self.cache_k.keys():
            # Safety check: skip if head_id doesn't exist in gqa_q
            if head_id not in gqa_q:
                continue
            gqa_q[head_id] = gqa_q[head_id] * self.q_pre_attn_scalar # / math.sqrt(self.head_dim)
            scores[head_id] = torch.matmul(gqa_q[head_id], self.cache_k[head_id][:, :self.cached_ids[head_id]].unsqueeze(0).transpose(2, 3)).float() 
            if self.cached_ids[head_id] == 0:
                result_shape = list(scores[head_id].size())[0:-1]
                max_scores[head_id] = torch.full(result_shape, float('-inf'))
            else:
                max_scores[head_id] = torch.max(scores[head_id], dim=-1).values
        return scores, max_scores
    
    def forward_softmax_shard(self, scores, max_scores):
        def calc_exp_sum(scores):
            #s_QKT = torch.nn.functional.softmax(scores, dim=-1) # original
            scores_exp = torch.exp(scores)
            scores_sum = torch.sum(scores_exp, dim=-1)
            return scores_exp, scores_sum

        exp_sum = {}
        soft_scores = {}
        for head_id in scores:
            # Safety check: skip if head_id doesn't exist in max_scores
            if head_id not in max_scores:
                continue
            scores[head_id] = scores[head_id].float()
            scores[head_id] -= max_scores[head_id].unsqueeze(-1).float()
            soft_scores[head_id], exp_sum[head_id] = calc_exp_sum(scores[head_id])
        return soft_scores, exp_sum
    
    def forward_sQKT_V_shard(self, soft_scores):
        sdpa_res = {}
        for head_id in self.cache_v.keys():
            # Safety check: skip if head_id doesn't exist in soft_scores
            if head_id not in soft_scores:
                continue
            # Make sure that this is in float32 so that the accumulation across chiplets works
            sdpa_res[head_id] = torch.matmul(soft_scores[head_id].to(torch.float32), self.cache_v[head_id][:, :self.cached_ids[head_id]].to(torch.float32).unsqueeze(1))
        return sdpa_res
        
    