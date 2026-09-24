import math
import torch

class KVCache:
    def __init__(self, my_model, layer_id, batch_size, max_seq_len, gqa, kv_heads, head_dim, q_pre_attn_scalar, dtype):
        self.my_model           = my_model
        self.layer_id           = layer_id
        self.batch_size         = batch_size
        self.max_seq_len        = max_seq_len
        self.gqa                = gqa
        self.kv_heads           = kv_heads
        self.head_dim           = head_dim
        self.q_pre_attn_scalar  = q_pre_attn_scalar
        self.dtype              = dtype

        self.cached_ids         = 0
        self.cache_k            = None
        self.cache_v            = None

        #self.reset()
    
    def load_from_pretrained(self, k_params, v_params):
        self.cached_ids         = k_params.size()[2]
        self.cache_k            = k_params.to(self.dtype)
        self.cache_v            = v_params.to(self.dtype)

    def reset(self):
        self.cached_ids         = 0
        self.cache_k            = torch.zeros((self.batch_size, self.kv_heads, self.max_seq_len, self.head_dim)).to(self.dtype)
        self.cache_v            = torch.zeros((self.batch_size, self.kv_heads, self.max_seq_len, self.head_dim)).to(self.dtype)
    
    # soft(QK^T)V
    def create_mask(self, seq_len, start_pos):
        mask = torch.full((seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seq_len, start_pos)), mask]).to(self.dtype)
        return mask

    def write_kv(self, seq_len, k_vector, v_vector):
        self.cache_k[:self.batch_size, :, self.cached_ids : self.cached_ids + seq_len] = k_vector
        self.cache_v[:self.batch_size, :, self.cached_ids : self.cached_ids + seq_len] = v_vector # BS Head, Seq, Head Dim
    
    def forward_repeat_kv(self, seq_len):
        def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
            """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
            bs, n_kv_heads, slen, head_dim = x.shape
            if n_rep == 1:
                return x
            x = x[:, :, None, :, :].expand(bs, n_kv_heads, n_rep, slen, head_dim)
            return x.reshape(bs, n_kv_heads * n_rep, slen, head_dim)

        keys            = self.cache_k[:self.batch_size, :, : self.cached_ids + seq_len] 
        values          = self.cache_v[:self.batch_size, :, : self.cached_ids + seq_len]

        self.keys       = repeat_kv(keys, self.gqa)  # (bs, n_local_heads, cache_len + seqlen, head_dim) # heads are repeated in order...
        self.values     = repeat_kv(values, self.gqa)

    def forward_QKT(self, seq_len, q_vector):
        q_vector        = q_vector * self.q_pre_attn_scalar # / math.sqrt(self.head_dim)
        scores          = torch.matmul(q_vector, self.keys.transpose(2, 3)).float()  # bs, heads, seq len, cached ids
        return scores

    def forward_soft(self, seq_len, scores):
        mask            = self.create_mask(seq_len, self.cached_ids)
        scores          = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        soft_scores     = torch.nn.functional.softmax(scores.float(), dim=-1).to(self.dtype)
        return soft_scores

    def forward_V(self, seq_len, soft_scores):
        output          = torch.matmul(soft_scores, self.values)  # (bs, n_local_heads, seqlen, head_dim)
        hidden_states   = output.transpose(1, 2).contiguous().view(self.batch_size, seq_len, -1).to(self.dtype)
        self.cached_ids += seq_len
        return hidden_states

