import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import scipy.signal
import os
import csv
import math
from pathlib import Path
from copy import deepcopy

# Dynamically load PistilSimulator from pistil_runner.py
import importlib.util as _importlib_util

_SIM_ROOT = Path(__file__).parent / "sim-v2-4-pistil-sim-clean"
_PISTIL_RUNNER_PATH = _SIM_ROOT / "pistil_runner.py"
_spec = _importlib_util.spec_from_file_location("pistil_runner", _PISTIL_RUNNER_PATH)
_pistil_module = _importlib_util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_pistil_module)
PistilSimulator = _pistil_module.PistilSimulator


# ============== Helper Functions ==============
def _round_to_nearest(value, choices):
    """Helper to snap a numeric value to the closest allowed choice."""
    choices = np.array(choices, dtype=float)
    idx = (np.abs(choices - float(value))).argmin()
    return choices[idx]


def discounted_cumulative_sums(x, discount):
    """Compute discounted cumulative sums (for GAE) [2]."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# ============== Transformer Building Blocks ==============
# Mirrors transformer_rand_weights_state.py [1]

class CustomDecoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward=32, dropout=0.1):
        super(CustomDecoderLayer, self).__init__()

        self.linear1  = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        # Self-attention
        tgt2 = F.scaled_dot_product_attention(tgt, tgt, tgt, tgt_mask)
        tgt  = tgt + self.dropout1(tgt2)
        tgt  = self.norm1(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt  = tgt + self.dropout2(tgt2)
        tgt  = self.norm2(tgt)

        return tgt


class CustomTransformerDecoder(nn.Module):
    def __init__(self, d_model, num_layers, dim_feedforward=32, dropout=0.1):
        super(CustomTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            CustomDecoderLayer(d_model, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, tgt_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, tgt_mask)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# ============== Transformer Actor ==============

class TransformerActor(nn.Module):
    """
    Transformer-based Actor for Pistil hardware design.
    Replaces MLPActor; uses per-decision output heads like
    transformer_architecture_informed_state.py [4].
    Each decision variable has its own Linear output head,
    but all share the same transformer encoder trunk [1].
    """
    def __init__(self, device, params, num_decisions=9, action_dims=None):
        super(TransformerActor, self).__init__()
        self.device       = device
        self.num_decisions = num_decisions
        self.action_dims  = action_dims   # list[int]: num choices per decision
        self.clip_ratio   = params['clip_ratio']

        self.dense_dim = 16  # lightweight [1]

        self.encoder             = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim,
            num_layers=1,
            dim_feedforward=self.dense_dim,
            dropout=0.1,
        )
        # One output head per decision variable [4]
        self.output_heads = nn.ModuleList([
            nn.Linear(self.dense_dim, dim) for dim in action_dims
        ])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9
        )
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, decision_idx):
        """
        x             : [batch, seq_len]
        decision_idx  : int — selects which output head to use [4]
        returns       : [batch, seq_len, action_dims[decision_idx]]
        """
        with autocast(device_type=self.device.type, dtype=torch.float16):
            x    = x.unsqueeze(-1)                          # [B, S, 1]
            x    = self.encoder(x)                           # [B, S, D]
            x    = self.positional_encoding(x)               # [B, S, D]
            mask = self.generate_square_subsequent_mask(
                x.size(1)
            ).to(self.device)
            x    = self.transformer_decoder(x, tgt_mask=mask)  # [B, S, D]
            x    = self.output_heads[decision_idx](x)        # [B, S, C]
            x    = F.softmax(x, dim=-1)
        return x

    def sample_action(self, observations, decision_idx):
        """
        Sample one action per batch element for the given decision step.
        Mirrors Actor.sample_action in transformer_rand_weights_state.py [1].
        """
        if len(observations[0]) == 0:
            observations = [[0.0] for _ in range(len(observations))]

        obs_tensor = torch.tensor(
            observations, dtype=torch.float32
        ).to(self.device)
        output      = self.forward(obs_tensor, decision_idx)
        output_last = output[:, -1, :]                       # [B, C]

        log_probs = torch.log(output_last + 1e-10)
        dist      = torch.distributions.Categorical(logits=log_probs)
        actions   = dist.sample()
        action_log_probs = log_probs.gather(
            1, actions.unsqueeze(-1)
        ).squeeze(-1)

        return action_log_probs, actions

    def ppo_update(self, observations, actions, old_logprobs,
                   advantages, decision_indices):
        """
        Vectorised PPO actor update — one pass over the full batch.
        decision_indices is a plain Python list of ints (one per sample) [2].
        """
        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type, dtype=torch.float16):
            new_log_probs = torch.zeros(
                len(observations), dtype=torch.float32, device=self.device
            )

            # Group samples by decision index to avoid a Python loop
            # over every individual sample [1][4]
            dec_idx_tensor = torch.tensor(
                decision_indices, dtype=torch.long, device=self.device
            )
            for d in range(self.num_decisions):
                mask = dec_idx_tensor == d
                if not mask.any():
                    continue

                pred_probs = self.forward(
                    observations[mask], d
                )[:, -1, :]                                   # [m, C]
                log_p      = torch.log(pred_probs + 1e-10)
                gathered   = torch.sum(
                    torch.mul(
                        F.one_hot(
                            actions[mask].long(),
                            num_classes=self.action_dims[d]
                        ).float(),
                        log_p
                    ),
                    dim=-1,
                )
                new_log_probs[mask] = gathered

            ratio = torch.exp(new_log_probs - old_logprobs)
            min_advantage = torch.where(
                advantages > 0,
                (1 + self.clip_ratio) * advantages,
                (1 - self.clip_ratio) * advantages,
            )
            policy_loss = -torch.mean(
                torch.min(ratio * advantages, min_advantage)
            )

        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        kl = torch.abs(torch.mean(new_log_probs - old_logprobs))
        return policy_loss.item(), kl.item()


# ============== Transformer Critic ==============

class TransformerCritic(nn.Module):
    """
    Transformer-based Critic for Pistil hardware design.
    Replaces MLPCritic; architecture mirrors Critic in
    transformer_rand_weights_state.py [1].
    Predicts a value vector of size num_objectives.
    """
    def __init__(self, device, params, num_decisions=9, num_objectives=2):
        super(TransformerCritic, self).__init__()
        self.device          = device
        self.num_objectives  = num_objectives
        self.num_decisions   = num_decisions

        self.dense_dim = 16  # lightweight [1]

        self.encoder             = nn.Linear(1, self.dense_dim)
        self.positional_encoding = PositionalEncoding(self.dense_dim)
        self.transformer_decoder = CustomTransformerDecoder(
            d_model=self.dense_dim,
            num_layers=1,
            dim_feedforward=self.dense_dim,
            dropout=0.1,
        )
        self.output_layer = nn.Linear(self.dense_dim, num_objectives)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1000, gamma=0.9
        )
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        with autocast(device_type=self.device.type, dtype=torch.float16):
            x    = x.unsqueeze(-1)                           # [B, S, 1]
            x    = self.encoder(x)                            # [B, S, D]
            x    = self.positional_encoding(x)                # [B, S, D]
            mask = self.generate_square_subsequent_mask(
                x.size(1)
            ).to(self.device)
            x    = self.transformer_decoder(x, tgt_mask=mask) # [B, S, D]
            x    = self.output_layer(x)                       # [B, S, num_objectives]
            x    = x[:, -1, :]                                # [B, num_objectives] last token
        return x

    def sample_critic(self, observations):
        """
        Accepts a list of observation lists (possibly variable length).
        Pads each to (num_objectives + num_decisions) and returns value tensor.
        Mirrors Critic.sample_critic in transformer_rand_weights_state.py [1].
        """
        input_dim = self.num_decisions + self.num_objectives
        padded = []
        for obs in observations:
            o = list(obs)
            while len(o) < input_dim:
                o.append(0.0)
            padded.append(o[:input_dim])

        obs_tensor = torch.tensor(padded, dtype=torch.float32).to(self.device)
        return self.forward(obs_tensor)

    def ppo_update(self, observations, returns, weights):
        """
        Critic PPO update.
        Mirrors Critic.ppo_update in transformer_rand_weights_state.py [1].
        """
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            pred_values = self.forward(observations)              # [batch, num_objectives]
            pred_reward = torch.sum(
                -pred_values * weights, dim=-1
            )                                                     # [batch]
            value_loss  = torch.mean((pred_reward - returns) ** 2)

        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return value_loss.item()


# ============== Pistil Evaluator Wrapper ==============
class PistilEvaluator:
    """
    Evaluator wrapper for Pistil simulator.
    Saves ALL metrics to points.csv for comprehensive data capture [6].
    """

    TMAC_CHOICES       = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
    MEM_BUF_CHOICES    = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    BANK_GROUP_CHOICES = [1, 2, 3, 4]
    RANK_CHOICES       = [1, 2, 3, 4]
    FRAC_BANK_CHOICES  = [0.5, 0.75, 1.0]

    @staticmethod
    def _generate_power_of_2_choices(min_val, max_val):
        choices, power = [], 0
        while True:
            value = 2 ** power
            if value < min_val:
                power += 1
                continue
            if value > max_val:
                break
            choices.append(value)
            power += 1
        return choices if choices else [min_val]

    def __init__(
        self,
        model_name="llama3-8b",
        output_dir=None,
        allowed_num_cus=None,
        batch_bounds=(1, 64),
        kv_cache_bounds=(1024, 8192),
    ):
        if allowed_num_cus is None:
            allowed_num_cus = [16, 32, 64, 96, 128]

        self.model_name     = model_name
        self.allowed_num_cus = sorted(
            set(int(v) for v in allowed_num_cus if v % 4 == 0)
        )

        self.BATCH_CHOICES    = self._generate_power_of_2_choices(*batch_bounds)
        self.KV_CACHE_CHOICES = self._generate_power_of_2_choices(*kv_cache_bounds)

        self.all_choices = [
            self.allowed_num_cus,   # 0: num_cus
            self.TMAC_CHOICES,      # 1: tmacs
            self.MEM_BUF_CHOICES,   # 2: mem_buf_cap
            self.MEM_BUF_CHOICES,   # 3: net_buf_cap
            self.BANK_GROUP_CHOICES,# 4: bank_groups
            self.RANK_CHOICES,      # 5: ranks
            self.FRAC_BANK_CHOICES, # 6: frac_bank_cap
            self.BATCH_CHOICES,     # 7: batch_size
            self.KV_CACHE_CHOICES,  # 8: kv_cache
        ]

        self.action_dims   = [len(c) for c in self.all_choices]
        self.num_decisions = len(self.all_choices)
        self.num_objectives = 2   # latency, energy

        self.sim = PistilSimulator()

        self.points_csv_path        = output_dir
        self.points_csv_initialized = False
        self._initialize_points_csv()

        self.max_latency = 1000.0  # ms
        self.max_energy  = 5000.0  # mJ

    # ------------------------------------------------------------------
    # CSV helpers (unchanged from original rlPistil.py [6])
    # ------------------------------------------------------------------

    def _initialize_points_csv(self):
        if not self.points_csv_path:
            return

        decision_cols = [
            "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
            "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
            "batch_size", "kv_cache",
        ]
        metric_cols = [
            "latency_ms", "energy_mJ",
            "latency_per_token_ms", "energy_per_inference_mJ", "energy_per_token_mJ",
            "prefill_tokens_per_sec",
            "average_power_W", "system_power_W",
            "average_power_mem_W", "average_power_comp_W", "average_power_net_W",
            "system_cost",
            "chiplet_silicon_cost", "memory_cost_2xHBLC", "package_cost",
            "package_silicon_cost", "package_memory_cost", "package_substrate_cost",
            "system_silicon_cost", "system_memory_cost", "system_substrate_cost",
            "system_pcb_cost",
            "avg_comp_util", "avg_mem_util",
            "system_compute_TOPS", "system_bandwidth_TBps", "system_capacity_GB",
            "num_packages", "num_chiplets",
            "hblc_capacity_GB", "hblc_bandwidth_GBps", "hblc_bw_per_capacity",
            "peak_buffer_MB", "avg_cache_util_MB",
            "model_weight_capacity_GB", "kv_cache_capacity_per_batch_GB",
            "total_model_capacity_GB",
            "cores_per_cu", "mem_buffer_size", "net_buffer_size", "tmacs_per_core",
            "hblc_bank_groups", "hblc_ranks", "hblc_frac_bank_cap",
        ]

        if (
            not os.path.exists(self.points_csv_path)
            or os.path.getsize(self.points_csv_path) == 0
        ):
            with open(self.points_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(decision_cols + metric_cols + ["algorithm"])
            print(f"Initialized points.csv at {self.points_csv_path}")
        self.points_csv_initialized = True

    def _save_to_points_csv(self, params, metrics_dict):
        metric_cols = [
            "latency_ms", "energy_mJ",
            "latency_per_token_ms", "energy_per_inference_mJ", "energy_per_token_mJ",
            "prefill_tokens_per_sec",
            "average_power_W", "system_power_W",
            "average_power_mem_W", "average_power_comp_W", "average_power_net_W",
            "system_cost",
            "chiplet_silicon_cost", "memory_cost_2xHBLC", "package_cost",
            "package_silicon_cost", "package_memory_cost", "package_substrate_cost",
            "system_silicon_cost", "system_memory_cost", "system_substrate_cost",
            "system_pcb_cost",
            "avg_comp_util", "avg_mem_util",
            "system_compute_TOPS", "system_bandwidth_TBps", "system_capacity_GB",
            "num_packages", "num_chiplets",
            "hblc_capacity_GB", "hblc_bandwidth_GBps", "hblc_bw_per_capacity",
            "peak_buffer_MB", "avg_cache_util_MB",
            "model_weight_capacity_GB", "kv_cache_capacity_per_batch_GB",
            "total_model_capacity_GB",
            "cores_per_cu", "mem_buffer_size", "net_buffer_size", "tmacs_per_core",
            "hblc_bank_groups", "hblc_ranks", "hblc_frac_bank_cap",
        ]

        if not self.points_csv_initialized:
            self._initialize_points_csv()

        row_data = [
            params["num_cus"], params["num_tmacs"],
            params["mem_buf_cap"], params["net_buf_cap"],
            params["mem_banks_per_group"], params["mem_ranks"],
            params["mem_frac_bank_cap"],
            params["batch_size"], params["kv_cache"],
        ]
        for col in metric_cols:
            row_data.append(metrics_dict.get(col, 0.0))
        row_data.append(getattr(self, 'algorithm_label', 'Deep RL'))

        try:
            with open(self.points_csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row_data)
            print(
                f"[PISTIL RL] Saved point: {params['num_cus']} CUs, "
                f"latency={metrics_dict.get('latency_ms', 0):.2f}ms, "
                f"energy={metrics_dict.get('energy_mJ', 0):.2f}mJ"
            )
        except Exception as e:
            print(f"[PISTIL RL] Error saving to points.csv: {e}")
            raise

    # ------------------------------------------------------------------
    # Design helpers
    # ------------------------------------------------------------------

    def decode_actions(self, action_indices):
        return [self.all_choices[i][action_indices[i]] for i in range(self.num_decisions)]

    def normalize_design(self, design_values):
        normalized = []
        for i, val in enumerate(design_values):
            choices = self.all_choices[i]
            min_val, max_val = min(choices), max(choices)
            norm = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            normalized.append(norm)
        return normalized

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, action_indices):
        """
        Evaluate a design (list of 9 action indices).
        Returns: (normalized_objectives, raw_objectives)
        """
        design_values = self.decode_actions(action_indices)

        params = {
            "num_cus":           int(design_values[0]),
            "num_tmacs":         int(design_values[1]),
            "mem_buf_cap":       float(design_values[2]),
            "net_buf_cap":       float(design_values[3]),
            "mem_banks_per_group": int(design_values[4]),
            "mem_ranks":         int(design_values[5]),
            "mem_frac_bank_cap": float(design_values[6]),
            "model":             self.model_name,
            "batch_size":        int(design_values[7]),
            "kv_cache":          int(design_values[8]),
            # Fixed constants
            "w_dtype":              0.5,
            "kv_dtype":             1.0,
            "prefill":              "False",
            "prefill_chunk_size":   0,
            "prefill_cached":       0,
            "sim_num_layers":       -1,
            "lm_head":              "False",
            "plot_exe":             "False",
            "gen_trace":            True,
            "sim_standalone":       True,
            "base_config":          "pistil-sys-base.json",
        }

        try:
            # Edge-case guard: avoid seq_len=0 crash in kv_cache.write_kv
            if (
                params.get("prefill") == "False"
                and int(params.get("prefill_cached", 0)) == 0
                and int(params.get("kv_cache", 0)) > 0
            ):
                params = dict(params)
                params["prefill_cached"] = 1
                print("[PISTIL RL] Pre-flight: set prefill_cached=1 to avoid seq_len=0 crash")

            print(
                f"[PISTIL RL] Evaluating: {params['num_cus']} CUs, "
                f"{params['num_tmacs']} TMACs, "
                f"batch={params['batch_size']}, kv_cache={params['kv_cache']}"
            )

            self.sim.run_dse_point(params)
            metrics_dict = self._load_all_metrics(params)
            latency_ms   = metrics_dict["latency_ms"]
            energy_mJ    = metrics_dict["energy_mJ"]

            self._save_to_points_csv(params, metrics_dict)
            print(f"[PISTIL RL] Results: latency={latency_ms:.2f}ms, energy={energy_mJ:.2f}mJ")

        except Exception as e:
            print(f"[PISTIL RL] ERROR: {e}")
            latency_ms, energy_mJ = 1e9, 1e9

        self.max_latency = max(self.max_latency, latency_ms)
        self.max_energy  = max(self.max_energy,  energy_mJ)

        raw_objectives  = [latency_ms, energy_mJ]
        norm_objectives = [
            latency_ms / self.max_latency,
            energy_mJ  / self.max_energy,
        ]
        return norm_objectives, raw_objectives

    def _load_all_metrics(self, params):
        results_dir = os.path.join(self.sim.sim_root, self.sim.results_dir)
        if not os.path.exists(results_dir):
            raise FileNotFoundError(f"Pistil results directory not found: {results_dir}")

        csv_files = [
            os.path.join(results_dir, f)
            for f in os.listdir(results_dir)
            if f.endswith(".csv")
        ]
        if not csv_files:
            raise FileNotFoundError(f"No Pistil result CSVs found in: {results_dir}")

        latest_csv = max(csv_files, key=os.path.getmtime)
        metrics    = {}
        with open(latest_csv, "r") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) != 2:
                    continue
                key, value = row
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value

        total_latency_s  = float(metrics.get("total_latency_s", 0.0))
        total_energy_J   = float(metrics.get("total_energy_J",  0.0))
        num_layers       = float(metrics.get("num_layers",       1.0))
        sim_num_layers   = float(metrics.get("sim_num_layers",  -1.0))
        batch_size       = float(metrics.get("batch_size", params["batch_size"]))
        num_chiplets     = float(metrics.get("num_chiplets", params["num_cus"]))

        if sim_num_layers != num_layers and sim_num_layers != -1:
            scale      = num_layers / sim_num_layers
            latency_ms = total_latency_s * scale * 1000.0
            energy_mJ  = total_energy_J  * scale * 1000.0 * num_chiplets
        else:
            latency_ms = total_latency_s * 1000.0
            energy_mJ  = total_energy_J  * 1000.0 * num_chiplets

        latency_per_token_ms      = latency_ms / batch_size if batch_size > 0 else 0.0
        energy_per_inference_mJ   = energy_mJ
        energy_per_token_mJ       = energy_mJ / batch_size if batch_size > 0 else 0.0

        num_cus       = float(metrics.get("num_chiplets", params["num_cus"]))
        avg_power_W   = float(metrics.get("average_power_W", 0.0))
        system_power_W = num_cus * avg_power_W
        system_cost   = float(metrics.get("system_cost", 0.0))

        prefill_batch      = metrics.get("prefill_batch", False)
        prefill_chunk_size = float(metrics.get("prefill_chunk_size", 0.0))
        if prefill_batch and prefill_chunk_size > 0 and latency_ms > 0:
            prefill_tokens_per_sec = (
                prefill_chunk_size * batch_size
            ) / (latency_ms / 1000.0)
        else:
            prefill_tokens_per_sec = (
                batch_size / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
            )

        avg_comp_util  = float(metrics.get("avg_comp_util",  0.0))
        avg_mem_util   = float(metrics.get("avg_mem_util",   0.0))
        num_cus        = float(metrics.get("num_chiplets",   params["num_cus"]))
        avg_power_W    = float(metrics.get("average_power_W", 0.0))
        system_power_W = num_cus * avg_power_W
        system_cost    = float(metrics.get("system_cost",    0.0))

        # Build the comprehensive metrics dictionary [6]
        all_metrics = dict(metrics)
        all_metrics.update({
            "latency_ms":                    float(latency_ms),
            "energy_mJ":                     float(energy_mJ),
            "latency_per_token_ms":          latency_per_token_ms,
            "energy_per_inference_mJ":       energy_per_inference_mJ,
            "energy_per_token_mJ":           energy_per_token_mJ,
            "prefill_tokens_per_sec":        prefill_tokens_per_sec,
            "average_power_W":               avg_power_W,
            "system_power_W":                system_power_W,
            "average_power_mem_W":           float(metrics.get("average_power_mem_W",  0.0)),
            "average_power_comp_W":          float(metrics.get("average_power_comp_W", 0.0)),
            "average_power_net_W":           float(metrics.get("average_power_net_W",  0.0)),
            "system_cost":                   system_cost,
            "chiplet_silicon_cost":          float(metrics.get("chiplet_silicon_cost",   0.0)),
            "memory_cost_2xHBLC":            float(metrics.get("memory_cost_2xHBLC",     0.0)),
            "package_cost":                  float(metrics.get("package_cost",            0.0)),
            "package_silicon_cost":          float(metrics.get("package_silicon_cost",    0.0)),
            "package_memory_cost":           float(metrics.get("package_memory_cost",     0.0)),
            "package_substrate_cost":        float(metrics.get("package_substrate_cost",  0.0)),
            "system_silicon_cost":           float(metrics.get("system_silicon_cost",     0.0)),
            "system_memory_cost":            float(metrics.get("system_memory_cost",      0.0)),
            "system_substrate_cost":         float(metrics.get("system_substrate_cost",   0.0)),
            "system_pcb_cost":               float(metrics.get("system_pcb_cost",         0.0)),
            "avg_comp_util":                 avg_comp_util,
            "avg_mem_util":                  avg_mem_util,
            "num_chiplets":                  num_cus,
            "system_compute_TOPS":           float(metrics.get("system_compute_TOPS",    0.0)),
            "system_bandwidth_TBps":         float(metrics.get("system_bandwidth_TBps",  0.0)),
            "system_capacity_GB":            float(metrics.get("system_capacity_GB",     0.0)),
            "num_packages":                  float(metrics.get("num_packages",           0.0)),
            "hblc_capacity_GB":              float(metrics.get("hblc_capacity_GB",       0.0)),
            "hblc_bandwidth_GBps":           float(metrics.get("hblc_bandwidth_GBps",    0.0)),
            "hblc_bw_per_capacity":          float(metrics.get("hblc_bw_per_capacity",   0.0)),
            "peak_buffer_MB":                float(metrics.get("peak_buffer_MB",          0.0)),
            "avg_cache_util_MB":             float(metrics.get("avg_cache_util_MB",       0.0)),
            "model_weight_capacity_GB":      float(metrics.get("model_weight_capacity_GB",       0.0)),
            "kv_cache_capacity_per_batch_GB":float(metrics.get("kv_cache_capacity_per_batch_GB", 0.0)),
            "total_model_capacity_GB":       float(metrics.get("total_model_capacity_GB",        0.0)),
            "cores_per_cu":                  float(metrics.get("cores_per_cu",           0.0)),
            "mem_buffer_size":               float(metrics.get("mem_buffer_size",        0.0)),
            "net_buffer_size":               float(metrics.get("net_buffer_size",        0.0)),
            "tmacs_per_core":                float(metrics.get("tmacs_per_core",         0.0)),
            "hblc_bank_groups":              float(metrics.get("hblc_bank_groups",       0.0)),
            "hblc_ranks":                    float(metrics.get("hblc_ranks",             0.0)),
            "hblc_frac_bank_cap":            float(metrics.get("hblc_frac_bank_cap",     0.0)),
        })

        return all_metrics


# ============== Main PPO Training Loop ==============
def run_ppo_pistil(params):
    """
    Run PPO optimization for Pistil hardware design using transformer models.
    Based on the approach in ppo_rand_weights_state.py [2] and rlCascade.py [5].
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    evaluator = PistilEvaluator(
        model_name=params.get('model_name',       'llama3-8b'),
        output_dir=params.get('output_dir',       None),
        allowed_num_cus=params.get('allowed_num_cus', None),
        batch_bounds=params.get('batch_bounds',   (1, 64)),
        kv_cache_bounds=params.get('kv_cache_bounds', (1024, 8192)),
    )
    evaluator.algorithm_label = 'Deep RL'

    num_decisions  = evaluator.num_decisions   # 9
    num_objectives = evaluator.num_objectives  # 2
    action_dims    = evaluator.action_dims     # list[int], length 9

    # Instantiate transformer-based actor and critic [1]
    actor = TransformerActor(
        device, params,
        num_decisions=num_decisions,
        action_dims=action_dims,
    ).to(device)

    critic = TransformerCritic(
        device, params,
        num_decisions=num_decisions,
        num_objectives=num_objectives,
    ).to(device)

    # Warm up lazy layers [2]
    dummy = torch.zeros(
        (1, num_objectives + num_decisions), dtype=torch.float32
    ).to(device)
    actor(dummy, 0)
    critic(dummy)

    epochs         = params['num_epochs']
    mini_batch_size = params['mini_batch_size']
    gamma          = params['gamma']
    lam            = params['lambda']

    all_designs    = []
    all_objectives = []

    for epoch in range(epochs):

        batch_rewards          = [[] for _ in range(mini_batch_size)]
        batch_actions          = [[] for _ in range(mini_batch_size)]
        batch_logprobs         = [[] for _ in range(mini_batch_size)]
        batch_observations     = [[] for _ in range(mini_batch_size)]
        batch_designs          = [[] for _ in range(mini_batch_size)]
        batch_decision_indices = [[] for _ in range(mini_batch_size)]

        # Random scalarization weights [2]
        weights = np.random.rand(mini_batch_size, num_objectives)
        weights = weights / weights.sum(axis=1, keepdims=True)

        for idx in range(mini_batch_size):
            batch_observations[idx] = weights[idx].tolist()

        # --- 1. Sample Actions ---
        with torch.no_grad():
            for decision_step in range(num_decisions):
                obs_for_actor = []
                for idx in range(mini_batch_size):
                    obs = batch_observations[idx].copy()
                    while len(obs) < num_objectives + num_decisions:
                        obs.append(0.0)
                    obs_for_actor.append(obs)

                log_probs_out, actions_out = actor.sample_action(
                    obs_for_actor, decision_step
                )
                log_probs_out = log_probs_out.tolist()
                actions_out   = actions_out.tolist()

                for idx in range(mini_batch_size):
                    batch_actions[idx].append(actions_out[idx])
                    batch_logprobs[idx].append(log_probs_out[idx])
                    batch_decision_indices[idx].append(decision_step)

                    # Normalize action value for the observation sequence [6]
                    action_val = evaluator.all_choices[decision_step][actions_out[idx]]
                    choices    = evaluator.all_choices[decision_step]
                    span       = max(choices) - min(choices)
                    norm_val   = (
                        (action_val - min(choices)) / span if span > 0 else 0.5
                    )
                    batch_observations[idx].append(norm_val)
                    batch_rewards[idx].append(0.0)
                    batch_designs[idx].append(int(actions_out[idx]))

        # --- 2. Evaluate Designs and Calculate Final Rewards ---
        epoch_objectives = []
        for idx in range(mini_batch_size):
            design = batch_designs[idx]
            norm_obj, raw_obj = evaluator.evaluate(design)

            all_designs.append(design)
            all_objectives.append(raw_obj)
            epoch_objectives.append(raw_obj)

            reward = -np.dot(weights[idx], norm_obj)
            batch_rewards[idx][-1] = reward

        # --- 3. Compute Critic Values ---
        critic_values = []
        with torch.no_grad():
            for decision_step in range(num_decisions):
                obs_for_critic = []
                for idx in range(mini_batch_size):
                    obs = batch_observations[idx][
                        :decision_step + 1 + num_objectives
                    ]
                    while len(obs) < num_objectives + num_decisions:
                        obs.append(0.0)
                    obs_for_critic.append(obs)

                crit_vals    = critic.sample_critic(obs_for_critic).cpu().numpy()
                weighted_vals = np.sum(weights * crit_vals, axis=1)
                critic_values.append(weighted_vals)

        # Reshape critic values per sample
        values = [[] for _ in range(mini_batch_size)]
        for step_vals in critic_values:
            for idx, val in enumerate(step_vals):
                values[idx].append(val)
        for idx in range(mini_batch_size):
            values[idx].append(values[idx][-1])  # Bootstrap

        # --- 4. Compute Advantages (GAE) [2] ---
        all_advantages = []
        all_returns    = []
        for idx in range(mini_batch_size):
            rewards_arr = np.array(batch_rewards[idx])
            vals        = np.array(values[idx])

            deltas     = rewards_arr + gamma * vals[1:] - vals[:-1]
            advantages = discounted_cumulative_sums(deltas, gamma * lam)
            returns    = discounted_cumulative_sums(rewards_arr, gamma * lam)

            all_advantages.append(advantages)
            all_returns.append(returns)

        adv_flat             = np.concatenate(all_advantages)
        adv_mean, adv_std    = np.mean(adv_flat), np.std(adv_flat)
        all_advantages       = [
            (a - adv_mean) / (adv_std + 1e-8) for a in all_advantages
        ]

        # --- 5. Prepare Training Tensors ---
        obs_tensor, act_tensor, logp_tensor   = [], [], []
        adv_tensor, ret_tensor, weight_tensor = [], [], []
        dec_idx_list                          = []

        for idx in range(mini_batch_size):
            for step in range(num_decisions):
                obs = batch_observations[idx][:step + 1 + num_objectives]
                obs = obs + [0.0] * (
                    num_objectives + num_decisions - len(obs)
                )

                obs_tensor.append(obs)
                act_tensor.append(batch_actions[idx][step])
                logp_tensor.append(batch_logprobs[idx][step])
                adv_tensor.append(all_advantages[idx][step])
                ret_tensor.append(all_returns[idx][step])
                weight_tensor.append(weights[idx])
                dec_idx_list.append(batch_decision_indices[idx][step])

        obs_tensor    = torch.tensor(obs_tensor,    dtype=torch.float32).to(device)
        act_tensor    = torch.tensor(act_tensor,    dtype=torch.long   ).to(device)
        logp_tensor   = torch.tensor(logp_tensor,   dtype=torch.float32).to(device)
        adv_tensor    = torch.tensor(adv_tensor,    dtype=torch.float32).to(device)
        ret_tensor    = torch.tensor(ret_tensor,    dtype=torch.float32).to(device)
        weight_tensor = torch.tensor(
            np.array(weight_tensor), dtype=torch.float32
        ).to(device)

        # --- 6. PPO Updates [1][2] ---
        for _ in range(params['update_iterations']):
            actor_loss, kl = actor.ppo_update(
                obs_tensor, act_tensor, logp_tensor,
                adv_tensor, dec_idx_list
            )
            if kl > params['target_kl']:
                break

        for _ in range(params['update_iterations']):
            critic_loss = critic.ppo_update(
                obs_tensor, ret_tensor, weight_tensor
            )

        # Logging
        avg_latency = np.mean([o[0] for o in epoch_objectives])
        avg_energy  = np.mean([o[1] for o in epoch_objectives])

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | "
                f"KL: {kl:.5f} | "
                f"Avg Latency: {avg_latency:.2f}ms | Avg Energy: {avg_energy:.2f}mJ"
            )

    return np.array(all_designs), np.array(all_objectives)


# ============== Entry Point ==============
def runPPOPistil(
    num_epochs=50,
    mini_batch_size=8,
    model_name="llama3-8b",
    output_dir=None,
    allowed_num_cus=None,
    batch_bounds=(1, 64),
    kv_cache_bounds=(1024, 8192),
):
    """
    Main entry point for running PPO on Pistil.
    Transformer-based actor/critic replace the original MLP versions [1].
    Similar interface to runGAPistil in gaPistil.py [6].
    """
    params = {
        'num_epochs':        num_epochs,
        'mini_batch_size':   mini_batch_size,
        'learning_rate':     1e-3,
        'gamma':             0.99,
        'lambda':            0.95,
        'clip_ratio':        0.2,
        'target_kl':         0.01,
        'update_iterations': 5,
        'model_name':        model_name,
        'output_dir':        output_dir,
        'allowed_num_cus':   allowed_num_cus,
        'batch_bounds':      batch_bounds,
        'kv_cache_bounds':   kv_cache_bounds,
    }

    print("=" * 80)
    print("Running PPO (Transformer) Optimization for Pistil")
    print(f"  Model: {model_name}")
    print(f"  Epochs: {num_epochs}, Batch Size: {mini_batch_size}")
    print(f"  Batch Bounds: {batch_bounds}, KV Cache Bounds: {kv_cache_bounds}")
    print("=" * 80)

    all_designs, all_objectives = run_ppo_pistil(params)

    print("\n" + "=" * 80)
    print("Optimization Complete!")
    print(f"  Total designs evaluated: {len(all_designs)}")
    if len(all_objectives):
        print(f"  Best Latency: {np.min(all_objectives[:, 0]):.2f}ms")
        print(f"  Best Energy:  {np.min(all_objectives[:, 1]):.2f}mJ")
    print("=" * 80)

    # Build design points [5][6]
    design_points = []
    _eval = PistilEvaluator(
        model_name=model_name,
        allowed_num_cus=allowed_num_cus,
        output_dir=output_dir,
    )
    for i, (design, obj) in enumerate(zip(all_designs, all_objectives)):
        decoded = _eval.decode_actions(design)
        dp = {
            'latency_ms':  obj[0],
            'energy_mJ':   obj[1],
            'design': {
                'num_cus':          int(decoded[0]),
                'num_tmacs':        int(decoded[1]),
                'mem_buf_cap':      float(decoded[2]),
                'net_buf_cap':      float(decoded[3]),
                'bank_groups':      int(decoded[4]),
                'ranks':            int(decoded[5]),
                'frac_bank_cap':    float(decoded[6]),
                'batch_size':       int(decoded[7]),
                'kv_cache':         int(decoded[8]),
            },
            'additional_metrics': {'episode': i},
        }
        design_points.append(dp)

    return np.array(all_designs), np.array(all_objectives), design_points






















# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.amp import GradScaler, autocast
# import scipy.signal
# import os
# import csv
# from pathlib import Path
# from copy import deepcopy

# # Dynamically load PistilSimulator from pistil_runner.py
# import importlib.util as _importlib_util

# _SIM_ROOT = Path(__file__).parent / "sim-v2-4-pistil-sim-clean"
# _PISTIL_RUNNER_PATH = _SIM_ROOT / "pistil_runner.py"
# _spec = _importlib_util.spec_from_file_location("pistil_runner", _PISTIL_RUNNER_PATH)
# _pistil_module = _importlib_util.module_from_spec(_spec)
# assert _spec.loader is not None
# _spec.loader.exec_module(_pistil_module)
# PistilSimulator = _pistil_module.PistilSimulator


# # ============== Helper Functions ==============
# def _round_to_nearest(value, choices):
#     """Helper to snap a numeric value to the closest allowed choice."""
#     choices = np.array(choices, dtype=float)
#     idx = (np.abs(choices - float(value))).argmin()
#     return choices[idx]


# def discounted_cumulative_sums(x, discount):
#     """Compute discounted cumulative sums (for GAE)."""
#     return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# # ============== MLP Actor ==============
# class MLPActor(nn.Module):
#     def __init__(self, device, params, num_decisions=9, action_dims=None):
#         """
#         Actor network for Pistil: outputs probability distribution for each decision variable.
#         Each decision variable has its own set of discrete choices.
#         """
#         super(MLPActor, self).__init__()
#         self.device = device
#         self.num_decisions = num_decisions
#         self.action_dims = action_dims  # List of number of choices for each decision
#         self.clip_ratio = params['clip_ratio']
        
#         hidden_dim = 64
#         # Input: current partial design + weight vector (2 objectives)
#         input_dim = num_decisions + 2
        
#         self.shared_net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )
        
#         # Separate output heads for each decision variable
#         self.output_heads = nn.ModuleList([
#             nn.Linear(hidden_dim, dim) for dim in action_dims
#         ])
        
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
#         self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

#     def forward(self, x, decision_idx):
#         """Forward pass for a specific decision index."""
#         with autocast(device_type=self.device.type, dtype=torch.float16):
#             hidden = self.shared_net(x)
#             logits = self.output_heads[decision_idx](hidden)
#             probs = F.softmax(logits, dim=-1)
#         return probs

#     def sample_action(self, observations, decision_idx):
#         """Sample action for a specific decision variable."""
#         obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
#         probs = self.forward(obs_tensor, decision_idx)
#         dist = torch.distributions.Categorical(probs)
#         actions = dist.sample()
#         log_probs = dist.log_prob(actions)
#         return log_probs, actions

#     def ppo_update(self, observations, actions, old_logprobs, advantages, decision_indices):
#         self.optimizer.zero_grad()
#         total_loss = 0.0
        
#         with autocast(device_type=self.device.type, dtype=torch.float16):
#             for i in range(len(observations)):
#                 obs = observations[i:i+1]
#                 act = actions[i]
#                 old_lp = old_logprobs[i]
#                 adv = advantages[i]
#                 dec_idx = decision_indices[i]
                
#                 probs = self.forward(obs, dec_idx)
#                 dist = torch.distributions.Categorical(probs)
#                 new_logprob = dist.log_prob(act)
                
#                 ratio = torch.exp(new_logprob - old_lp)
#                 clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
#                 total_loss += -torch.min(ratio * adv, clipped_ratio * adv)
            
#             policy_loss = total_loss / len(observations)
        
#         self.scaler.scale(policy_loss).backward()
#         self.scaler.step(self.optimizer)
#         self.scaler.update()
        
#         # Approximate KL
#         with torch.no_grad():
#             kl = torch.mean(old_logprobs - old_logprobs).item()  # Simplified
#         return policy_loss.item(), kl


# # ============== MLP Critic ==============
# class MLPCritic(nn.Module):
#     def __init__(self, device, params, num_decisions=9, num_objectives=2):
#         """
#         Critic network: predicts value for each objective dimension.
#         """
#         super(MLPCritic, self).__init__()
#         self.device = device
#         self.num_objectives = num_objectives
        
#         hidden_dim = 64
#         input_dim = num_decisions + num_objectives
        
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_objectives)
#         )
        
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
#         self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

#     def forward(self, x):
#         with autocast(device_type=self.device.type, dtype=torch.float16):
#             return self.net(x)

#     def sample_critic(self, observations):
#         obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
#         return self.forward(obs_tensor)

#     def ppo_update(self, observations, returns, weights):
#         self.optimizer.zero_grad()
#         with autocast(device_type=self.device.type, dtype=torch.float16):
#             pred_values = self.forward(observations)
#             pred_reward = torch.sum(pred_values * weights, dim=-1)
#             value_loss = torch.mean((pred_reward - returns) ** 2)
        
#         self.scaler.scale(value_loss).backward()
#         self.scaler.step(self.optimizer)
#         self.scaler.update()
        
#         return value_loss.item()


# # ============== Pistil Evaluator Wrapper ==============
# class PistilEvaluator:
#     """
#     Evaluator wrapper for Pistil simulator, adapted from gaPistil.py structure [2].
#     Saves ALL metrics to points.csv for comprehensive data capture.
#     """
    
#     # Decision variable choices (from gaPistil.py [2])
#     TMAC_CHOICES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
#     MEM_BUF_CHOICES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
#     BANK_GROUP_CHOICES = [1, 2, 3, 4]
#     RANK_CHOICES = [1, 2, 3, 4]
#     FRAC_BANK_CHOICES = [0.5, 0.75, 1.0]
    
#     @staticmethod
#     def _generate_power_of_2_choices(min_val, max_val):
#         """Generate list of powers of 2 within the given bounds."""
#         choices = []
#         power = 0
#         while True:
#             value = 2 ** power
#             if value < min_val:
#                 power += 1
#                 continue
#             if value > max_val:
#                 break
#             choices.append(value)
#             power += 1
#         return choices if choices else [min_val]
    
#     def __init__(self, model_name="llama3-8b", output_dir=None, allowed_num_cus=None,
#                  batch_bounds=(1, 64), kv_cache_bounds=(1024, 8192)):
        
#         if allowed_num_cus is None:
#             allowed_num_cus = [16, 32, 64, 96, 128]
        
#         self.model_name = model_name
#         self.allowed_num_cus = sorted(set(int(v) for v in allowed_num_cus if v % 4 == 0))
        
#         # Generate power-of-2 choices
#         self.BATCH_CHOICES = self._generate_power_of_2_choices(batch_bounds[0], batch_bounds[1])
#         self.KV_CACHE_CHOICES = self._generate_power_of_2_choices(kv_cache_bounds[0], kv_cache_bounds[1])
        
#         # All choice lists for each decision variable
#         self.all_choices = [
#             self.allowed_num_cus,      # 0: num_cus
#             self.TMAC_CHOICES,         # 1: tmacs
#             self.MEM_BUF_CHOICES,      # 2: mem_buf_cap
#             self.MEM_BUF_CHOICES,      # 3: net_buf_cap
#             self.BANK_GROUP_CHOICES,   # 4: bank_groups
#             self.RANK_CHOICES,         # 5: ranks
#             self.FRAC_BANK_CHOICES,    # 6: frac_bank_cap
#             self.BATCH_CHOICES,        # 7: batch_size
#             self.KV_CACHE_CHOICES,     # 8: kv_cache
#         ]
        
#         self.action_dims = [len(choices) for choices in self.all_choices]
#         self.num_decisions = len(self.all_choices)
#         self.num_objectives = 2  # latency, energy
        
#         # Initialize simulator
#         self.sim = PistilSimulator()
        
#         # Output directory
#         self.points_csv_path = output_dir
#         self.points_csv_initialized = False
#         self._initialize_points_csv()
        
#         # For normalization
#         self.max_latency = 1000.0  # ms
#         self.max_energy = 5000.0  # mJ

#     def _initialize_points_csv(self):
#         """
#         Initialize points.csv with header row.
#         Includes ALL metrics from pistil_runner for comprehensive data capture [2].
#         """
#         # Guard: if no output path was provided, skip silently
#         if not self.points_csv_path:
#             return

#         decision_cols = [
#             "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
#             "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
#             "batch_size", "kv_cache"
#         ]
        
#         # Include ALL metrics from pistil_runner for comprehensive data capture [2]
#         metric_cols = [
#             # Core objectives (for RL)
#             "latency_ms", "energy_mJ",
#             # Derived performance metrics
#             "latency_per_token_ms", "energy_per_inference_mJ", "energy_per_token_mJ",
#             "prefill_tokens_per_sec",
#             # Power metrics
#             "average_power_W", "system_power_W",
#             "average_power_mem_W", "average_power_comp_W", "average_power_net_W",
#             # Cost metrics (all cost breakdowns)
#             "system_cost",
#             "chiplet_silicon_cost", "memory_cost_2xHBLC", "package_cost",
#             "package_silicon_cost", "package_memory_cost", "package_substrate_cost",
#             "system_silicon_cost", "system_memory_cost", "system_substrate_cost", "system_pcb_cost",
#             # Utilization metrics
#             "avg_comp_util", "avg_mem_util",
#             # System configuration
#             "system_compute_TOPS", "system_bandwidth_TBps", "system_capacity_GB",
#             "num_packages", "num_chiplets",
#             # Memory/HBLC metrics
#             "hblc_capacity_GB", "hblc_bandwidth_GBps", "hblc_bw_per_capacity",
#             "peak_buffer_MB", "avg_cache_util_MB",
#             # Model capacity metrics
#             "model_weight_capacity_GB", "kv_cache_capacity_per_batch_GB", "total_model_capacity_GB",
#             # Additional system parameters
#             "cores_per_cu", "mem_buffer_size", "net_buffer_size", "tmacs_per_core",
#             "hblc_bank_groups", "hblc_ranks", "hblc_frac_bank_cap"
#         ]
        
#         if not os.path.exists(self.points_csv_path) or os.path.getsize(self.points_csv_path) == 0:
#             with open(self.points_csv_path, "w", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(decision_cols + metric_cols + ["algorithm"])
#             self.points_csv_initialized = True
#             print(f"Initialized points.csv at {self.points_csv_path}")

#     def decode_actions(self, action_indices):
#         """Convert action indices to actual parameter values."""
#         return [self.all_choices[i][action_indices[i]] for i in range(self.num_decisions)]

#     def normalize_design(self, design_values):
#         """Normalize design values to [0, 1] range for network input."""
#         normalized = []
#         for i, val in enumerate(design_values):
#             choices = self.all_choices[i]
#             min_val, max_val = min(choices), max(choices)
#             if max_val > min_val:
#                 norm = (val - min_val) / (max_val - min_val)
#             else:
#                 norm = 0.5
#             normalized.append(norm)
#         return normalized

#     def evaluate(self, action_indices):
#         """
#         Evaluate a design (list of 9 action indices).
#         Returns: (normalized_objectives, raw_objectives)
#         """
#         design_values = self.decode_actions(action_indices)
        
#         params = {
#             "num_cus": int(design_values[0]),
#             "num_tmacs": int(design_values[1]),
#             "mem_buf_cap": float(design_values[2]),
#             "net_buf_cap": float(design_values[3]),
#             "mem_banks_per_group": int(design_values[4]),
#             "mem_ranks": int(design_values[5]),
#             "mem_frac_bank_cap": float(design_values[6]),
#             "model": self.model_name,
#             "batch_size": int(design_values[7]),
#             "kv_cache": int(design_values[8]),
#             # Fixed constants
#             "w_dtype": 0.5,
#             "kv_dtype": 1.0,
#             "prefill": "False",
#             "prefill_chunk_size": 0,
#             "prefill_cached": 0,
#             "sim_num_layers": -1,
#             "lm_head": "False",
#             "plot_exe": "False",
#             "gen_trace": True,
#             "sim_standalone": True,
#             "base_config": "pistil-sys-base.json",
#         }
        
#         try:
#             # Pre-flight check: decode-only mode with prefill_cached=0 and
#             # kv_cache equal to max_seq can trigger seq_len=0 in kv_cache.write_kv.
#             # Force prefill_cached to 1 in this edge case to avoid the crash.
#             if (params.get("prefill") == "False"
#                     and int(params.get("prefill_cached", 0)) == 0
#                     and int(params.get("kv_cache", 0)) > 0):
#                 params = dict(params)  # don't mutate the original
#                 params["prefill_cached"] = 1
#                 print(f"[PISTIL RL] Pre-flight: set prefill_cached=1 to avoid seq_len=0 crash")
#             print(f"[PISTIL RL] Evaluating: {params['num_cus']} CUs, {params['num_tmacs']} TMACs, "
#                   f"batch={params['batch_size']}, kv_cache={params['kv_cache']}")
            
#             self.sim.run_dse_point(params)
#             metrics_dict = self._load_all_metrics(params)
#             latency_ms = metrics_dict["latency_ms"]
#             energy_mJ = metrics_dict["energy_mJ"]
            
#             # Save ALL metrics to CSV [2]
#             self._save_to_points_csv(params, metrics_dict)
            
#             print(f"[PISTIL RL] Results: latency={latency_ms:.2f}ms, energy={energy_mJ:.2f}mJ")
            
#         except Exception as e:
#             print(f"[PISTIL RL] ERROR: {e}")
#             latency_ms, energy_mJ = 1e9, 1e9
        
#         # Update max values
#         self.max_latency = max(self.max_latency, latency_ms)
#         self.max_energy = max(self.max_energy, energy_mJ)
        
#         raw_objectives = [latency_ms, energy_mJ]
#         norm_objectives = [
#             latency_ms / self.max_latency,
#             energy_mJ / self.max_energy
#         ]
        
#         return norm_objectives, raw_objectives

#     def _load_all_metrics(self, params):
#         """
#         Load ALL metrics from the Pistil results CSV and compute derived metrics [2].
#         Returns a comprehensive dictionary with all metrics for plotting/analysis.
#         """
#         results_dir = os.path.join(self.sim.sim_root, self.sim.results_dir)
#         if not os.path.exists(results_dir):
#             raise FileNotFoundError(f"Pistil results directory not found: {results_dir}")

#         csv_files = [
#             os.path.join(results_dir, f)
#             for f in os.listdir(results_dir)
#             if f.endswith(".csv")
#         ]
#         if not csv_files:
#             raise FileNotFoundError(f"No Pistil result CSVs found in: {results_dir}")

#         latest_csv = max(csv_files, key=os.path.getmtime)

#         # Parse the summary CSV which has "Metric,Value" rows
#         metrics = {}
#         with open(latest_csv, "r") as f:
#             reader = csv.reader(f)
#             header = next(reader, None)
#             for row in reader:
#                 if len(row) != 2:
#                     continue
#                 key, value = row
#                 try:
#                     metrics[key] = float(value)
#                 except ValueError:
#                     metrics[key] = value

#         # Derive latency and energy [2]
#         total_latency_s = float(metrics.get("total_latency_s", 0.0))
#         total_energy_J = float(metrics.get("total_energy_J", 0.0))
#         num_layers = float(metrics.get("num_layers", 1.0))
#         sim_num_layers = float(metrics.get("sim_num_layers", -1.0))
#         batch_size = float(metrics.get("batch_size", params["batch_size"]))

#         # Scale if simulation only ran subset of layers
#         if sim_num_layers != num_layers and sim_num_layers != -1:
#             latency_ms = total_latency_s * num_layers / sim_num_layers * 1000.0
#             energy_mJ = total_energy_J * num_layers / sim_num_layers * 1000.0
#         else:
#             latency_ms = total_latency_s * 1000.0
#             energy_mJ = total_energy_J * 1000.0

#         # Compute derived metrics [2]
#         latency_per_token_ms = latency_ms / batch_size if batch_size > 0 else 0.0
#         energy_per_inference_mJ = energy_mJ
#         energy_per_token_mJ = energy_mJ / batch_size if batch_size > 0 else 0.0
        
#         num_cus = float(metrics.get("num_chiplets", params["num_cus"]))
#         avg_power_W = float(metrics.get("average_power_W", 0.0))
#         system_power_W = num_cus * avg_power_W
        
#         system_cost = float(metrics.get("system_cost", 0.0))
        
#         prefill_batch = metrics.get("prefill_batch", False)
#         prefill_chunk_size = float(metrics.get("prefill_chunk_size", 0.0))
#         if prefill_batch and prefill_chunk_size > 0 and latency_ms > 0:
#             prefill_tokens_per_sec = (prefill_chunk_size * batch_size) / (latency_ms / 1000.0)
#         else:
#             prefill_tokens_per_sec = batch_size / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        
#         avg_comp_util = float(metrics.get("avg_comp_util", 0.0))
#         avg_mem_util = float(metrics.get("avg_mem_util", 0.0))
        
#         # Build comprehensive metrics dictionary [2]
#         all_metrics = dict(metrics)
#         all_metrics.update({
#             "latency_ms": float(latency_ms),
#             "energy_mJ": float(energy_mJ),
#             "latency_per_token_ms": latency_per_token_ms,
#             "energy_per_inference_mJ": energy_per_inference_mJ,
#             "energy_per_token_mJ": energy_per_token_mJ,
#             "prefill_tokens_per_sec": prefill_tokens_per_sec,
#             "average_power_W": avg_power_W,
#             "system_power_W": system_power_W,
#             "average_power_mem_W": float(metrics.get("average_power_mem_W", 0.0)),
#             "average_power_comp_W": float(metrics.get("average_power_comp_W", 0.0)),
#             "average_power_net_W": float(metrics.get("average_power_net_W", 0.0)),
#             "system_cost": system_cost,
#             "chiplet_silicon_cost": float(metrics.get("chiplet_silicon_cost", 0.0)),
#             "memory_cost_2xHBLC": float(metrics.get("memory_cost_2xHBLC", 0.0)),
#             "package_cost": float(metrics.get("package_cost", 0.0)),
#             "package_silicon_cost": float(metrics.get("package_silicon_cost", 0.0)),
#             "package_memory_cost": float(metrics.get("package_memory_cost", 0.0)),
#             "package_substrate_cost": float(metrics.get("package_substrate_cost", 0.0)),
#             "system_silicon_cost": float(metrics.get("system_silicon_cost", 0.0)),
#             "system_memory_cost": float(metrics.get("system_memory_cost", 0.0)),
#             "system_substrate_cost": float(metrics.get("system_substrate_cost", 0.0)),
#             "system_pcb_cost": float(metrics.get("system_pcb_cost", 0.0)),
#             "avg_comp_util": avg_comp_util,
#             "avg_mem_util": avg_mem_util,
#             "num_chiplets": num_cus,
#             "system_compute_TOPS": float(metrics.get("system_compute_TOPS", 0.0)),
#             "system_bandwidth_TBps": float(metrics.get("system_bandwidth_TBps", 0.0)),
#             "system_capacity_GB": float(metrics.get("system_capacity_GB", 0.0)),
#             "num_packages": float(metrics.get("num_packages", 0.0)),
#             "hblc_capacity_GB": float(metrics.get("hblc_capacity_GB", 0.0)),
#             "hblc_bandwidth_GBps": float(metrics.get("hblc_bandwidth_GBps", 0.0)),
#             "hblc_bw_per_capacity": float(metrics.get("hblc_bw_per_capacity", 0.0)),
#             "peak_buffer_MB": float(metrics.get("peak_buffer_MB", 0.0)),
#             "avg_cache_util_MB": float(metrics.get("avg_cache_util_MB", 0.0)),
#             "model_weight_capacity_GB": float(metrics.get("model_weight_capacity_GB", 0.0)),
#             "kv_cache_capacity_per_batch_GB": float(metrics.get("kv_cache_capacity_per_batch_GB", 0.0)),
#             "total_model_capacity_GB": float(metrics.get("total_model_capacity_GB", 0.0)),
#         })
        
#         return all_metrics

#     def _save_to_points_csv(self, params, metrics_dict):
#         """
#         Save design point with all metrics to points.csv [2].
#         Includes decision variables + ALL performance/cost metrics from pistil_runner.
#         """
#         # Define CSV columns: decision variables first, then metrics
#         decision_cols = [
#             "num_cus", "num_tmacs", "mem_buf_cap", "net_buf_cap",
#             "mem_banks_per_group", "mem_ranks", "mem_frac_bank_cap",
#             "batch_size", "kv_cache"
#         ]
        
#         # Include ALL metrics from pistil_runner for comprehensive data capture [2]
#         metric_cols = [
#             # Core objectives (for RL)
#             "latency_ms", "energy_mJ",
#             # Derived performance metrics
#             "latency_per_token_ms", "energy_per_inference_mJ", "energy_per_token_mJ",
#             "prefill_tokens_per_sec",
#             # Power metrics
#             "average_power_W", "system_power_W",
#             "average_power_mem_W", "average_power_comp_W", "average_power_net_W",
#             # Cost metrics (all cost breakdowns)
#             "system_cost",
#             "chiplet_silicon_cost", "memory_cost_2xHBLC", "package_cost",
#             "package_silicon_cost", "package_memory_cost", "package_substrate_cost",
#             "system_silicon_cost", "system_memory_cost", "system_substrate_cost", "system_pcb_cost",
#             # Utilization metrics
#             "avg_comp_util", "avg_mem_util",
#             # System configuration
#             "system_compute_TOPS", "system_bandwidth_TBps", "system_capacity_GB",
#             "num_packages", "num_chiplets",
#             # Memory/HBLC metrics
#             "hblc_capacity_GB", "hblc_bandwidth_GBps", "hblc_bw_per_capacity",
#             "peak_buffer_MB", "avg_cache_util_MB",
#             # Model capacity metrics
#             "model_weight_capacity_GB", "kv_cache_capacity_per_batch_GB", "total_model_capacity_GB",
#             # Additional system parameters
#             "cores_per_cu", "mem_buffer_size", "net_buffer_size", "tmacs_per_core",
#             "hblc_bank_groups", "hblc_ranks", "hblc_frac_bank_cap"
#         ]
        
#         # Ensure header exists (should already be initialized, but check just in case)
#         if not self.points_csv_initialized:
#             self._initialize_points_csv()
        
#         # Prepare row data - decision variables first
#         row_data = [
#             params["num_cus"],
#             params["num_tmacs"],
#             params["mem_buf_cap"],
#             params["net_buf_cap"],
#             params["mem_banks_per_group"],
#             params["mem_ranks"],
#             params["mem_frac_bank_cap"],
#             params["batch_size"],
#             params["kv_cache"],
#         ]
        
#         # Add all metrics in the same order as metric_cols [2]
#         for col in metric_cols:
#             row_data.append(metrics_dict.get(col, 0.0))
#         row_data.append(getattr(self, 'algorithm_label', 'Deep RL'))
        
#         # Append row to CSV
#         try:
#             with open(self.points_csv_path, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(row_data)
#             print(f"[PISTIL RL] Saved point to CSV: {params['num_cus']} CUs, "
#                 f"latency={metrics_dict.get('latency_ms', 0):.2f}ms, "
#                 f"energy={metrics_dict.get('energy_mJ', 0):.2f}mJ")
#             print(f"  Per-token: latency={metrics_dict.get('latency_per_token_ms', 0):.3f}ms, "
#                 f"energy={metrics_dict.get('energy_per_token_mJ', 0):.3f}mJ")
#         except Exception as e:
#             print(f"[PISTIL RL] Error saving to points.csv: {e}")
#             raise


# def run_ppo_pistil(params):
#     """
#     Run PPO optimization for Pistil chiplet design.
#     Based on the approach in rlCascade.py [1].
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # Initialize evaluator and models
#     evaluator = PistilEvaluator(
#         model_name=params.get('model_name', 'llama3-8b'),
#         output_dir=params.get('output_dir', None),
#         allowed_num_cus=params.get('allowed_num_cus', None),
#         batch_bounds=params.get('batch_bounds', (1, 64)),
#         kv_cache_bounds=params.get('kv_cache_bounds', (1024, 8192))
#     )
#     evaluator.algorithm_label = 'Deep RL'
    
#     num_decisions = evaluator.num_decisions  # 9 decision variables
#     num_objectives = evaluator.num_objectives  # 2 (latency, energy)
#     action_dims = evaluator.action_dims  # List of choices per decision
    
#     actor = MLPActor(device, params, num_decisions=num_decisions, action_dims=action_dims).to(device)
#     critic = MLPCritic(device, params, num_decisions=num_decisions, num_objectives=num_objectives).to(device)
    
#     # Training parameters
#     epochs = params['num_epochs']
#     mini_batch_size = params['mini_batch_size']
#     gamma = params['gamma']
#     lam = params['lambda']
    
#     all_designs = []
#     all_objectives = []
    
#     for epoch in range(epochs):
#         print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
#         # Storage for this epoch
#         batch_rewards = [[] for _ in range(mini_batch_size)]
#         batch_actions = [[] for _ in range(mini_batch_size)]
#         batch_logprobs = [[] for _ in range(mini_batch_size)]
#         batch_observations = [[] for _ in range(mini_batch_size)]
#         batch_designs = [[] for _ in range(mini_batch_size)]
#         batch_decision_indices = [[] for _ in range(mini_batch_size)]
        
#         # Generate random weights for multi-objective scalarization [1]
#         weights = np.random.rand(mini_batch_size, num_objectives)
#         weights = weights / weights.sum(axis=1, keepdims=True)
        
#         # Initialize observations with weights
#         for idx in range(mini_batch_size):
#             batch_observations[idx] = weights[idx].tolist()
        
#         # --- 1. Sample Actions (build designs sequentially) ---
#         with torch.no_grad():
#             for decision_step in range(num_decisions):
#                 # Prepare observations: weights + current partial design (padded)
#                 obs_for_actor = []
#                 for idx in range(mini_batch_size):
#                     obs = batch_observations[idx].copy()
#                     # Pad to fixed length: weights (2) + design (9 decisions normalized)
#                     while len(obs) < num_objectives + num_decisions:
#                         obs.append(0.0)
#                     obs_for_actor.append(obs)
                
#                 # Sample actions for this decision variable
#                 log_probs_list = []
#                 actions_list = []
#                 for idx in range(mini_batch_size):
#                     log_prob, action = actor.sample_action([obs_for_actor[idx]], decision_step)
#                     log_probs_list.append(log_prob.item())
#                     actions_list.append(action.item())
                
#                 for idx in range(mini_batch_size):
#                     batch_actions[idx].append(actions_list[idx])
#                     batch_logprobs[idx].append(log_probs_list[idx])
#                     batch_decision_indices[idx].append(decision_step)
                    
#                     # Normalize the action for observation
#                     action_val = evaluator.all_choices[decision_step][actions_list[idx]]
#                     choices = evaluator.all_choices[decision_step]
#                     norm_val = (action_val - min(choices)) / (max(choices) - min(choices) + 1e-8)
#                     batch_observations[idx].append(norm_val)
                    
#                     batch_rewards[idx].append(0.0)  # Intermediate reward = 0
#                     batch_designs[idx].append(int(actions_list[idx]))
        
#         # --- 2. Evaluate Designs and Calculate Final Rewards ---
#         epoch_objectives = []
#         for idx in range(mini_batch_size):
#             design = batch_designs[idx]
#             norm_obj, raw_obj = evaluator.evaluate(design)
            
#             all_designs.append(design)
#             all_objectives.append(raw_obj)
#             epoch_objectives.append(raw_obj)
            
#             # Reward: negative weighted sum (minimizing both objectives) [1]
#             reward = -np.dot(weights[idx], norm_obj)
#             batch_rewards[idx][-1] = reward
        
#         # --- 3. Compute Critic Values ---
#         critic_values = []
#         with torch.no_grad():
#             for decision_step in range(num_decisions):
#                 obs_for_critic = []
#                 for idx in range(mini_batch_size):
#                     obs = batch_observations[idx][:decision_step + 1 + num_objectives]
#                     # Pad to fixed length
#                     while len(obs) < num_objectives + num_decisions:
#                         obs.append(0.0)
#                     obs_for_critic.append(obs)
                
#                 crit_vals = critic.sample_critic(obs_for_critic).cpu().numpy()
#                 # Weighted sum of critic outputs [1]
#                 weighted_vals = np.sum(weights * crit_vals, axis=1)
#                 critic_values.append(weighted_vals)
        
#         # Reshape critic values per sample
#         values = [[] for _ in range(mini_batch_size)]
#         for step_vals in critic_values:
#             for idx, val in enumerate(step_vals):
#                 values[idx].append(val)
#         for idx in range(mini_batch_size):
#             values[idx].append(values[idx][-1])  # Bootstrap
        
#         # --- 4. Compute Advantages (GAE) [1] ---
#         all_advantages = []
#         all_returns = []
#         for idx in range(mini_batch_size):
#             rewards = np.array(batch_rewards[idx])
#             vals = np.array(values[idx])
            
#             deltas = rewards + gamma * vals[1:] - vals[:-1]
#             advantages = discounted_cumulative_sums(deltas, gamma * lam)
#             returns = discounted_cumulative_sums(rewards, gamma * lam)
            
#             all_advantages.append(advantages)
#             all_returns.append(returns)
        
#         # Normalize advantages [1]
#         adv_flat = np.concatenate(all_advantages)
#         adv_mean, adv_std = np.mean(adv_flat), np.std(adv_flat)
#         all_advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in all_advantages]
        
#         # --- 5. Prepare Training Tensors ---
#         obs_tensor, act_tensor, logp_tensor = [], [], []
#         adv_tensor, ret_tensor, weight_tensor, dec_idx_tensor = [], [], [], []
        
#         for idx in range(mini_batch_size):
#             for step in range(num_decisions):
#                 obs = batch_observations[idx][:step + 1 + num_objectives]
#                 obs = obs + [0.0] * (num_objectives + num_decisions - len(obs))
                
#                 obs_tensor.append(obs)
#                 act_tensor.append(batch_actions[idx][step])
#                 logp_tensor.append(batch_logprobs[idx][step])
#                 adv_tensor.append(all_advantages[idx][step])
#                 ret_tensor.append(all_returns[idx][step])
#                 weight_tensor.append(weights[idx])
#                 dec_idx_tensor.append(batch_decision_indices[idx][step])
        
#         obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32).to(device)
#         act_tensor = torch.tensor(act_tensor, dtype=torch.long).to(device)
#         logp_tensor = torch.tensor(logp_tensor, dtype=torch.float32).to(device)
#         adv_tensor = torch.tensor(adv_tensor, dtype=torch.float32).to(device)
#         ret_tensor = torch.tensor(ret_tensor, dtype=torch.float32).to(device)
#         weight_tensor = torch.tensor(weight_tensor, dtype=torch.float32).to(device)
        
#         # --- 6. PPO Updates [1] ---
#         for _ in range(params['update_iterations']):
#             actor_loss, kl = actor.ppo_update(
#                 obs_tensor, act_tensor, logp_tensor, adv_tensor, dec_idx_tensor
#             )
#             if kl > params['target_kl']:
#                 break
        
#         for _ in range(params['update_iterations']):
#             critic_loss = critic.ppo_update(obs_tensor, ret_tensor, weight_tensor)
        
#         # Logging
#         avg_latency = np.mean([o[0] for o in epoch_objectives])
#         avg_energy = np.mean([o[1] for o in epoch_objectives])
        
#         if epoch % 5 == 0 or epoch == epochs - 1:
#             print(f"Epoch {epoch+1}/{epochs} | "
#                   f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | "
#                   f"Avg Latency: {avg_latency:.2f}ms | Avg Energy: {avg_energy:.2f}mJ")
    
#     return np.array(all_designs), np.array(all_objectives)


# # ============== Entry Point ==============
# def runPPOPistil(
#     num_epochs=50,
#     mini_batch_size=8,
#     model_name="llama3-8b",
#     output_dir=None,
#     allowed_num_cus=None,
#     batch_bounds=(1, 64),
#     kv_cache_bounds=(1024, 8192)
# ):
#     """
#     Main entry point for running PPO on Pistil.
#     Similar interface to runGAPistil in gaPistil.py [2].
#     """
#     params = {
#         'num_epochs': num_epochs,
#         'mini_batch_size': mini_batch_size,
#         'learning_rate': 1e-3,
#         'gamma': 0.99,
#         'lambda': 0.95,
#         'clip_ratio': 0.2,
#         'target_kl': 0.01,
#         'update_iterations': 5,
#         'model_name': model_name,
#         'output_dir': output_dir,
#         'allowed_num_cus': allowed_num_cus,
#         'batch_bounds': batch_bounds,
#         'kv_cache_bounds': kv_cache_bounds
#     }
    
#     print(f"=" * 80)
#     print(f"Running PPO Optimization for Pistil")
#     print(f"  Model: {model_name}")
#     print(f"  Epochs: {num_epochs}, Batch Size: {mini_batch_size}")
#     print(f"  Batch Bounds: {batch_bounds}, KV Cache Bounds: {kv_cache_bounds}")
#     print(f"=" * 80)
    
#     all_designs, all_objectives = run_ppo_pistil(params)
    
#     print(f"\n" + "=" * 80)
#     print(f"Optimization Complete!")
#     print(f"  Total designs evaluated: {len(all_designs)}")
#     print(f"  Best Latency: {np.min(all_objectives[:, 0]):.2f}ms")
#     print(f"  Best Energy: {np.min(all_objectives[:, 1]):.2f}mJ")
#     print(f"=" * 80)

#     # Build design points similar to rlCascade.py format [1]
#     design_points = []
#     evaluator = PistilEvaluator(model_name=model_name, allowed_num_cus=allowed_num_cus, output_dir=output_dir)
    
#     for i, (design, obj) in enumerate(zip(all_designs, all_objectives)):
#         decoded_values = evaluator.decode_actions(design)
#         dp = {
#             'latency_ms': obj[0],
#             'energy_mJ': obj[1],
#             'design': {
#                 'num_cus': int(decoded_values[0]),
#                 'num_tmacs': int(decoded_values[1]),
#                 'mem_buf_cap': float(decoded_values[2]),
#                 'net_buf_cap': float(decoded_values[3]),
#                 'bank_groups': int(decoded_values[4]),
#                 'ranks': int(decoded_values[5]),
#                 'frac_bank_cap': float(decoded_values[6]),
#                 'batch_size': int(decoded_values[7]),
#                 'kv_cache': int(decoded_values[8]),
#             },
#             'additional_metrics': {'episode': i},
#         }
#         design_points.append(dp)
    
#     return np.array(all_designs), np.array(all_objectives), design_points


# # if __name__ == "__main__":
# #     # Example usage
# #     designs, objectives, design_points = runPPOPistil(
# #         num_epochs=5,
# #         mini_batch_size=4,
# #         model_name="llama3-8b"
# #     )