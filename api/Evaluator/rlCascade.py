import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import scipy.signal
import sys
import os
import json
from copy import deepcopy

# Import Cascade evaluation components
from api.Evaluator.cascade.chiplet_model.dse.lib.chiplet_system import ChipletSystem
from api.Evaluator.cascade.chiplet_model.dse.lib.trace_parser import TraceParser


# ============== MLP Actor ==============
class MLPActor(nn.Module):
    def __init__(self, device, params, num_actions=12, action_choices=4):
        """
        Actor network for Cascade: outputs probability distribution over 4 chiplet types
        for each of the 12 positions.
        """
        super(MLPActor, self).__init__()
        self.device = device
        self.num_actions = num_actions  # 12 chiplet positions
        self.action_choices = action_choices  # 4 chiplet types (gpu, atten, sparse, conv)
        self.clip_ratio = params['clip_ratio']
        
        hidden_dim = 16
        # Input: current partial design (flattened) + weight vector
        input_dim = num_actions + 2  # 12 positions + 2 objective weights
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_choices)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        with autocast(device_type=self.device.type, dtype=torch.float16):
            logits = self.net(x)
            probs = F.softmax(logits, dim=-1)
        return probs

    def sample_action(self, observations):
        """Sample actions given observations."""
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        probs = self.forward(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return log_probs, actions

    def ppo_update(self, observations, actions, old_logprobs, advantages):
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            probs = self.forward(observations)
            dist = torch.distributions.Categorical(probs)
            new_logprobs = dist.log_prob(actions)
            
            ratio = torch.exp(new_logprobs - old_logprobs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = -torch.mean(torch.min(ratio * advantages, clipped_ratio * advantages))
        
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        kl = torch.mean(old_logprobs - new_logprobs).item()
        return policy_loss.item(), kl


# ============== MLP Critic ==============
class MLPCritic(nn.Module):
    def __init__(self, device, params, num_actions=12, num_objectives=2):
        """
        Critic network: predicts value for each objective dimension.
        """
        super(MLPCritic, self).__init__()
        self.device = device
        self.num_objectives = num_objectives
        
        hidden_dim = 64
        input_dim = num_actions + num_objectives  # design + weights
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_objectives)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=params['learning_rate'])
        self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        with autocast(device_type=self.device.type, dtype=torch.float16):
            return self.net(x)

    def sample_critic(self, observations):
        obs_tensor = torch.tensor(observations, dtype=torch.float32).to(self.device)
        return self.forward(obs_tensor)

    def ppo_update(self, observations, returns, weights):
        self.optimizer.zero_grad()
        with autocast(device_type=self.device.type, dtype=torch.float16):
            pred_values = self.forward(observations)
            # Weighted sum of predicted values
            pred_reward = torch.sum(pred_values * weights, dim=-1)
            value_loss = torch.mean((pred_reward - returns) ** 2)
        
        self.scaler.scale(value_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return value_loss.item()


# ============== Cascade Evaluator Wrapper ==============
class CascadeEvaluator:
    def __init__(self, trace="gpt-j-65536-weighted"):
        WORKSPACE = sys.path[0] + '/api/Evaluator/cascade/chiplet_model'
        self.TRACE_DIR = WORKSPACE + '/traces'
        self.CHIPLET_LIBRARY = WORKSPACE + '/dse/chiplet-library'
        self.EXPERIMENT_DIR = WORKSPACE + '/dse/experiments/' + trace + '.json'
        self.OUTPUT_DIR = WORKSPACE + '/dse/results'
        self.tp = TraceParser(self.TRACE_DIR, self.EXPERIMENT_DIR)
        
        # Design space parameters
        self.num_positions = 12
        self.num_chiplet_types = 4  # 0=gpu, 1=atten, 2=sparse, 3=conv
        self.num_objectives = 2  # latency, energy
        
        # For normalization (estimated max values)
        self.max_latency = 100.0  # ms
        self.max_energy = 500.0   # mJ

    def evaluate(self, design):
        """
        Evaluate a design (list of 12 integers 0-3).
        Returns: (normalized_objectives, raw_objectives)
        """
        from collections import Counter
        numChips = Counter(design)
        desired_chiplets = (
            ["gpu"] * numChips[0] + 
            ["atten"] * numChips[1] + 
            ["sparse"] * numChips[2] + 
            ["conv"] * numChips[3]
        )
        
        agg_kernel_results = []
        num_channels = 16
        
        for TRACE_ID in range(len(self.tp.all_traces)):
            cs = ChipletSystem(self.CHIPLET_LIBRARY, verbose=0)
            cs.configure_system(desired_chiplets, self.tp.optimization_goal)
            cs.init_system_bandwidth(bw_per_channel=64, num_channels=num_channels)
            
            cut_dim = "batch" if self.tp.all_traces[TRACE_ID].get_model() == "dnn" else "weights"
            kernel_results = cs.characterize_workload(
                self.tp.get_trace(TRACE_ID), cut_dim=cut_dim, dtype=2
            )
            agg_kernel_results += kernel_results * self.tp.all_traces[TRACE_ID].weighted_score
        
        # Calculate totals
        total_exe = sum(k["total"]["exe_time"] for k in agg_kernel_results) * 1000  # ms
        total_energy = sum(k["total"]["energy"] for k in agg_kernel_results) * 1e3  # mJ
        
        # Update max values if needed
        self.max_latency = max(self.max_latency, total_exe)
        self.max_energy = max(self.max_energy, total_energy)
        
        raw_objectives = [total_exe, total_energy]
        norm_objectives = [
            total_exe / self.max_latency,
            total_energy / self.max_energy
        ]

        result_file = self.OUTPUT_DIR + "/points.csv"
        entry = f"{total_exe},{total_energy},{numChips[0]},{numChips[1]},{numChips[2]},{numChips[3]}\n"
        # Check if entry already exists
        exists = False
        try:
            with open(result_file, "r") as f:
                for line in f:
                    if line.strip() == entry.strip():
                        exists = True
                        break
        except FileNotFoundError:
            pass  # File does not exist yet

        if not exists:
            with open(result_file, "a") as f:
                f.write(entry)
                print(f"Summary saved to {result_file}")

            context_file = self.OUTPUT_DIR + "/pointContext/" + f"{numChips[0]}gpu{numChips[1]}attn{numChips[2]}sparse{numChips[3]}conv.json"
            os.makedirs(os.path.dirname(context_file), exist_ok=True)
            with open(context_file, "w") as f:
                jsonData = []
                for ind, result in enumerate(agg_kernel_results):
                    resultCopy = deepcopy(result)
                    resultCopy["kernal_number"] = ind
                    jsonData.append(resultCopy)
                json.dump(jsonData, f, indent=4)
            print(f"Results saved to {context_file}")

        else:
            print(f"Entry already exists in {result_file}, not writing duplicate.")
        
        return norm_objectives, raw_objectives


# ============== Helper Functions ==============
def discounted_cumulative_sums(x, discount):
    """Compute discounted cumulative sums (for GAE)."""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# ============== Main PPO Training Loop ==============
def run_ppo_cascade(params):
    """
    Run PPO optimization for Cascade chiplet design.
    Based on the approach in ppo_rand_weights_state.py [1].
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize evaluator and models
    evaluator = CascadeEvaluator(trace=params.get('trace', 'gpt-j-65536-weighted'))
    num_actions = evaluator.num_positions  # 12
    num_objectives = evaluator.num_objectives  # 2
    
    actor = MLPActor(device, params, num_actions=num_actions).to(device)
    critic = MLPCritic(device, params, num_actions=num_actions, num_objectives=num_objectives).to(device)
    
    # Training parameters
    epochs = params['num_epochs']
    mini_batch_size = params['mini_batch_size']
    gamma = params['gamma']
    lam = params['lambda']
    
    all_designs = []
    all_objectives = []
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        # Storage for this epoch
        batch_rewards = [[] for _ in range(mini_batch_size)]
        batch_actions = [[] for _ in range(mini_batch_size)]
        batch_logprobs = [[] for _ in range(mini_batch_size)]
        batch_observations = [[] for _ in range(mini_batch_size)]
        batch_designs = [[] for _ in range(mini_batch_size)]
        
        # Generate random weights for multi-objective scalarization [1]
        weights = np.random.rand(mini_batch_size, num_objectives)
        weights = weights / weights.sum(axis=1, keepdims=True)
        
        # Initialize observations with weights
        for idx in range(mini_batch_size):
            batch_observations[idx] = weights[idx].tolist()
        
        # --- 1. Sample Actions (build designs sequentially) ---
        with torch.no_grad():
            for action_step in range(num_actions):
                # Prepare observations: weights + current partial design (padded)
                obs_for_actor = []
                for idx in range(mini_batch_size):
                    obs = batch_observations[idx].copy()
                    # Pad to fixed length: weights (2) + design (12)
                    while len(obs) < num_objectives + num_actions:
                        obs.append(0.0)
                    obs_for_actor.append(obs)
                
                log_probs, actions = actor.sample_action(obs_for_actor)
                log_probs = log_probs.tolist()
                actions = actions.tolist()
                
                for idx in range(mini_batch_size):
                    batch_actions[idx].append(actions[idx])
                    batch_logprobs[idx].append(log_probs[idx])
                    batch_observations[idx].append(float(actions[idx]))
                    batch_rewards[idx].append(0.0)  # Intermediate reward = 0
                    batch_designs[idx].append(int(actions[idx]))
        
        # --- 2. Evaluate Designs and Calculate Final Rewards ---
        epoch_objectives = []
        for idx in range(mini_batch_size):
            design = batch_designs[idx]
            norm_obj, raw_obj = evaluator.evaluate(design)
            
            all_designs.append(design)
            all_objectives.append(raw_obj)
            epoch_objectives.append(raw_obj)
            
            # Reward: negative weighted sum (minimizing both objectives) [1]
            reward = -np.dot(weights[idx], norm_obj)
            batch_rewards[idx][-1] = reward
        
        # --- 3. Compute Critic Values ---
        critic_values = []
        with torch.no_grad():
            for action_step in range(num_actions):
                obs_for_critic = []
                for idx in range(mini_batch_size):
                    obs = batch_observations[idx][:action_step + 1 + num_objectives]
                    # Pad to fixed length
                    while len(obs) < num_objectives + num_actions:
                        obs.append(0.0)
                    obs_for_critic.append(obs)
                
                crit_vals = critic.sample_critic(obs_for_critic).cpu().numpy()
                # Weighted sum of critic outputs
                weighted_vals = np.sum(weights * crit_vals, axis=1)
                critic_values.append(weighted_vals)
        
        # Reshape critic values per sample
        values = [[] for _ in range(mini_batch_size)]
        for step_vals in critic_values:
            for idx, val in enumerate(step_vals):
                values[idx].append(val)
        for idx in range(mini_batch_size):
            values[idx].append(values[idx][-1])  # Bootstrap
        
        # --- 4. Compute Advantages (GAE) [1] ---
        all_advantages = []
        all_returns = []
        for idx in range(mini_batch_size):
            rewards = np.array(batch_rewards[idx])
            vals = np.array(values[idx])
            
            deltas = rewards + gamma * vals[1:] - vals[:-1]
            advantages = discounted_cumulative_sums(deltas, gamma * lam)
            returns = discounted_cumulative_sums(rewards, gamma * lam)
            
            all_advantages.append(advantages)
            all_returns.append(returns)
        
        # Normalize advantages
        adv_flat = np.concatenate(all_advantages)
        adv_mean, adv_std = np.mean(adv_flat), np.std(adv_flat)
        all_advantages = [(a - adv_mean) / (adv_std + 1e-8) for a in all_advantages]
        
        # --- 5. Prepare Training Tensors ---
        obs_tensor, act_tensor, logp_tensor = [], [], []
        adv_tensor, ret_tensor, weight_tensor = [], [], []
        
        for idx in range(mini_batch_size):
            for step in range(num_actions):
                obs = batch_observations[idx][:step + 1 + num_objectives]
                obs = obs + [0.0] * (num_objectives + num_actions - len(obs))
                
                obs_tensor.append(obs)
                act_tensor.append(batch_actions[idx][step])
                logp_tensor.append(batch_logprobs[idx][step])
                adv_tensor.append(all_advantages[idx][step])
                ret_tensor.append(all_returns[idx][step])
                weight_tensor.append(weights[idx])
        
        obs_tensor = torch.tensor(obs_tensor, dtype=torch.float32).to(device)
        act_tensor = torch.tensor(act_tensor, dtype=torch.long).to(device)
        logp_tensor = torch.tensor(logp_tensor, dtype=torch.float32).to(device)
        adv_tensor = torch.tensor(adv_tensor, dtype=torch.float32).to(device)
        ret_tensor = torch.tensor(ret_tensor, dtype=torch.float32).to(device)
        weight_tensor = torch.tensor(weight_tensor, dtype=torch.float32).to(device)
        
        # --- 6. PPO Updates [1] ---
        for _ in range(params['update_iterations']):
            actor_loss, kl = actor.ppo_update(obs_tensor, act_tensor, logp_tensor, adv_tensor)
            if kl > params['target_kl']:
                break
        
        for _ in range(params['update_iterations']):
            critic_loss = critic.ppo_update(obs_tensor, ret_tensor, weight_tensor)
        
        # Logging
        avg_latency = np.mean([o[0] for o in epoch_objectives])
        avg_energy = np.mean([o[1] for o in epoch_objectives])
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Actor Loss: {actor_loss:.4f} | Critic Loss: {critic_loss:.4f} | "
                  f"Avg Latency: {avg_latency:.2f}ms | Avg Energy: {avg_energy:.2f}mJ")
    
    return np.array(all_designs), np.array(all_objectives)


# ============== Entry Point ==============
def runPPOCascade(num_epochs=50, mini_batch_size=8, trace="gpt-j-65536-weighted"):
    """
    Main entry point for running PPO on Cascade.
    Similar interface to runGACascade in gaCascade.py [3].
    """
    params = {
        'num_epochs': num_epochs,
        'mini_batch_size': mini_batch_size,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'lambda': 0.95,
        'clip_ratio': 0.2,
        'target_kl': 0.01,
        'update_iterations': 5,
        'trace': trace
    }
    
    print(f"Running PPO Optimization for Cascade")
    print(f"Trace: {trace}")
    print(f"Epochs: {num_epochs}, Batch Size: {mini_batch_size}")
    
    all_designs, all_objectives = run_ppo_cascade(params)
    
    print(f"\nOptimization Complete!")
    print(f"Total designs evaluated: {len(all_designs)}")
    print(f"Best Latency: {np.min(all_objectives[:, 0]):.2f}ms")
    print(f"Best Energy: {np.min(all_objectives[:, 1]):.2f}mJ")

    design_points = []
    for i, (design, obj) in enumerate(zip(all_designs, all_objectives)):
        from collections import Counter
        numChips = Counter(design)
        dp = {
            'execution_time_ms': obj[0],
            'energy_mj': obj[1],
            'chiplets': {
                'GPU': int(numChips[0]),
                'Attention': int(numChips[1]),
                'Sparse': int(numChips[2]),
                'Convolution': int(numChips[3])
            },
            'additional_metrics': {'episode': i},
            'context_file_path': ''
        }
        design_points.append(dp)
    
    return np.array(all_designs), np.array(all_objectives), design_points


# if __name__ == "__main__":
#     # Example usage
#     designs, objectives = runPPOCascade(
#         num_epochs=20,
#         mini_batch_size=4,
#         model_name="llama3-8b"
#     )