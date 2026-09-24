import numpy as np
from scipy.stats import lognorm
from collections import Counter, defaultdict

def discrete_lognormal_pmf(max_items=128, sigma=1.0):
    mean = 3.0
    x = np.arange(max_items)
    pdf = lognorm.pdf(x + 1e-6, s=sigma, scale=np.exp(mean))  # shift to avoid 0
    pmf = pdf / pdf.sum()
    return pmf

def sample_discrete_lognormal(sample_count=1024, max_items=128, seed=None):
    if seed is not None:
        np.random.seed(seed)
    pmf = discrete_lognormal_pmf(max_items)
    samples = np.random.choice(np.arange(max_items), size=sample_count, p=pmf)
    return samples

def pad_and_average(distributions):
    max_len = max(len(d) for d in distributions)
    padded = [d + [0] * (max_len - len(d)) for d in distributions]
    return np.mean(padded, axis=0)

def rounded_distribution(avg_dist, target_sum):
    """Round values so they sum to target_sum (i.e., batch size)."""
    floored = np.floor(avg_dist).astype(int)
    remainder = target_sum - np.sum(floored)
    
    # Compute fractional parts
    fractions = avg_dist - floored
    # Get indices of largest fractional parts
    increment_indices = np.argsort(fractions)[-remainder:] if remainder > 0 else []

    for idx in increment_indices:
        floored[idx] += 1
    return floored

import matplotlib.pyplot as plt

class MoEDist:
    def __init__(self, sim_batch_size, num_experts):
        self.batch_size = sim_batch_size
        self.num_experts = num_experts

        group_by_unique_count = defaultdict(list)
        frequency_counter = Counter()

        NUM_TIMES = 10000
        for _ in range(NUM_TIMES):
            all_samples = sample_discrete_lognormal(
                sample_count=self.batch_size,
                max_items=self.num_experts
            )
            counter = Counter(all_samples)
            dist = sorted(counter.values(), reverse=True)
            num_unique = len(dist)
            group_by_unique_count[num_unique].append(dist)
            frequency_counter[num_unique] += 1

        # print(f"\n=== Batch size: {batch_size} ===")
        # print("Unique experts frequency:")
        for unique_count in sorted(frequency_counter.keys()):
            freq = frequency_counter[unique_count] / NUM_TIMES
            # print(f"  {unique_count} unique: {freq:.2%}")

        # print("Average (rounded) distributions:")
        most_common_unique = max(frequency_counter, key=frequency_counter.get)     
        avg_dist = pad_and_average(group_by_unique_count[most_common_unique])
        rounded_dist = rounded_distribution(avg_dist, target_sum=self.batch_size)
        rounded_list = rounded_dist.tolist()
        # print(f"  Unique experts = {most_common_unique}: {rounded_list} (sum = {sum(rounded_dist)})")

        self.avg_experts_routed = most_common_unique
        self.avg_routed_distribution = rounded_list

    def plot_dist():
        TOTAL_SAMPLES = 10000
        NUM_EXPERTS = self.num_experts

        unique_indices, all_samples = sample_discrete_lognormal(
                                                sample_count    = TOTAL_SAMPLES, 
                                                max_items       = NUM_EXPERTS
                                            )
                                            
        # Histogram (before deduplication)
        counts, bins = np.histogram(all_samples, bins=NUM_EXPERTS, range=(0, NUM_EXPERTS))

        # Normalize so average bar height is 1.0
        normalized_counts = counts / (TOTAL_SAMPLES / NUM_EXPERTS)

        # Sort bars by frequency
        sorted_counts = np.sort(normalized_counts)[::-1]  # descending order

        # Plot sorted frequency bars
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(NUM_EXPERTS), sorted_counts, color='salmon', edgecolor='black')
        plt.axhline(1.0, color='black', linestyle='--', linewidth=1.5, label='Uniform Average = 1.0')
        plt.title(f'Sorted Frequency Histogram (Log-Normal Samples = {TOTAL_SAMPLES})')
        plt.xlabel('Rank (Most Frequent → Least Frequent)')
        plt.ylabel('Relative Frequency (vs. Uniform)')
        plt.grid(True, axis='y')
        plt.legend()
        plt.tight_layout()
        plt.savefig("expert_dist.pdf")