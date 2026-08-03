import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from tqdm.notebook import tqdm
from numba import jit, float64, int32


#Agglomerative Information Bottleneck implementation by Michel Kunkler: https://github.com/ltsstar/TaskExecutionTimeMining/blob/main/src/TaskExecutionTimeMining/information_bottleneck.py
#this code represents a modified version of the original code, using asymmetric normalization for mutual information.
# Custom entropy function for Numba compatibility
@jit(nopython=True)
def custom_entropy(p):
    """Compute entropy of a probability distribution, handling zero probabilities."""
    entropy_val = 0.0
    for prob in p:
        if prob > 0:  # 0 * log(0) is defined as 0 in information theory
            entropy_val -= prob * np.log2(prob)
    return entropy_val

# JIT-compiled function to compute H(Y|T') for a cluster merge
@jit(nopython=True)
def compute_conditional_entropy(joint_counts, n_current_clusters, i, j, n_bins_y):
    """Compute H(Y|T') for merging clusters i and j."""
    temp_counts = np.zeros((n_current_clusters - 1, n_bins_y), dtype=float64)
    
    # Compute new joint distribution after merging
    for c in range(n_current_clusters):
        if c == j:
            continue
        new_c = c if c < j else c - 1
        temp_counts[new_c] = joint_counts[c]
        if c == i:
            temp_counts[new_c] += joint_counts[j]
    
    # Normalize to get probabilities
    total = temp_counts.sum()
    if total == 0:
        return 0.0  # Handle edge case of no samples
    temp_prob = temp_counts / total
    p_t = temp_prob.sum(axis=1)
    p_y_given_t = np.zeros_like(temp_prob)
    
    # Compute conditional probabilities
    for c in range(len(p_t)):
        if p_t[c] > 0:
            p_y_given_t[c] = temp_prob[c] / p_t[c]
    
    # Compute H(Y|T') = sum_c P(T'=c) * H(Y|T'=c)
    H_Y_given_T = 0.0
    for c in range(len(p_t)):
        if p_t[c] > 0 and p_y_given_t[c].sum() > 0:
            H_Y_given_T += p_t[c] * custom_entropy(p_y_given_t[c])
    
    return H_Y_given_T

# JIT-compiled function to compute H(Y|T) for current clustering
@jit(nopython=True)
def compute_current_conditional_entropy(joint_counts, n_current_clusters, n_bins_y):
    """Compute H(Y|T) for the current clustering."""
    total = joint_counts.sum()
    if total == 0:
        return 0.0
    p_t = joint_counts.sum(axis=1) / total
    H_Y_given_T = 0.0
    for c in range(n_current_clusters):
        if p_t[c] > 0:
            p_y_given_t_c = joint_counts[c] / joint_counts[c].sum()
            if p_y_given_t_c.sum() > 0:
                H_Y_given_T += p_t[c] * custom_entropy(p_y_given_t_c)
    return H_Y_given_T

# Agglomerative Information Bottleneck algorithm
def agglomerative_information_bottleneck(X_d, Y_d, n_clusters, n_bins_x, n_bins_y):
    # Initialize clusters: each bin of X_d is its own cluster
    clusters = np.arange(n_bins_x)
    cluster_assignments = X_d.copy()  # Current cluster for each sample
    n_current_clusters = n_bins_x
    if n_current_clusters > 300: #set limit to 300 in favor execution time, can be removed by demand, but will be slow for large datasets
        return cluster_assignments, []
    # Debug: Check max indices in X_d and Y_d
    if np.max(X_d) >= n_bins_x or np.max(Y_d) >= n_bins_y:
        raise ValueError(f"Bin indices out of bounds: max(X_d)={np.max(X_d)} vs n_bins_x={n_bins_x}, max(Y_d)={np.max(Y_d)} vs n_bins_y={n_bins_y}")
    print(n_bins_x, n_bins_y)
    # Compute initial P(T, Y_d)
    joint_counts = np.zeros((n_bins_x, n_bins_y), dtype=np.float64)
    for i in range(len(X_d)):
        joint_counts[cluster_assignments[i], Y_d[i]] += 1
    # Compute H(Y) (constant across merges)
    p_y = joint_counts.sum(axis=0)
    total = p_y.sum()
    if total > 0:
        p_y = p_y / total
        H_Y = custom_entropy(p_y)
    else:
        H_Y = 0.0
    development = []
    H_Y_given_T = compute_current_conditional_entropy(joint_counts, n_current_clusters, n_bins_y)
    MI = H_Y - H_Y_given_T
    MI_norm = MI/H_Y if H_Y != 0 else MI #usage of asymmtetric normalization by https://arxiv.org/pdf/2307.01282
    MI_norm_original = MI_norm
    development.append((n_current_clusters, MI_norm, ()))
    print(f"Initial (Clusters: {n_current_clusters}): Mutual Information I(T; Y) = {MI_norm:.6f}")
    if MI_norm_original == 0 or MI_norm_original > 0 and MI_norm/MI_norm_original < 0.8:#check if information loss is not higher than 20% due to compression
        return cluster_assignments, development
    # Progress bar for cluster merging
    total_merges = max(0, n_bins_x - n_clusters)
    with tqdm(total=total_merges, desc="Merging clusters") as pbar:
        iteration = 0
        while n_current_clusters > n_clusters:
            min_loss = float('inf')
            merge_pair = None
            print(f'{n_current_clusters} {n_clusters}\n')
            # Calculate number of pairs to evaluate
            total_pairs = (n_current_clusters * (n_current_clusters - 1)) // 2
            # Sub-progress bar for pair evaluations
            with tqdm(total=total_pairs, desc=f"Iteration {iteration + 1} pairs", position=1, leave=False) as pair_pbar:
                pair_count = 0
                # Evaluate all pairs of clusters for merging
                for i in range(n_current_clusters):
                    for j in range(i + 1, n_current_clusters):
                        # Compute H(Y|T') using JIT-compiled function
                        H_Y_given_T = compute_conditional_entropy(joint_counts, n_current_clusters, i, j, n_bins_y)
                        
                        # Debug: Verify probability distributions
                        """temp_counts = np.zeros((n_current_clusters - 1, n_bins_y))
                        for c in range(n_current_clusters):
                            if c == j:
                                continue
                            new_c = c if c < j else c - 1
                            temp_counts[new_c] = joint_counts[c]
                            if c == i:
                                temp_counts[new_c] += joint_counts[j]
                        temp_prob = temp_counts / temp_counts.sum() if temp_counts.sum() > 0 else temp_counts
                        p_t = temp_prob.sum(axis=1)
                        p_y = temp_prob.sum(axis=0)
                        if p_y.sum() > 0 and not np.isclose(p_y.sum(), 1.0, rtol=1e-5):
                            print(f"Warning: p_y does not sum to 1: {p_y.sum()}")
                        for c in range(len(p_t)):
                            if p_t[c] > 0:
                                p_y_given_t_c = temp_prob[c] / p_t[c] if p_t[c] > 0 else np.zeros(n_bins_y)
                                if p_y_given_t_c.sum() > 0 and not np.isclose(p_y_given_t_c.sum(), 1.0, rtol=1e-5):
                                    print(f"Warning: p_y_given_t[{c}] does not sum to 1: {p_y_given_t_c.sum()}")"""
                        
                        # Update best merge
                        if H_Y_given_T < min_loss:
                            min_loss = H_Y_given_T
                            merge_pair = (i, j)

                        pair_count += 1
                        pair_pbar.update(1)  # Update pair progress bar after each evaluation
            
            # Perform the merge
            i, j = merge_pair
            cluster_assignments[cluster_assignments == clusters[j]] = clusters[i]
            clusters[j] = clusters[i]
            joint_counts[i] += joint_counts[j]
            joint_counts = np.delete(joint_counts, j, axis=0)
            clusters = np.delete(clusters, j)
            n_current_clusters -= 1
            
            # Compute and print I(T; Y) for the current clustering
            H_Y_given_T = compute_current_conditional_entropy(joint_counts, n_current_clusters, n_bins_y)
            MI = H_Y - H_Y_given_T
            MI_norm = MI/H_Y if H_Y != 0 else MI #usage of asymmtetric normalization by https://arxiv.org/pdf/2307.01282
            development.append((n_current_clusters, MI_norm, merge_pair))
            print(f"Iteration {iteration + 1} (Clusters: {n_current_clusters}): Mutual Information I(T; Y) = {MI_norm:.6f}")
            if MI_norm_original == 0 or MI_norm_original > 0 and MI_norm/MI_norm_original < 0.8: #check if information loss is not higher than 20% due to compression
                return cluster_assignments, development
            iteration += 1
            pbar.update(1)  # Update progress bar after each merge
    # Map back to k clusters (relabel for consistency)
    unique_clusters = np.unique(cluster_assignments)
    cluster_map = {old: new for new, old in enumerate(unique_clusters)}
    cluster_assignments = np.array([cluster_map[c] for c in cluster_assignments])
    
    return cluster_assignments, development

# Main function to cluster X based on MI with Y
def information_bottleneck_clustering(X, Y, n_clusters=3, n_bins_x=50, n_bins_y=20, x_is_discrete=False, y_is_discrete=False):
    # Validate input data
    if len(X) == 0 or len(Y) == 0:
        raise ValueError("Input arrays X or Y are empty")
    
    # Validate X for NaN/Inf only if not discrete
    if not x_is_discrete:
        if np.any(np.isnan(X)):
            raise ValueError("Input array X contains NaN values")
        if np.any(np.isinf(X)):
            raise ValueError("Input array X contains infinite values")
    else:
        # Ensure X has valid discrete values (non-empty, no None)
        if np.any(X == None):
            raise ValueError("Input array X contains None values")
    
    # Validate Y for NaN/Inf only if not discrete
    if not y_is_discrete:
        if np.any(np.isnan(Y)):
            raise ValueError("Input array Y contains NaN values")
        if np.any(np.isinf(Y)):
            raise ValueError("Input array Y contains infinite values")
    else:
        if np.any(Y == None):
            raise ValueError("Input array Y contains None values")
    
    # Adjust n_bins_x and n_bins_y to not exceed unique values
    n_bins_x = min(n_bins_x, len(np.unique(X)))
    n_bins_y = min(n_bins_y, len(np.unique(Y)))

    # Process X
    if x_is_discrete:
        # For discrete X, map unique values to consecutive integers
        unique_x, X_d = np.unique(X, return_inverse=True)
        n_bins_x = len(unique_x)
        X_d = X_d.astype(int)
        discretizer_x = None  # No discretizer needed
    else:
        # Discretize continuous X using quantile strategy
        discretizer_x = KBinsDiscretizer(n_bins=n_bins_x, encode='ordinal', strategy='quantile')
        X_d = discretizer_x.fit_transform(X.reshape(-1, 1)).astype(int).flatten()
    
    # Process Y
    if y_is_discrete:
        # For discrete Y, map unique values to consecutive integers
        unique_y, Y_d = np.unique(Y, return_inverse=True)
        n_bins_y = len(unique_y)
        Y_d = Y_d.astype(int)
        discretizer_y = None  # No discretizer needed
    else:
        # Discretize continuous Y using quantile strategy
        discretizer_y = KBinsDiscretizer(n_bins=n_bins_y, encode='ordinal', strategy='quantile')
        Y_d = discretizer_y.fit_transform(Y.reshape(-1, 1)).astype(int).flatten()
    
    # Remap X_d and Y_d to use only populated bins
    valid_x_bins = np.unique(X_d)
    valid_y_bins = np.unique(Y_d)
    n_bins_x = len(valid_x_bins)
    n_bins_y = len(valid_y_bins)
    x_bin_map = {old: new for new, old in enumerate(valid_x_bins)}
    y_bin_map = {old: new for new, old in enumerate(valid_y_bins)}
    X_d = np.array([x_bin_map[x] for x in X_d])
    Y_d = np.array([y_bin_map[y] for y in Y_d])
    
    # Debug: Print bin information
    #print(f"X {'discrete (categorical)' if x_is_discrete else 'continuous (quantile bins)'}, Adjusted n_bins_x: {n_bins_x}, Max X_d index: {np.max(X_d)}, Unique X_d bins: {len(np.unique(X_d))}")
    #print(f"Y {'discrete' if y_is_discrete else 'continuous (quantile bins)'}, Adjusted n_bins_y: {n_bins_y}, Max Y_d index: {np.max(Y_d)}, Unique Y_d bins: {len(np.unique(Y_d))}")
    
    # Run agglomerative IB with adjusted n_bins_x and n_bins_y
    cluster_assignments, development = agglomerative_information_bottleneck(X_d, Y_d, n_clusters, n_bins_x, n_bins_y)
    
    # Map discretized bins back to continuous X ranges (or original discrete values)
    if x_is_discrete:
        # For discrete X, map back to original unique values
        bin_edges = unique_x  # Use unique values as "bin edges"
        cluster_ranges = []
        for c in range(n_clusters):
            bins_in_cluster = np.unique(X_d[cluster_assignments == c])
            ranges = [bin_edges[i] for i in bins_in_cluster]  # List discrete values
            ranges.sort()  # Sort for clarity (if numeric)
            cluster_ranges.append(ranges)
    else:
        # For continuous X, use bin edges from discretizer
        bin_edges = discretizer_x.bin_edges_[0]
        cluster_ranges = []
        for c in range(n_clusters):
            bins_in_cluster = np.unique(X_d[cluster_assignments == c])
            ranges = [(bin_edges[i], bin_edges[i + 1]) for i in bins_in_cluster]
            ranges.sort()  # Sort for clarity
            cluster_ranges.append(ranges)
    
    return cluster_assignments, cluster_ranges, development