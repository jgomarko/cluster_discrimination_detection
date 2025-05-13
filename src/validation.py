# Statistical validation methods
"""
Statistical validation methods for discrimination detection.

This module implements statistical validation methods described in the paper:
- Permutation test for CMI significance
- Bootstrap confidence intervals for CMI
- Statistical comparison between clusters
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.cmi import calculate_cmi

def permutation_test(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns, 
                     num_permutations=1000, random_state=42, show_progress=True):
    """
    Perform permutation test to assess statistical significance of CMI.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        num_permutations (int): Number of permutations
        random_state (int): Random seed for reproducibility
        show_progress (bool): Whether to show progress bar
        
    Returns:
        dict: Test results including p-value
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Calculate observed CMI
    observed_cmi = calculate_cmi(
        data, 
        clusters, 
        sensitive_columns, 
        outcome_column, 
        nonsensitive_columns
    )
    
    print(f"Observed CMI: {observed_cmi:.4f}")
    print(f"Performing permutation test with {num_permutations} permutations...")
    
    # Initialize permutation distribution
    permuted_cmi_values = []
    
    # Create iterator with or without progress bar
    if show_progress:
        iterator = tqdm(range(num_permutations))
    else:
        iterator = range(num_permutations)
    
    # Perform permutation test
    for m in iterator:
        # Create a permuted dataset by shuffling sensitive attributes
        permuted_data = data.copy()
        
        # Shuffle each sensitive column independently
        for column in sensitive_columns:
            permuted_data[column] = np.random.permutation(permuted_data[column].values)
        
        # Calculate CMI for permuted data
        permuted_cmi = calculate_cmi(
            permuted_data, 
            clusters, 
            sensitive_columns, 
            outcome_column, 
            nonsensitive_columns
        )
        
        permuted_cmi_values.append(permuted_cmi)
    
    # Calculate p-value (proportion of permuted CMI values >= observed CMI)
    p_value = np.mean(np.array(permuted_cmi_values) >= observed_cmi)
    
    # Calculate effect size (Cohen's d)
    # This measures the standardized difference between observed CMI and null distribution
    effect_size = (observed_cmi - np.mean(permuted_cmi_values)) / np.std(permuted_cmi_values)
    
    print(f"Permutation test complete.")
    print(f"p-value: {p_value:.4f}")
    print(f"Effect size (Cohen's d): {effect_size:.2f}")
    
    return {
        'observed_cmi': observed_cmi,
        'permuted_cmi_values': permuted_cmi_values,
        'p_value': p_value,
        'effect_size': effect_size,
        'significant': p_value < 0.05
    }

def bootstrap_ci(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns,
                num_bootstraps=1000, confidence_level=0.95, random_state=42, show_progress=True):
    """
    Calculate bootstrap confidence intervals for CMI.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        num_bootstraps (int): Number of bootstrap samples
        confidence_level (float): Confidence level (e.g., 0.95 for 95% CI)
        random_state (int): Random seed for reproducibility
        show_progress (bool): Whether to show progress bar
        
    Returns:
        dict: Bootstrap results including confidence intervals
    """
    # Set random seed
    np.random.seed(random_state)
    
    # Calculate observed CMI
    observed_cmi = calculate_cmi(
        data, 
        clusters, 
        sensitive_columns, 
        outcome_column, 
        nonsensitive_columns
    )
    
    print(f"Observed CMI: {observed_cmi:.4f}")
    print(f"Calculating bootstrap confidence interval with {num_bootstraps} samples...")
    
    n = len(data)
    bootstrap_cmi_values = []
    
    # Create iterator with or without progress bar
    if show_progress:
        iterator = tqdm(range(num_bootstraps))
    else:
        iterator = range(num_bootstraps)
    
    for b in iterator:
        # Create bootstrap sample by sampling with replacement
        indices = np.random.choice(n, size=n, replace=True)
        bootstrap_data = data.iloc[indices].reset_index(drop=True)
        bootstrap_clusters = clusters[indices]
        
        # Calculate CMI for bootstrap sample
        bootstrap_cmi = calculate_cmi(
            bootstrap_data, 
            bootstrap_clusters, 
            sensitive_columns, 
            outcome_column, 
            nonsensitive_columns
        )
        
        bootstrap_cmi_values.append(bootstrap_cmi)
    
    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    bootstrap_cmi_values = np.array(bootstrap_cmi_values)
    lower_bound = np.percentile(bootstrap_cmi_values, lower_percentile)
    upper_bound = np.percentile(bootstrap_cmi_values, upper_percentile)
    
    print(f"Bootstrap confidence interval ({confidence_level*100:.0f}%): [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    return {
        'observed_cmi': observed_cmi,
        'bootstrap_cmi_values': bootstrap_cmi_values,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'mean': np.mean(bootstrap_cmi_values),
        'std': np.std(bootstrap_cmi_values)
    }

def compare_clusters_statistically(data, clusters1, clusters2, sensitive_columns, 
                                  outcome_column, nonsensitive_columns, 
                                  num_bootstraps=1000, random_state=42):
    """
    Statistically compare CMI values between two cluster configurations.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters1 (np.ndarray): First cluster assignments
        clusters2 (np.ndarray): Second cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        num_bootstraps (int): Number of bootstrap samples
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Comparison results
    """
    # Calculate bootstrap statistics for both configurations
    print("Computing bootstrap statistics for first clustering configuration...")
    bootstrap1 = bootstrap_ci(
        data, clusters1, sensitive_columns, outcome_column, nonsensitive_columns,
        num_bootstraps, 0.95, random_state
    )
    
    print("\nComputing bootstrap statistics for second clustering configuration...")
    bootstrap2 = bootstrap_ci(
        data, clusters2, sensitive_columns, outcome_column, nonsensitive_columns,
        num_bootstraps, 0.95, random_state
    )
    
    # Calculate Z-score for difference
    z_score = (bootstrap1['mean'] - bootstrap2['mean']) / np.sqrt(
        bootstrap1['std']**2 + bootstrap2['std']**2
    )
    
    # Calculate p-value (two-tailed test)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    
    # Determine if difference is significant
    is_significant = p_value < 0.05
    
    print(f"\nComparison results:")
    print(f"CMI for first configuration: {bootstrap1['mean']:.4f}")
    print(f"CMI for second configuration: {bootstrap2['mean']:.4f}")
    print(f"Difference: {bootstrap1['mean'] - bootstrap2['mean']:.4f}")
    print(f"Z-score: {z_score:.2f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Statistically significant: {is_significant}")
    
    return {
        'cmi1': bootstrap1['mean'],
        'cmi2': bootstrap2['mean'],
        'difference': bootstrap1['mean'] - bootstrap2['mean'],
        'z_score': z_score,
        'p_value': p_value,
        'is_significant': is_significant
    }

def validate_discrimination_cluster(data, cluster_id, sensitive_columns, outcome_column, nonsensitive_columns, 
                                    num_permutations=1000, num_bootstraps=1000, random_state=42):
    """
    Perform comprehensive validation of discrimination detection for a specific cluster.
    
    Parameters:
        data (pd.DataFrame): The data with cluster assignments in 'cluster' column
        cluster_id: ID of the cluster to validate
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        num_permutations (int): Number of permutations for permutation test
        num_bootstraps (int): Number of bootstrap samples
        random_state (int): Random seed for reproducibility
        
    Returns:
        dict: Validation results
    """
    print(f"Validating discrimination in cluster {cluster_id}...")
    
    # Extract data for this cluster
    cluster_data = data[data['cluster'] == cluster_id].copy()
    cluster_size = len(cluster_data)
    
    print(f"Cluster size: {cluster_size} samples")
    
    # Skip if cluster is too small
    if cluster_size < 20:
        print(f"Cluster too small for reliable validation")
        return None
    
    # Since we're analyzing a single cluster, cluster assignments are all 0
    cluster_assignments = np.zeros(cluster_size)
    
    # Perform permutation test
    print("\nPerforming permutation test...")
    perm_test = permutation_test(
        cluster_data, 
        cluster_assignments, 
        sensitive_columns, 
        outcome_column, 
        nonsensitive_columns,
        num_permutations=num_permutations,
        random_state=random_state
    )
    
    # Calculate bootstrap confidence interval
    print("\nCalculating bootstrap confidence interval...")
    bootstrap = bootstrap_ci(
        cluster_data, 
        cluster_assignments, 
        sensitive_columns, 
        outcome_column, 
        nonsensitive_columns,
        num_bootstraps=num_bootstraps,
        random_state=random_state
    )
    
    # Generate summary
    validation_results = {
        'cluster_id': cluster_id,
        'cluster_size': cluster_size,
        'observed_cmi': perm_test['observed_cmi'],
        'p_value': perm_test['p_value'],
        'effect_size': perm_test['effect_size'],
        'ci_lower': bootstrap['lower_bound'],
        'ci_upper': bootstrap['upper_bound'],
        'significant': perm_test['p_value'] < 0.05
    }
    
    print("\nValidation summary:")
    print(f"CMI: {validation_results['observed_cmi']:.4f}")
    print(f"p-value: {validation_results['p_value']:.4f}")
    print(f"Effect size: {validation_results['effect_size']:.2f}")
    print(f"95% CI: [{validation_results['ci_lower']:.4f}, {validation_results['ci_upper']:.4f}]")
    print(f"Statistically significant: {validation_results['significant']}")
    
    return validation_results

def plot_permutation_test(permutation_test_results, save_path=None):
    """
    Visualize permutation test results.
    
    Parameters:
        permutation_test_results (dict): Results from permutation_test()
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram of permutation distribution
    plt.hist(permutation_test_results['permuted_cmi_values'], bins=30, alpha=0.7,
             label='Null Distribution')
    
    # Add vertical line for observed CMI
    plt.axvline(permutation_test_results['observed_cmi'], color='red', linewidth=2, 
                label=f'Observed CMI: {permutation_test_results["observed_cmi"]:.4f}')
    
    # Annotation for p-value and effect size
    plt.annotate(f'p-value: {permutation_test_results["p_value"]:.4f}\n'
                 f'Effect size: {permutation_test_results["effect_size"]:.2f}',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 ha='right', va='top',
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.xlabel('CMI Value')
    plt.ylabel('Frequency')
    plt.title('Permutation Test for CMI Significance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Permutation test plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_bootstrap_distribution(bootstrap_results, save_path=None):
    """
    Visualize bootstrap distribution and confidence interval.
    
    Parameters:
        bootstrap_results (dict): Results from bootstrap_ci()
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Create histogram of bootstrap distribution
    plt.hist(bootstrap_results['bootstrap_cmi_values'], bins=30, alpha=0.7,
             label='Bootstrap Distribution')
    
    # Add vertical lines for confidence interval
    plt.axvline(bootstrap_results['lower_bound'], color='green', linewidth=2, linestyle='--',
                label=f'Lower Bound: {bootstrap_results["lower_bound"]:.4f}')
    plt.axvline(bootstrap_results['upper_bound'], color='red', linewidth=2, linestyle='--',
                label=f'Upper Bound: {bootstrap_results["upper_bound"]:.4f}')
    
    # Add vertical line for observed value
    plt.axvline(bootstrap_results['observed_cmi'], color='blue', linewidth=2,
                label=f'Observed CMI: {bootstrap_results["observed_cmi"]:.4f}')
    
    plt.xlabel('CMI Value')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution with 95% Confidence Interval')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Bootstrap distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

# Example usage
if __name__ == "__main__":
    from src.preprocessing import load_adult_dataset, preprocess_adult_dataset
    from src.clustering import MultiClusteringAlgorithm
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_adult_dataset()
    processed_data, sensitive_columns, nonsensitive_columns, outcome_column = preprocess_adult_dataset(data)
    
    # Take a small subset for quick testing
    subset_size = 1000
    subset_data = processed_data.sample(subset_size, random_state=42).reset_index(drop=True)
    
    # Perform clustering
    print("\nPerforming clustering...")
    clustering = MultiClusteringAlgorithm()
    clusters = clustering.fit(subset_data[nonsensitive_columns].values, algorithm='kmeans', n_clusters=3)
    
    # Add clusters to data
    subset_data['cluster'] = clusters
    
    # Perform validation for the first cluster
    cluster_id = 0
    cluster_data = subset_data[subset_data['cluster'] == cluster_id]
    cluster_assignments = np.zeros(len(cluster_data))
    
    # Permutation test
    print("\nPerforming permutation test...")
    perm_results = permutation_test(
        cluster_data,
        cluster_assignments,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns,
        num_permutations=100  # Reduced for quick testing
    )
    
    # Plot permutation test results
    plot_permutation_test(perm_results)
    
    # Bootstrap confidence interval
    print("\nCalculating bootstrap confidence interval...")
    bootstrap_results = bootstrap_ci(
        cluster_data,
        cluster_assignments,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns,
        num_bootstraps=100  # Reduced for quick testing
    )
    
    # Plot bootstrap distribution
    plot_bootstrap_distribution(bootstrap_results)
