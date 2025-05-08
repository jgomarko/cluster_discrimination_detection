"""
Statistical validation methods for discrimination detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..cmi import calculate_cmi

def permutation_test(data, clusters, sensitive_columns, outcome_column, 
                    nonsensitive_columns=None, num_permutations=100):
    """
    Perform a permutation test to validate the significance of the observed CMI.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing all attributes
    clusters : numpy.ndarray
        Cluster assignments for each sample
    sensitive_columns : list
        List of column names for sensitive attributes
    outcome_column : str
        Column name for the outcome variable
    nonsensitive_columns : list, optional
        List of column names for nonsensitive attributes
    num_permutations : int, optional
        Number of permutations to perform
        
    Returns:
    --------
    dict
        Dictionary containing observed_cmi, permutation_distribution, and p_value
    """
    # Calculate observed CMI
    observed_cmi = calculate_cmi(
        data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    
    # Perform permutation test
    permutation_distribution = []
    
    for _ in range(num_permutations):
        # Create a permuted dataset
        permuted_data = data.copy()
        
        # Permute the sensitive attributes within clusters
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 1:
                for sens_col in sensitive_columns:
                    # Get sensitive values for this cluster
                    sensitive_values = permuted_data.loc[cluster_indices, sens_col].values
                    
                    # Permute the values
                    np.random.shuffle(sensitive_values)
                    
                    # Update the dataset
                    permuted_data.loc[cluster_indices, sens_col] = sensitive_values
        
        # Calculate CMI for permuted data
        permuted_cmi = calculate_cmi(
            permuted_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
        )
        
        permutation_distribution.append(permuted_cmi)
    
    # Calculate p-value
    p_value = np.mean(np.array(permutation_distribution) >= observed_cmi)
    
    return {
        'observed_cmi': observed_cmi,
        'permutation_distribution': permutation_distribution,
        'p_value': p_value
    }

def bootstrap_ci(data, clusters, sensitive_columns, outcome_column, 
                nonsensitive_columns=None, num_bootstraps=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for the CMI.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing all attributes
    clusters : numpy.ndarray
        Cluster assignments for each sample
    sensitive_columns : list
        List of column names for sensitive attributes
    outcome_column : str
        Column name for the outcome variable
    nonsensitive_columns : list, optional
        List of column names for nonsensitive attributes
    num_bootstraps : int, optional
        Number of bootstrap samples
    confidence : float, optional
        Confidence level for the interval
        
    Returns:
    --------
    dict
        Dictionary containing bootstrap_distribution and confidence interval
    """
    # Calculate observed CMI
    observed_cmi = calculate_cmi(
        data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    
    # Perform bootstrap sampling
    bootstrap_distribution = []
    
    for _ in range(num_bootstraps):
        # Create a bootstrap sample with replacement
        bootstrap_indices = np.random.choice(
            len(data), len(data), replace=True
        )
        bootstrap_data = data.iloc[bootstrap_indices].reset_index(drop=True)
        
        # Get corresponding clusters
        bootstrap_clusters = clusters[bootstrap_indices]
        
        # Calculate CMI for the bootstrap sample
        bootstrap_cmi = calculate_cmi(
            bootstrap_data, bootstrap_clusters, sensitive_columns, 
            outcome_column, nonsensitive_columns
        )
        
        bootstrap_distribution.append(bootstrap_cmi)
    
    # Calculate confidence interval
    alpha = (1 - confidence) / 2
    lower_ci = np.percentile(bootstrap_distribution, alpha * 100)
    upper_ci = np.percentile(bootstrap_distribution, (1 - alpha) * 100)
    
    return {
        'observed_cmi': observed_cmi,
        'bootstrap_distribution': bootstrap_distribution,
        'ci': (lower_ci, upper_ci),
        'confidence': confidence
    }

def plot_permutation_test(permutation_results):
    """
    Visualize the permutation test results.
    
    Parameters:
    -----------
    permutation_results : dict
        Output from the permutation_test function
    """
    observed_cmi = permutation_results['observed_cmi']
    permutation_distribution = permutation_results['permutation_distribution']
    p_value = permutation_results['p_value']
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of permutation distribution
    plt.hist(permutation_distribution, bins=30, alpha=0.7, color='skyblue',
            label='Permutation distribution')
    
    # Add observed value
    plt.axvline(x=observed_cmi, color='red', linestyle='--',
               label=f'Observed CMI: {observed_cmi:.4f}')
    
    # Add annotation for p-value
    plt.text(0.7, 0.9, f'p-value: {p_value:.4f}',
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('CMI Value')
    plt.ylabel('Frequency')
    plt.title('Permutation Test for CMI Significance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('permutation_test.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_bootstrap_distribution(bootstrap_results):
    """
    Visualize the bootstrap distribution and confidence interval.
    
    Parameters:
    -----------
    bootstrap_results : dict
        Output from the bootstrap_ci function
    """
    observed_cmi = bootstrap_results['observed_cmi']
    bootstrap_distribution = bootstrap_results['bootstrap_distribution']
    lower_ci, upper_ci = bootstrap_results['ci']
    confidence = bootstrap_results['confidence']
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of bootstrap distribution
    plt.hist(bootstrap_distribution, bins=30, alpha=0.7, color='lightgreen',
            label='Bootstrap distribution')
    
    # Add observed value
    plt.axvline(x=observed_cmi, color='red', linestyle='--',
               label=f'Observed CMI: {observed_cmi:.4f}')
    
    # Add confidence interval
    plt.axvline(x=lower_ci, color='blue', linestyle=':',
               label=f'Lower CI: {lower_ci:.4f}')
    plt.axvline(x=upper_ci, color='blue', linestyle=':',
               label=f'Upper CI: {upper_ci:.4f}')
    
    # Shade the confidence interval region
    plt.axvspan(lower_ci, upper_ci, alpha=0.2, color='blue',
               label=f'{confidence*100}% Confidence Interval')
    
    plt.xlabel('CMI Value')
    plt.ylabel('Frequency')
    plt.title('Bootstrap Distribution and Confidence Interval')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('bootstrap_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()