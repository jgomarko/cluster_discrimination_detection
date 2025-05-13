"""
Conditional Mutual Information (CMI) calculation for discrimination detection.

This module implements the CMI calculation methods described in the paper:
- Basic CMI calculation
- Hierarchical CMI decomposition
- Interaction information calculation

These measures are used to detect hidden discrimination within clusters.
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.preprocessing import OneHotEncoder

def entropy(probabilities):
    """
    Calculate entropy for a set of probabilities.
    
    Parameters:
        probabilities (numpy.ndarray): Array of probabilities
        
    Returns:
        float: Entropy value
    """
    # Filter out zero probabilities to avoid log(0)
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

def conditional_entropy_discrete(data, condition_columns, target_column):
    """
    Calculate conditional entropy H(Y|X) for discrete variables.
    
    Parameters:
        data (pd.DataFrame): Data containing the variables
        condition_columns (list): Columns to condition on
        target_column (str): Target column
        
    Returns:
        float: Conditional entropy value
    """
    # Group by condition columns
    if len(condition_columns) == 1:
        grouped = data.groupby(condition_columns[0])
    else:
        # Create a composite key for multiple columns
        data['_condition_key'] = data[condition_columns].apply(
            lambda row: '_'.join(row.astype(str)), axis=1
        )
        grouped = data.groupby('_condition_key')
    
    total_entropy = 0
    total_samples = len(data)
    
    for group_name, group_data in grouped:
        # Calculate probability of this condition
        condition_prob = len(group_data) / total_samples
        
        # Calculate target distribution in this condition
        target_counts = group_data[target_column].value_counts(normalize=True)
        
        # Calculate entropy for this condition
        condition_entropy = entropy(target_counts.values)
        
        # Weight by condition probability
        total_entropy += condition_prob * condition_entropy
    
    return total_entropy

def calculate_cmi(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns):
    """
    Calculate Conditional Mutual Information (CMI) as described in the paper.
    
    CMI(Y; S | X, C) = H(Y | X, C) - H(Y | X, S, C)
    
    This implementation uses a discrete approach for all variables to avoid
    issues with mixed data types.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        
    Returns:
        float: CMI value
    """
    # Add cluster assignments to data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Make sure all columns are treated as categorical
    # First ensure all columns are string type for consistent handling
    for col in nonsensitive_columns + sensitive_columns + ['cluster', outcome_column]:
        if col in data_with_clusters.columns:
            data_with_clusters[col] = data_with_clusters[col].astype(str)
    
    # For numerical columns, we can bin them to create discrete categories
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numerical_cols:
        if col in nonsensitive_columns and col in data_with_clusters.columns:
            # Bin the column into 5 categories
            data_with_clusters[col] = pd.qcut(
                data[col], 
                q=5, 
                labels=False, 
                duplicates='drop'
            ).astype(str)
    
    # Calculate H(Y | X, C)
    # Include cluster column in conditioning
    condition_columns_xc = nonsensitive_columns + ['cluster']
    
    h_y_given_xc = conditional_entropy_discrete(
        data_with_clusters, 
        condition_columns_xc,
        outcome_column
    )
    
    # Calculate H(Y | X, S, C)
    # Include sensitive columns and cluster column in conditioning
    condition_columns_xsc = nonsensitive_columns + sensitive_columns + ['cluster']
    
    h_y_given_xsc = conditional_entropy_discrete(
        data_with_clusters, 
        condition_columns_xsc,
        outcome_column
    )
    
    # CMI = H(Y | X, C) - H(Y | X, S, C)
    cmi = h_y_given_xc - h_y_given_xsc
    
    return cmi

def calculate_cmi_per_cluster(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns):
    """
    Calculate CMI for each cluster separately.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        
    Returns:
        dict: Dictionary of CMI values per cluster
    """
    # Add cluster assignments to data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Calculate CMI for each cluster
    cmi_per_cluster = {}
    unique_clusters = np.unique(clusters)
    
    for cluster_id in unique_clusters:
        # Filter data for this cluster
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        
        # Skip small clusters (need enough samples for reliable estimation)
        if len(cluster_data) < 20:
            print(f"Skipping cluster {cluster_id} with only {len(cluster_data)} samples")
            continue
        
        # For single-cluster data, cluster column doesn't add information
        # so we calculate CMI within this cluster only
        cluster_cmi = calculate_cmi(
            cluster_data, 
            np.zeros(len(cluster_data)),  # All samples in same cluster
            sensitive_columns, 
            outcome_column, 
            nonsensitive_columns
        )
        
        cmi_per_cluster[cluster_id] = cluster_cmi
    
    return cmi_per_cluster

def hierarchical_cmi_decomposition(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns):
    """
    Decompose CMI hierarchically to see contribution of each sensitive attribute.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        
    Returns:
        dict: Contribution of each sensitive attribute to CMI
    """
    contributions = {}
    
    # Calculate base CMI with no sensitive attributes
    base_sensitive = []
    base_cmi = 0  # With no sensitive attributes, CMI is 0
    
    # Incrementally add each sensitive attribute
    for i, sensitive_attr in enumerate(sensitive_columns):
        current_sensitive = base_sensitive + [sensitive_attr]
        
        # Calculate CMI with current set of sensitive attributes
        current_cmi = calculate_cmi(
            data, 
            clusters, 
            current_sensitive, 
            outcome_column, 
            nonsensitive_columns
        )
        
        # Contribution is the difference from previous CMI
        contributions[sensitive_attr] = current_cmi - base_cmi
        
        # Update base for next iteration
        base_sensitive = current_sensitive
        base_cmi = current_cmi
    
    return contributions

def interaction_information(data, clusters, sensitive_attr1, sensitive_attr2, outcome_column, nonsensitive_columns):
    """
    Calculate interaction information between two sensitive attributes.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_attr1 (str): First sensitive attribute
        sensitive_attr2 (str): Second sensitive attribute
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        
    Returns:
        float: Interaction information value
    """
    # Calculate I(Y;S1,S2|X) - joint CMI
    cmi_both = calculate_cmi(
        data, 
        clusters, 
        [sensitive_attr1, sensitive_attr2], 
        outcome_column, 
        nonsensitive_columns
    )
    
    # Calculate I(Y;S1|X) - CMI for first attribute
    cmi_s1 = calculate_cmi(
        data, 
        clusters, 
        [sensitive_attr1], 
        outcome_column, 
        nonsensitive_columns
    )
    
    # Calculate I(Y;S2|X) - CMI for second attribute
    cmi_s2 = calculate_cmi(
        data, 
        clusters, 
        [sensitive_attr2], 
        outcome_column, 
        nonsensitive_columns
    )
    
    # Interaction information
    interaction = cmi_both - cmi_s1 - cmi_s2
    
    return interaction

def cmi_heatmap(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns):
    """
    Create a CMI heatmap data for combinations of sensitive attributes.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        nonsensitive_columns (list): List of non-sensitive attribute columns
        
    Returns:
        pd.DataFrame: Heatmap data with CMI values for attribute combinations
    """
    if len(sensitive_columns) < 2:
        raise ValueError("Need at least 2 sensitive columns for heatmap analysis")
    
    # Create heatmap data
    heatmap_data = []
    
    # Add cluster assignments to data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Process each cluster
    for cluster_id in np.unique(clusters):
        # Filter data for this cluster
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster_id]
        
        # Skip small clusters
        if len(cluster_data) < 20:
            continue
        
        # Get all unique values for each sensitive attribute
        attribute_values = {}
        for col in sensitive_columns:
            attribute_values[col] = cluster_data[col].unique()
        
        # For each combination of values for the first two sensitive attributes
        for val1 in attribute_values[sensitive_columns[0]]:
            for val2 in attribute_values[sensitive_columns[1]]:
                # Create filter for this combination
                mask = (cluster_data[sensitive_columns[0]] == val1) & \
                       (cluster_data[sensitive_columns[1]] == val2)
                
                subgroup_data = cluster_data[mask]
                
                # Skip if too few samples
                if len(subgroup_data) < 10:
                    continue
                
                # Calculate CMI for this subgroup
                # Since we're already within a specific combination of sensitive attributes,
                # we don't include sensitive_columns in the CMI calculation
                # We're measuring how much information other sensitive attributes provide
                if len(sensitive_columns) > 2:
                    other_sensitive = [col for col in sensitive_columns 
                                     if col != sensitive_columns[0] and col != sensitive_columns[1]]
                    
                    subgroup_cmi = calculate_cmi(
                        subgroup_data,
                        np.zeros(len(subgroup_data)),  # All in same subgroup
                        other_sensitive,
                        outcome_column,
                        nonsensitive_columns
                    )
                else:
                    # If only 2 sensitive attributes, use outcome rate as proxy
                    subgroup_cmi = subgroup_data[outcome_column].mean()
                
                # Add to heatmap data
                heatmap_data.append({
                    'cluster': cluster_id,
                    sensitive_columns[0]: val1,
                    sensitive_columns[1]: val2,
                    'count': len(subgroup_data),
                    'cmi': subgroup_cmi
                })
    
    return pd.DataFrame(heatmap_data)

# Example usage
if __name__ == "__main__":
    # Generate a synthetic example dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Create non-sensitive attributes
    X1 = np.random.uniform(0, 1, n_samples)
    X2 = np.random.uniform(0, 1, n_samples)
    
    # Create sensitive attributes
    S1 = np.random.binomial(1, 0.5, n_samples)  # Binary attribute (e.g., sex)
    S2 = np.random.choice(['A', 'B', 'C'], n_samples)  # Categorical attribute (e.g., race)
    
    # Create outcome that depends on both non-sensitive and sensitive attributes
    Y = np.zeros(n_samples, dtype=int)
    
    # X1 > 0.7 tends to make Y=1
    Y[X1 > 0.7] = 1
    
    # S1=1 and S2='A' makes Y more likely to be 1
    for i in range(n_samples):
        if S1[i] == 1 and S2[i] == 'A' and np.random.random() < 0.8:
            Y[i] = 1
    
    # Create clusters
    clusters = np.zeros(n_samples, dtype=int)
    clusters[X1 > 0.5] = 1  # Simple clustering based on X1
    
    # Create DataFrame
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'S1': S1,
        'S2': S2,
        'Y': Y
    })
    
    # Calculate CMI
    sensitive_columns = ['S1', 'S2']
    outcome_column = 'Y'
    nonsensitive_columns = ['X1', 'X2']
    
    cmi = calculate_cmi(
        data,
        clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    print(f"Overall CMI: {cmi:.4f}")
    
    # Calculate CMI per cluster
    cmi_per_cluster = calculate_cmi_per_cluster(
        data,
        clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    print("\nCMI per cluster:")
    for cluster_id, cluster_cmi in cmi_per_cluster.items():
        print(f"  Cluster {cluster_id}: {cluster_cmi:.4f}")
    
    # Hierarchical decomposition
    contributions = hierarchical_cmi_decomposition(
        data,
        clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    print("\nContributions to CMI:")
    for attr, value in contributions.items():
        print(f"  {attr}: {value:.4f} ({value/cmi*100:.1f}%)")
    
    # Interaction information
    interaction = interaction_information(
        data,
        clusters,
        'S1',
        'S2',
        outcome_column,
        nonsensitive_columns
    )
    
    print(f"\nInteraction information: {interaction:.4f}")
    print(f"{'Synergistic' if interaction > 0 else 'Redundant'} effect between S1 and S2")
    
    # Generate heatmap data
    heatmap_data = cmi_heatmap(
        data,
        clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    print("\nHeatmap data:")
    print(heatmap_data.head())