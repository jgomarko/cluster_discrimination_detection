"""
Conditional Mutual Information calculations for discrimination detection.
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def calculate_cmi(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns=None):
    """
    Calculate the Conditional Mutual Information between sensitive attributes and outcome,
    conditioned on nonsensitive attributes.
    
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
        
    Returns:
    --------
    float
        Conditional Mutual Information value
    """
    # Handle case where nonsensitive_columns is None
    if nonsensitive_columns is None:
        nonsensitive_columns = [col for col in data.columns 
                              if col not in sensitive_columns + [outcome_column]]
    
    # Combine sensitive attributes into a single vector if multiple
    if len(sensitive_columns) > 1:
        # Create a joint representation of sensitive attributes
        sens_values = data[sensitive_columns].values
        # Use a simple hash function to create a unique combined value
        sensitive = np.sum(sens_values * np.array([10**i for i in range(len(sensitive_columns))]), axis=1)
    else:
        sensitive = data[sensitive_columns[0]].values
    
    # Get outcome values
    outcome = data[outcome_column].values
    
    # Calculate MI between sensitive attributes and outcome
    mi_sensitive_outcome = mutual_information(sensitive, outcome)
    
    # Calculate CMI by considering each cluster separately
    cmi = 0
    unique_clusters = np.unique(clusters)
    
    for cluster_id in unique_clusters:
        # Get samples in this cluster
        cluster_mask = clusters == cluster_id
        cluster_size = np.sum(cluster_mask)
        cluster_weight = cluster_size / len(data)
        
        # Calculate MI within this cluster
        cluster_sensitive = sensitive[cluster_mask]
        cluster_outcome = outcome[cluster_mask]
        
        if len(np.unique(cluster_sensitive)) > 1 and len(np.unique(cluster_outcome)) > 1:
            cluster_mi = mutual_information(cluster_sensitive, cluster_outcome)
            cmi += cluster_weight * cluster_mi
    
    return cmi

def calculate_cmi_per_cluster(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns=None):
    """
    Calculate CMI for each cluster separately.
    
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
        
    Returns:
    --------
    dict
        Dictionary mapping cluster IDs to CMI values
    """
    # Handle case where nonsensitive_columns is None
    if nonsensitive_columns is None:
        nonsensitive_columns = [col for col in data.columns 
                              if col not in sensitive_columns + [outcome_column]]
    
    # Combine sensitive attributes into a single vector if multiple
    if len(sensitive_columns) > 1:
        # Create a joint representation of sensitive attributes
        sens_values = data[sensitive_columns].values
        # Use a simple hash function to create a unique combined value
        sensitive = np.sum(sens_values * np.array([10**i for i in range(len(sensitive_columns))]), axis=1)
    else:
        sensitive = data[sensitive_columns[0]].values
    
    # Get outcome values
    outcome = data[outcome_column].values
    
    # Calculate CMI for each cluster
    cmi_per_cluster = {}
    unique_clusters = np.unique(clusters)
    
    for cluster_id in unique_clusters:
        # Get samples in this cluster
        cluster_mask = clusters == cluster_id
        cluster_sensitive = sensitive[cluster_mask]
        cluster_outcome = outcome[cluster_mask]
        
        # Skip if cluster has fewer than 10 samples or only one unique value
        if np.sum(cluster_mask) < 10 or len(np.unique(cluster_sensitive)) <= 1 or len(np.unique(cluster_outcome)) <= 1:
            cmi_per_cluster[cluster_id] = 0
            continue
        
        # Calculate MI within this cluster
        cmi_per_cluster[cluster_id] = mutual_information(cluster_sensitive, cluster_outcome)
    
    return cmi_per_cluster

def hierarchical_cmi_decomposition(data, clusters, sensitive_columns, outcome_column, nonsensitive_columns=None):
    """
    Decompose the CMI contribution of each sensitive attribute.
    
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
        
    Returns:
    --------
    dict
        Dictionary mapping sensitive attributes to their CMI contributions
    """
    # Handle case where nonsensitive_columns is None
    if nonsensitive_columns is None:
        nonsensitive_columns = [col for col in data.columns 
                              if col not in sensitive_columns + [outcome_column]]
    
    # Calculate individual contribution of each sensitive attribute
    contributions = {}
    
    for sens_attr in sensitive_columns:
        # Calculate CMI for just this attribute
        single_attr_cmi = calculate_cmi(
            data, 
            clusters, 
            [sens_attr], 
            outcome_column, 
            nonsensitive_columns
        )
        
        contributions[sens_attr] = single_attr_cmi
    
    return contributions

def interaction_information(data, clusters, sens_attr1, sens_attr2, outcome_column, nonsensitive_columns=None):
    """
    Calculate interaction information between two sensitive attributes.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing all attributes
    clusters : numpy.ndarray
        Cluster assignments for each sample
    sens_attr1 : str
        First sensitive attribute
    sens_attr2 : str
        Second sensitive attribute
    outcome_column : str
        Column name for the outcome variable
    nonsensitive_columns : list, optional
        List of column names for nonsensitive attributes
        
    Returns:
    --------
    float
        Interaction information value
    """
    # Handle case where nonsensitive_columns is None
    if nonsensitive_columns is None:
        nonsensitive_columns = [col for col in data.columns 
                              if col not in [sens_attr1, sens_attr2, outcome_column]]
    
    # Calculate CMI for individual attributes
    cmi1 = calculate_cmi(data, clusters, [sens_attr1], outcome_column, nonsensitive_columns)
    cmi2 = calculate_cmi(data, clusters, [sens_attr2], outcome_column, nonsensitive_columns)
    
    # Calculate joint CMI
    joint_cmi = calculate_cmi(data, clusters, [sens_attr1, sens_attr2], outcome_column, nonsensitive_columns)
    
    # Interaction information = Joint CMI - (CMI1 + CMI2)
    interaction = joint_cmi - (cmi1 + cmi2)
    
    return interaction

def mutual_information(x, y):
    """
    Calculate mutual information between two variables.
    
    Parameters:
    -----------
    x : numpy.ndarray
        First variable
    y : numpy.ndarray
        Second variable
        
    Returns:
    --------
    float
        Mutual information value
    """
    # Convert to 2D array as required by mutual_info_regression
    x = x.reshape(-1, 1)
    
    # Calculate MI in both directions and take average for symmetry
    mi_xy = mutual_info_regression(x, y, random_state=42)[0]
    mi_yx = mutual_info_regression(y.reshape(-1, 1), x.flatten(), random_state=42)[0]
    
    return (mi_xy + mi_yx) / 2