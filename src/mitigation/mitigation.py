"""
Fairness mitigation strategies for discrimination detection framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ..cmi import calculate_cmi, calculate_cmi_per_cluster

def reweighting(data, clusters, sensitive_columns, outcome_column):
    """
    Apply reweighting to mitigate discrimination.
    
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
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with added weight column
    """
    # Create a copy of the data
    reweighted_data = data.copy()
    
    # Add cluster column if it doesn't exist
    if 'cluster' not in reweighted_data.columns:
        reweighted_data['cluster'] = clusters
    
    # Initialize weights to 1
    reweighted_data['weight'] = 1.0
    
    # Process each cluster
    for cluster_id in np.unique(clusters):
        cluster_data = reweighted_data[reweighted_data['cluster'] == cluster_id]
        
        # Skip clusters with too few samples
        if len(cluster_data) < 20:
            continue
        
        # Calculate weights for each subgroup defined by sensitive attributes and outcome
        subgroups = []
        
        # Create a single subgroup identifier column
        subgroup_cols = sensitive_columns + [outcome_column]
        
        # Group by all combinations of sensitive attributes and outcome
        for _, subgroup in cluster_data.groupby(subgroup_cols):
            subgroups.append(subgroup)
        
        # Compute the average subgroup size
        avg_size = len(cluster_data) / len(subgroups)
        
        # Assign weights inversely proportional to subgroup size
        for subgroup in subgroups:
            if len(subgroup) > 0:
                weight = avg_size / len(subgroup)
                for idx in subgroup.index:
                    reweighted_data.loc[idx, 'weight'] = weight
    
    return reweighted_data

class FairnessRegularizedModel:
    """
    Model with fairness regularization to mitigate discrimination.
    """
    
    def __init__(self, lambda_param=1.0, base_estimator=None):
        """
        Initialize the fairness-regularized model.
        
        Parameters:
        -----------
        lambda_param : float
            Regularization parameter controlling the fairness-accuracy trade-off
        base_estimator : object, optional
            Base estimator to use. If None, LogisticRegression is used.
        """
        self.lambda_param = lambda_param
        
        if base_estimator is None:
            self.base_model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            self.base_model = base_estimator
            
        self.scaler = StandardScaler()
        self.sensitive_dict = {}
    
    def fit(self, X, y, sensitive, clusters, feature_names=None):
        """
        Fit the fairness-regularized model.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        sensitive : pandas.DataFrame or numpy.ndarray
            Sensitive attributes
        clusters : numpy.ndarray
            Cluster assignments
        feature_names : list, optional
            Names of features
            
        Returns:
        --------
        self
            Fitted model
        """
        # Store sensitive attribute mappings
        if isinstance(sensitive, pd.DataFrame):
            for col in sensitive.columns:
                self.sensitive_dict[col] = sensitive[col].values
        elif isinstance(sensitive, np.ndarray):
            self.sensitive_dict['sensitive'] = sensitive
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train base model with fairness regularization
        if hasattr(self.base_model, 'set_params'):
            # If model supports custom regularization, add fairness penalty
            self.base_model.set_params(
                C=1.0/(1.0 + self.lambda_param)
            )
        
        # Create sample weights based on fairness
        sample_weights = self._compute_fairness_weights(X, y, clusters)
        
        # Fit the model
        self.base_model.fit(X_scaled, y, sample_weight=sample_weights)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the fairness-regularized model.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        X_scaled = self.scaler.transform(X)
        return self.base_model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities with the fairness-regularized model.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Predicted probabilities
        """
        if hasattr(self.base_model, 'predict_proba'):
            X_scaled = self.scaler.transform(X)
            return self.base_model.predict_proba(X_scaled)
        else:
            raise AttributeError("Base model does not support predict_proba")
    
    def _compute_fairness_weights(self, X, y, clusters):
        """
        Compute sample weights to enforce fairness.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target vector
        clusters : numpy.ndarray
            Cluster assignments
            
        Returns:
        --------
        numpy.ndarray
            Sample weights
        """
        weights = np.ones(len(y))
        
        # Process each cluster
        for cluster_id in np.unique(clusters):
            mask = clusters == cluster_id
            
            if np.sum(mask) < 20:
                continue
            
            # Get cluster data
            cluster_X = X[mask]
            cluster_y = y[mask]
            
            # Process each sensitive attribute
            for sens_name, sens_values in self.sensitive_dict.items():
                cluster_sens = sens_values[mask]
                
                # Get unique sensitive values
                unique_sens = np.unique(cluster_sens)
                
                for sens_val in unique_sens:
                    # Get subgroup
                    subgroup_mask = cluster_sens == sens_val
                    subgroup_y = cluster_y[subgroup_mask]
                    
                    # Calculate outcome rate
                    subgroup_rate = np.mean(subgroup_y)
                    cluster_rate = np.mean(cluster_y)
                    
                    # Calculate weight based on outcome rate difference
                    weight_factor = 1.0 / (1.0 + self.lambda_param * abs(subgroup_rate - cluster_rate))
                    
                    # Apply weights to this subgroup
                    subgroup_indices = np.where(np.logical_and(mask, cluster_sens == sens_val))[0]
                    weights[subgroup_indices] *= weight_factor
        
        return weights

def subgroup_calibration(data, clusters, sensitive_columns, base_model=None):
    """
    Apply subgroup-specific calibration to improve fairness.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing all attributes
    clusters : numpy.ndarray
        Cluster assignments for each sample
    sensitive_columns : list
        List of column names for sensitive attributes
    base_model : object, optional
        Base classifier to calibrate. If None, LogisticRegression is used.
        
    Returns:
    --------
    dict
        Dictionary containing calibrated models and performance metrics
    """
    # Identify outcome column
    outcome_column = [col for col in data.columns 
                    if col not in sensitive_columns and col != 'cluster'][0]
    
    # Create a copy of the data with cluster information
    data_with_clusters = data.copy()
    if 'cluster' not in data_with_clusters.columns:
        data_with_clusters['cluster'] = clusters
    
    # Prepare feature columns
    feature_columns = [col for col in data.columns 
                     if col not in sensitive_columns + [outcome_column, 'cluster']]
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    train_data, val_data = train_test_split(data_with_clusters, test_size=0.2, random_state=42)
    
    # Create base model if None
    if base_model is None:
        base_model = LogisticRegression(max_iter=1000, random_state=42)
    
    # Train base model on all data
    X_train = train_data[feature_columns].values
    y_train = train_data[outcome_column].values
    base_model.fit(X_train, y_train)
    
    # Create calibrated models for each subgroup
    calibrated_models = {}
    performance = {}
    
    # Process each cluster
    for cluster_id in np.unique(clusters):
        # Get cluster data
        cluster_train = train_data[train_data['cluster'] == cluster_id]
        cluster_val = val_data[val_data['cluster'] == cluster_id]
        
        # Skip clusters with too few samples
        if len(cluster_train) < 50 or len(cluster_val) < 20:
            continue
        
        # Create subgroup identifier
        cluster_train['subgroup'] = cluster_train[sensitive_columns].apply(
            lambda row: '_'.join(str(row[col]) for col in sensitive_columns), axis=1
        )
        cluster_val['subgroup'] = cluster_val[sensitive_columns].apply(
            lambda row: '_'.join(str(row[col]) for col in sensitive_columns), axis=1
        )
        
        # Get unique subgroups
        subgroups = cluster_train['subgroup'].unique()
        
        for subgroup in subgroups:
            # Get subgroup data
            subgroup_train = cluster_train[cluster_train['subgroup'] == subgroup]
            subgroup_val = cluster_val[cluster_val['subgroup'] == subgroup]
            
            # Skip subgroups with too few samples
            if len(subgroup_train) < 30 or len(subgroup_val) < 10:
                continue
            
            # Create identifier
            subgroup_id = f"cluster{cluster_id}_{subgroup}"
            
            # Get features and target
            X_sub_train = subgroup_train[feature_columns].values
            y_sub_train = subgroup_train[outcome_column].values
            X_sub_val = subgroup_val[feature_columns].values
            y_sub_val = subgroup_val[outcome_column].values
            
            # Train calibrated model for this subgroup
            calibrated_model = CalibratedClassifierCV(
                base_estimator=base_model,
                cv='prefit',
                method='sigmoid'
            )
            
            try:
                calibrated_model.fit(X_sub_train, y_sub_train)
                calibrated_models[subgroup_id] = calibrated_model
                
                # Evaluate base model and calibrated model
                base_preds = base_model.predict(X_sub_val)
                base_acc = accuracy_score(y_sub_val, base_preds)
                
                calibrated_preds = calibrated_model.predict(X_sub_val)
                calibrated_acc = accuracy_score(y_sub_val, calibrated_preds)
                
                performance[subgroup_id] = {
                    'base_acc': base_acc,
                    'calibrated_acc': calibrated_acc,
                    'samples': len(subgroup_train)
                }
            except Exception as e:
                print(f"Error calibrating model for {subgroup_id}: {str(e)}")
    
    return {
        'calibrated_models': calibrated_models,
        'performance': performance
    }

def evaluate_mitigation(original_data, mitigated_data, clusters, sensitive_columns, outcome_column, 
                        nonsensitive_columns=None):
    """
    Evaluate the effectiveness of mitigation strategies.
    
    Parameters:
    -----------
    original_data : pandas.DataFrame
        Original data before mitigation
    mitigated_data : pandas.DataFrame
        Data after applying mitigation
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
        Dictionary containing evaluation metrics
    """
    # Calculate CMI before and after mitigation
    original_cmi = calculate_cmi(
        original_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    
    mitigated_cmi = calculate_cmi(
        mitigated_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    
    # Calculate reduction percentage
    cmi_reduction = 100 * (1 - mitigated_cmi / original_cmi) if original_cmi > 0 else 0
    
    # Calculate per-cluster CMI reduction
    orig_cmi_per_cluster = calculate_cmi_per_cluster(
        original_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    
    mitig_cmi_per_cluster = calculate_cmi_per_cluster(
        mitigated_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    
    per_cluster_reduction = {}
    
    for cluster_id in orig_cmi_per_cluster.keys():
        if cluster_id in mitig_cmi_per_cluster:
            orig = orig_cmi_per_cluster[cluster_id]
            mitig = mitig_cmi_per_cluster[cluster_id]
            
            if orig > 0:
                reduction = 100 * (1 - mitig / orig)
                per_cluster_reduction[cluster_id] = reduction
            else:
                per_cluster_reduction[cluster_id] = 0
    
    return {
        'original_cmi': original_cmi,
        'mitigated_cmi': mitigated_cmi,
        'cmi_reduction': cmi_reduction,
        'per_cluster_reduction': per_cluster_reduction
    }

def plot_mitigation_comparison(original_data, mitigated_data, clusters, sensitive_columns, outcome_column,
                              nonsensitive_columns=None):
    """
    Visualize the comparison of original and mitigated discrimination.
    
    Parameters:
    -----------
    original_data : pandas.DataFrame
        Original data before mitigation
    mitigated_data : pandas.DataFrame
        Data after applying mitigation
    clusters : numpy.ndarray
        Cluster assignments for each sample
    sensitive_columns : list
        List of column names for sensitive attributes
    outcome_column : str
        Column name for the outcome variable
    nonsensitive_columns : list, optional
        List of column names for nonsensitive attributes
    """
    # Evaluate mitigation
    evaluation = evaluate_mitigation(
        original_data, mitigated_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    
    # Create a comparison visualization
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Overall CMI comparison
    plt.subplot(2, 2, 1)
    plt.bar(['Original', 'Mitigated'], 
           [evaluation['original_cmi'], evaluation['mitigated_cmi']],
           color=['skyblue', 'lightgreen'])
    plt.title('Overall Discrimination (CMI)')
    plt.ylabel('CMI')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add text annotations
    plt.text(0, evaluation['original_cmi'] + 0.01, 
            f"{evaluation['original_cmi']:.4f}", 
            ha='center')
    plt.text(1, evaluation['mitigated_cmi'] + 0.01, 
            f"{evaluation['mitigated_cmi']:.4f}", 
            ha='center')
    plt.text(0.5, 0.9 * max(evaluation['original_cmi'], evaluation['mitigated_cmi']),
            f"Reduction: {evaluation['cmi_reduction']:.2f}%",
            ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    # Plot 2: Per-cluster CMI comparison
    plt.subplot(2, 2, 2)
    
    # Get clusters to plot (just show highest discrimination clusters if many)
    cluster_cmis = calculate_cmi_per_cluster(
        original_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns
    )
    top_clusters = sorted(cluster_cmis.items(), key=lambda x: x[1], reverse=True)[:5]
    clusters_to_plot = [c[0] for c in top_clusters]
    
    # Prepare data for plotting
    cluster_labels = [f'Cluster {c}' for c in clusters_to_plot]
    original_values = [calculate_cmi(
        original_data[clusters == c], np.zeros(sum(clusters == c)), 
        sensitive_columns, outcome_column, nonsensitive_columns
    ) for c in clusters_to_plot]
    
    mitigated_values = [calculate_cmi(
        mitigated_data[clusters == c], np.zeros(sum(clusters == c)), 
        sensitive_columns, outcome_column, nonsensitive_columns
    ) for c in clusters_to_plot]
    
    # Create grouped bar chart
    x = np.arange(len(cluster_labels))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original', color='skyblue')
    plt.bar(x + width/2, mitigated_values, width, label='Mitigated', color='lightgreen')
    
    plt.xlabel('Cluster')
    plt.ylabel('CMI')
    plt.title('Discrimination by Cluster')
    plt.xticks(x, cluster_labels)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Reduction by cluster
    plt.subplot(2, 2, 3)
    reductions = [100 * (1 - mitig/orig) if orig > 0 else 0 
                 for orig, mitig in zip(original_values, mitigated_values)]
    
    bars = plt.bar(cluster_labels, reductions, color='lightcoral')
    plt.xlabel('Cluster')
    plt.ylabel('Reduction (%)')
    plt.title('Discrimination Reduction by Cluster')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add text annotations
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f"{height:.1f}%", ha='center')
    
    # Plot 4: Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    summary_text = f"""
    MITIGATION SUMMARY
    
    Overall discrimination (CMI):
        Original: {evaluation['original_cmi']:.4f}
        Mitigated: {evaluation['mitigated_cmi']:.4f}
        Reduction: {evaluation['cmi_reduction']:.2f}%
    
    Discrimination by group:
        Most affected: Cluster {top_clusters[0][0]}
        Least affected: Cluster {top_clusters[-1][0]}
    
    Largest reduction:
        {max(reductions):.1f}% in Cluster {clusters_to_plot[np.argmax(reductions)]}
    """
    
    plt.text(0.1, 0.5, summary_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.savefig('mitigation_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()