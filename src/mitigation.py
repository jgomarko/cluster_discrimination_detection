# Bias mitigation strategies
"""
Mitigation strategies for reducing discrimination in automated decision-making systems.

This module implements three main strategies from the paper:
1. Reweighting - Assigns weights to samples based on sensitive attributes
2. Fairness Regularization - Adds a fairness term to the standard loss function
3. Subgroup-Specific Calibration - Creates calibrated models for each subgroup

These strategies can be applied to reduce discrimination detected by the CMI analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from src.cmi import calculate_cmi

def reweighting(data, clusters, sensitive_columns, outcome_column):
    """
    Apply reweighting strategy to mitigate discrimination.
    
    Parameters:
        data (pd.DataFrame): The data
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        
    Returns:
        pd.DataFrame: Data with weights added
    """
    print("Applying reweighting strategy...")
    
    # Add cluster assignments to data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Create sensitive attribute combinations
    data_with_clusters['sensitive_group'] = data_with_clusters[sensitive_columns].apply(
        lambda row: '_'.join(row.astype(str)),
        axis=1
    )
    
    # Calculate weights for each cluster and sensitive group combination
    weights = {}
    
    for cluster in np.unique(clusters):
        cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster]
        cluster_size = len(cluster_data)
        
        if cluster_size < 10:  # Skip small clusters
            continue
            
        # Count samples in each sensitive group within this cluster
        group_counts = cluster_data['sensitive_group'].value_counts()
        group_proportions = group_counts / cluster_size
        
        # Calculate weights as inverse of group proportion
        for group, proportion in group_proportions.items():
            weights[(cluster, group)] = 1.0 / proportion if proportion > 0 else 1.0
    
    # Apply weights to each sample
    data_with_clusters['weight'] = data_with_clusters.apply(
        lambda row: weights.get((row['cluster'], row['sensitive_group']), 1.0),
        axis=1
    )
    
    # Normalize weights to sum to n (dataset size)
    n = len(data_with_clusters)
    data_with_clusters['weight'] = data_with_clusters['weight'] * n / data_with_clusters['weight'].sum()
    
    print("Reweighting complete.")
    print(f"Weight statistics: min={data_with_clusters['weight'].min():.2f}, " + 
          f"mean={data_with_clusters['weight'].mean():.2f}, " + 
          f"max={data_with_clusters['weight'].max():.2f}")
    
    return data_with_clusters

class FairnessRegularizedModel:
    """
    Model with fairness regularization based on CMI.
    
    This implements the fairness regularization framework from the paper:
    L_fair(θ) = L(θ) + λ · CMI(Y_pred; S | X, C)
    """
    
    def __init__(self, base_model=None, lambda_param=1.0):
        """
        Initialize the fairness-regularized model.
        
        Parameters:
            base_model: Base prediction model (default: LogisticRegression)
            lambda_param (float): Regularization parameter
        """
        self.base_model = base_model if base_model is not None else LogisticRegression(max_iter=1000)
        self.lambda_param = lambda_param
        self.fair_model = None
        self.scaler = StandardScaler()
    
    def fit(self, X, y, sensitive, clusters, nonsensitive_columns):
        """
        Train model with fairness regularization.
        
        Parameters:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            sensitive (pd.DataFrame): Sensitive attributes
            clusters (np.ndarray): Cluster assignments
            nonsensitive_columns (list): List of non-sensitive columns
            
        Returns:
            self: Trained model
        """
        # Split into train and validation for fairness assessment
        X_train, X_val, y_train, y_val, sensitive_train, sensitive_val, clusters_train, clusters_val = train_test_split(
            X, y, sensitive, clusters, test_size=0.3, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # First, train base model without fairness constraints
        self.base_model.fit(X_train_scaled, y_train)
        
        # Make predictions on validation set
        val_pred_prob = self.base_model.predict_proba(X_val_scaled)[:, 1]
        
        # Create validation dataframe for CMI calculation
        val_data = pd.DataFrame(X_val.values, columns=X_val.columns)
        val_data[sensitive.columns] = sensitive_val.reset_index(drop=True)
        val_data['prediction'] = (val_pred_prob > 0.5).astype(int)
        
        # Calculate initial CMI on predictions
        initial_cmi = calculate_cmi(
            val_data,
            clusters_val,
            sensitive.columns.tolist(),
            'prediction',
            nonsensitive_columns
        )
        
        print(f"Initial CMI before regularization: {initial_cmi:.4f}")
        
        # Implement fairness regularization
        # For simplicity, we'll use a reweighting approach where weights are
        # calculated to reduce discrimination
        
        # Apply reweighting strategy
        weighted_data = reweighting(
            pd.concat([X_train, sensitive_train, y_train], axis=1),
            clusters_train,
            sensitive.columns.tolist(),
            y_train.name
        )
        
        # Train model with fairness weights
        self.fair_model = LogisticRegression(max_iter=1000)
        self.fair_model.fit(
            X_train_scaled, 
            weighted_data[y_train.name],
            sample_weight=weighted_data['weight'].values
        )
        
        # Evaluate fairness improvement
        fair_pred_prob = self.fair_model.predict_proba(X_val_scaled)[:, 1]
        val_data['fair_prediction'] = (fair_pred_prob > 0.5).astype(int)
        
        # Calculate CMI after regularization
        final_cmi = calculate_cmi(
            val_data,
            clusters_val,
            sensitive.columns.tolist(),
            'fair_prediction',
            nonsensitive_columns
        )
        
        print(f"Final CMI after regularization: {final_cmi:.4f}")
        print(f"CMI reduction: {(1 - final_cmi/initial_cmi)*100:.2f}%")
        
        # Calculate accuracy impact
        initial_acc = accuracy_score(y_val, val_data['prediction'])
        final_acc = accuracy_score(y_val, val_data['fair_prediction'])
        
        print(f"Initial accuracy: {initial_acc:.4f}")
        print(f"Final accuracy: {final_acc:.4f}")
        print(f"Accuracy change: {(final_acc - initial_acc)*100:.2f}%")
        
        return self
    
    def predict(self, X):
        """Make predictions using the fair model."""
        if self.fair_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.fair_model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Make probability predictions using the fair model."""
        if self.fair_model is None:
            raise ValueError("Model has not been trained. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.fair_model.predict_proba(X_scaled)

def subgroup_calibration(data, clusters, sensitive_columns, base_model=None):
    """
    Apply subgroup-specific calibration to improve fairness.
    
    Parameters:
        data (pd.DataFrame): The data with features, sensitive attributes, and outcome
        clusters (np.ndarray): Cluster assignments
        sensitive_columns (list): List of sensitive attribute columns
        base_model: Base prediction model (default: LogisticRegression)
        
    Returns:
        dict: Calibrated models for each subgroup
    """
    print("Applying subgroup-specific calibration...")
    
    # Add cluster assignments to data
    data_with_clusters = data.copy()
    data_with_clusters['cluster'] = clusters
    
    # Create sensitive attribute combinations
    data_with_clusters['sensitive_group'] = data_with_clusters[sensitive_columns].apply(
        lambda row: '_'.join(row.astype(str)),
        axis=1
    )
    
    # Identify outcome column (assuming it's the only binary column not in sensitive_columns)
    binary_cols = data_with_clusters.select_dtypes(include=['int64', 'float64']).columns
    binary_cols = [col for col in binary_cols if set(data_with_clusters[col].unique()).issubset({0, 1})]
    outcome_column = [col for col in binary_cols if col not in sensitive_columns and col != 'cluster'][0]
    
    # Identify feature columns (all except sensitive, outcome, cluster, and sensitive_group)
    exclude_cols = sensitive_columns + [outcome_column, 'cluster', 'sensitive_group']
    feature_columns = [col for col in data_with_clusters.columns if col not in exclude_cols]
    
    # Split into train/test
    train_data, test_data = train_test_split(data_with_clusters, test_size=0.3, random_state=42)
    
    # Train a base model
    if base_model is None:
        base_model = LogisticRegression(max_iter=1000)
    
    X_train = train_data[feature_columns]
    y_train = train_data[outcome_column]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train base model
    base_model.fit(X_train_scaled, y_train)
    
    # Create calibrated models for each subgroup
    calibrated_models = {}
    performance = {}
    
    # Get unique combinations of cluster and sensitive group
    subgroups = []
    for cluster in np.unique(train_data['cluster']):
        for group in train_data[train_data['cluster'] == cluster]['sensitive_group'].unique():
            subgroups.append((cluster, group))
    
    # For each subgroup, create a calibrated model
    for cluster, group in subgroups:
        subgroup_key = f"cluster{cluster}_group{group}"
        
        # Get data for this subgroup
        mask = (train_data['cluster'] == cluster) & (train_data['sensitive_group'] == group)
        subgroup_data = train_data[mask]
        
        # Skip if too few samples
        if len(subgroup_data) < 20:
            print(f"Skipping {subgroup_key} (only {len(subgroup_data)} samples)")
            continue
        
        # Create a calibrated model for this subgroup
        subgroup_X = subgroup_data[feature_columns]
        subgroup_y = subgroup_data[outcome_column]
        
        # Scale features
        subgroup_X_scaled = scaler.transform(subgroup_X)
        
        # Use Platt scaling for calibration
        calibrated_model = CalibratedClassifierCV(
            base_estimator=base_model,
            method='sigmoid',
            cv='prefit'
        )
        
        # Fit on this subgroup's data
        calibrated_model.fit(subgroup_X_scaled, subgroup_y)
        
        # Store the model
        calibrated_models[subgroup_key] = {
            'model': calibrated_model,
            'scaler': scaler,
            'cluster': cluster,
            'group': group
        }
        
        # Evaluate on test data for this subgroup
        test_mask = (test_data['cluster'] == cluster) & (test_data['sensitive_group'] == group)
        test_subgroup = test_data[test_mask]
        
        if len(test_subgroup) >= 10:
            test_X = test_subgroup[feature_columns]
            test_y = test_subgroup[outcome_column]
            
            # Scale features
            test_X_scaled = scaler.transform(test_X)
            
            # Get predictions
            base_preds = base_model.predict(test_X_scaled)
            calibrated_preds = calibrated_model.predict(test_X_scaled)
            
            # Calculate metrics
            base_acc = accuracy_score(test_y, base_preds)
            calibrated_acc = accuracy_score(test_y, calibrated_preds)
            
            performance[subgroup_key] = {
                'base_acc': base_acc,
                'calibrated_acc': calibrated_acc,
                'samples': len(test_subgroup)
            }
    
    # Print summary
    print(f"Created calibrated models for {len(calibrated_models)} subgroups")
    
    # Calculate overall improvement
    if performance:
        base_accs = [p['base_acc'] * p['samples'] for p in performance.values()]
        calibrated_accs = [p['calibrated_acc'] * p['samples'] for p in performance.values()]
        total_samples = sum(p['samples'] for p in performance.values())
        
        weighted_base_acc = sum(base_accs) / total_samples
        weighted_calibrated_acc = sum(calibrated_accs) / total_samples
        
        print(f"Overall base accuracy: {weighted_base_acc:.4f}")
        print(f"Overall calibrated accuracy: {weighted_calibrated_acc:.4f}")
        print(f"Accuracy change: {(weighted_calibrated_acc - weighted_base_acc)*100:.2f}%")
    
    return {
        'calibrated_models': calibrated_models,
        'base_model': base_model,
        'scaler': scaler,
        'performance': performance
    }

def evaluate_mitigation(original_data, mitigated_data, sensitive_columns, outcome_column, clusters, nonsensitive_columns):
    """
    Evaluate the effectiveness of a mitigation strategy.
    
    Parameters:
        original_data (pd.DataFrame): Original data before mitigation
        mitigated_data (pd.DataFrame): Data after applying mitigation strategy
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        clusters (np.ndarray): Cluster assignments
        nonsensitive_columns (list): List of non-sensitive columns
        
    Returns:
        dict: Evaluation metrics
    """
    print("Evaluating mitigation effectiveness...")
    
    # Calculate CMI before and after mitigation
    original_cmi = calculate_cmi(
        original_data,
        clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    mitigated_cmi = calculate_cmi(
        mitigated_data,
        clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    # Calculate CMI reduction
    cmi_reduction = 1 - (mitigated_cmi / original_cmi) if original_cmi > 0 else 0
    
    # Calculate CMI for each cluster before and after
    original_cmi_per_cluster = {}
    mitigated_cmi_per_cluster = {}
    
    for cluster in np.unique(clusters):
        # Filter data for this cluster
        original_cluster = original_data[clusters == cluster]
        mitigated_cluster = mitigated_data[clusters == cluster]
        
        # Skip small clusters
        if len(original_cluster) < 20:
            continue
        
        # Calculate CMI for this cluster
        original_cmi_per_cluster[cluster] = calculate_cmi(
            original_cluster,
            np.zeros(len(original_cluster)),  # All in same cluster
            sensitive_columns,
            outcome_column,
            nonsensitive_columns
        )
        
        mitigated_cmi_per_cluster[cluster] = calculate_cmi(
            mitigated_cluster,
            np.zeros(len(mitigated_cluster)),  # All in same cluster
            sensitive_columns,
            outcome_column,
            nonsensitive_columns
        )
    
    # Print results
    print(f"Overall CMI before mitigation: {original_cmi:.4f}")
    print(f"Overall CMI after mitigation: {mitigated_cmi:.4f}")
    print(f"CMI reduction: {cmi_reduction*100:.2f}%")
    
    print("\nCMI per cluster:")
    for cluster in original_cmi_per_cluster:
        original = original_cmi_per_cluster[cluster]
        mitigated = mitigated_cmi_per_cluster.get(cluster, 0)
        reduction = 1 - (mitigated / original) if original > 0 else 0
        
        print(f"Cluster {cluster}: {original:.4f} -> {mitigated:.4f} ({reduction*100:.2f}% reduction)")
    
    return {
        'original_cmi': original_cmi,
        'mitigated_cmi': mitigated_cmi,
        'cmi_reduction': cmi_reduction,
        'original_cmi_per_cluster': original_cmi_per_cluster,
        'mitigated_cmi_per_cluster': mitigated_cmi_per_cluster
    }

def plot_mitigation_comparison(evaluation_results, save_path=None):
    """
    Create visualization comparing original and mitigated CMI values.
    
    Parameters:
        evaluation_results (dict): Results from evaluate_mitigation()
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data
    clusters = sorted(evaluation_results['original_cmi_per_cluster'].keys())
    original_values = [evaluation_results['original_cmi_per_cluster'][c] for c in clusters]
    mitigated_values = [evaluation_results['mitigated_cmi_per_cluster'].get(c, 0) for c in clusters]
    
    # Set up bar positions
    x = np.arange(len(clusters))
    width = 0.35
    
    # Create bars
    plt.bar(x - width/2, original_values, width, label='Original CMI')
    plt.bar(x + width/2, mitigated_values, width, label='Mitigated CMI')
    
    # Add labels and formatting
    plt.xlabel('Cluster')
    plt.ylabel('CMI Value')
    plt.title('CMI Values Before and After Mitigation')
    plt.xticks(x, [f'Cluster {c}' for c in clusters])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add overall reduction as text
    reduction = evaluation_results['cmi_reduction'] * 100
    plt.figtext(0.5, 0.01, f'Overall CMI Reduction: {reduction:.2f}%', 
                ha='center', fontsize=12)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Mitigation comparison plot saved to {save_path}")
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
    
    # Split data into features, sensitive attributes, and outcome
    X = subset_data[nonsensitive_columns]
    y = subset_data[outcome_column]
    sensitive = subset_data[sensitive_columns]
    
    # Apply reweighting
    print("\nApplying reweighting strategy...")
    reweighted_data = reweighting(subset_data, clusters, sensitive_columns, outcome_column)
    
    # Train model with fairness regularization
    print("\nTraining model with fairness regularization...")
    fair_model = FairnessRegularizedModel(lambda_param=1.0)
    fair_model.fit(X, y, sensitive, clusters, nonsensitive_columns)
    
    # Apply subgroup calibration
    print("\nApplying subgroup calibration...")
    calibration_results = subgroup_calibration(subset_data, clusters, sensitive_columns)
    
    print("\nMitigation strategies applied successfully.")
