#!/usr/bin/env python3
"""
Example script for detecting discrimination using the clustering framework.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules from the project
from src.preprocessing import load_adult_dataset, preprocess_adult_dataset
from src.clustering import MultiClusteringAlgorithm
from src.cmi import calculate_cmi, calculate_cmi_per_cluster
from src.validation import permutation_test, bootstrap_ci, plot_permutation_test
from src.mitigation import reweighting, FairnessRegularizedModel
from src.utils import (setup_plotting_style, ensure_directory, save_figure,
                      calculate_fairness_metrics, plot_fairness_metrics)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect discrimination in a dataset.')
    
    parser.add_argument('--dataset', type=str, default='adult',
                        help='Dataset to use (adult or custom)')
    
    parser.add_argument('--data_path', type=str, default='../data/adult.data',
                        help='Path to the dataset file')
    
    parser.add_argument('--sensitive', type=str, nargs='+', default=['sex', 'race'],
                        help='Names of sensitive attribute columns')
    
    parser.add_argument('--outcome', type=str, default='income',
                        help='Name of the outcome column')
    
    parser.add_argument('--n_clusters', type=int, default=4,
                        help='Number of clusters to use')
    
    parser.add_argument('--algorithm', type=str, default='kmeans',
                        choices=['kmeans', 'gmm', 'spectral', 'ensemble'],
                        help='Clustering algorithm to use')
    
    parser.add_argument('--mitigate', action='store_true',
                        help='Apply mitigation techniques')
    
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save results')
    
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility')
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up plotting style
    setup_plotting_style()
    
    # Ensure output directory exists
    ensure_directory(args.output_dir)
    ensure_directory(os.path.join(args.output_dir, 'figures'))
    
    print(f"Loading dataset: {args.dataset}")
    
    # Load and preprocess data based on the dataset
    if args.dataset == 'adult':
        # Download dataset if not already available
        if not os.path.exists(args.data_path):
            import urllib.request
            os.makedirs(os.path.dirname(args.data_path), exist_ok=True)
            print("Downloading Adult dataset...")
            urllib.request.urlretrieve(
                'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                args.data_path
            )
            print("Download complete.")
        
        # Load and preprocess the Adult dataset
        data = load_adult_dataset(args.data_path)
        processed_data, sensitive_columns, nonsensitive_columns, outcome_column = preprocess_adult_dataset(data)
    else:
        # Load custom dataset
        data = pd.read_csv(args.data_path)
        
        # For custom datasets, use the provided sensitive and outcome columns
        sensitive_columns = args.sensitive
        outcome_column = args.outcome
        
        # Identify non-sensitive columns
        nonsensitive_columns = [col for col in data.columns 
                              if col not in sensitive_columns + [outcome_column]]
        
        # Use the data as-is for custom datasets
        processed_data = data
    
    print(f"Dataset loaded. Shape: {processed_data.shape}")
    print(f"Sensitive columns: {sensitive_columns}")
    print(f"Outcome column: {outcome_column}")
    
    # Split data into features and target
    X = processed_data[nonsensitive_columns]
    y = processed_data[outcome_column]
    sensitive = processed_data[sensitive_columns]
    
    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=args.random_state
    )
    
    # Reconstruct DataFrames
    train_data = pd.concat([X_train, sensitive_train, y_train], axis=1)
    test_data = pd.concat([X_test, sensitive_test, y_test], axis=1)
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Testing set: {len(test_data)} samples")
    
    # Make sure we're only using numerical features for clustering
    numerical_cols = X_train.select_dtypes(include=['number']).columns
    X_train_numeric = X_train[numerical_cols]
    X_test_numeric = X_test[numerical_cols]
    
    print(f"Using {len(numerical_cols)} numerical features for clustering")
    
    # Initialize clustering algorithm
    clustering = MultiClusteringAlgorithm()
    
    # Perform clustering
    print(f"Performing clustering with {args.algorithm} algorithm...")
    clusters = clustering.fit(
        X_train_numeric.values, 
        algorithm=args.algorithm, 
        n_clusters=args.n_clusters,
        random_state=args.random_state
    )
    
    # Evaluate clustering quality
    metrics = clustering.evaluate_clusters(X_train_numeric.values, clusters)
    print(f"Clustering quality:")
    print(f"  Silhouette score: {metrics['silhouette_score']:.3f}")
    print(f"  Davies-Bouldin index: {metrics['davies_bouldin_index']:.3f}")
    
    # Calculate CMI
    train_data_with_clusters = train_data.copy()
    train_data_with_clusters['cluster'] = clusters
    
    print("Calculating overall CMI...")
    cmi = calculate_cmi(
        train_data_with_clusters, 
        clusters, 
        sensitive_columns, 
        outcome_column, 
        nonsensitive_columns
    )
    print(f"Overall CMI: {cmi:.4f}")
    
    # Calculate CMI per cluster
    print("Calculating CMI per cluster...")
    cmi_per_cluster = calculate_cmi_per_cluster(
        train_data_with_clusters,
        clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    # Plot CMI by cluster
    plt.figure(figsize=(10, 6))
    cluster_ids = list(cmi_per_cluster.keys())
    cmi_values_list = list(cmi_per_cluster.values())
    
    # Sort by CMI value
    sorted_indices = np.argsort(cmi_values_list)[::-1]
    sorted_clusters = [cluster_ids[i] for i in sorted_indices]
    sorted_values = [cmi_values_list[i] for i in sorted_indices]
    
    plt.bar(sorted_clusters, sorted_values)
    plt.xlabel('Cluster')
    plt.ylabel('CMI')
    plt.title('Conditional Mutual Information by Cluster')
    plt.xticks(sorted_clusters)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    fig = plt.gcf()
    save_figure(fig, 'cmi_by_cluster.png', os.path.join(args.output_dir, 'figures'))
    plt.close(fig)
    
    # Identify high-discrimination clusters
    threshold = np.mean(list(cmi_per_cluster.values())) + 0.5 * np.std(list(cmi_per_cluster.values()))
    high_discrim_clusters = [c for c, v in cmi_per_cluster.items() if v > threshold]
    
    print("\nCMI per cluster:")
    for cluster_id, cluster_cmi in sorted(cmi_per_cluster.items(), key=lambda x: x[1], reverse=True):
        print(f"  Cluster {cluster_id}: {cluster_cmi:.4f}" + 
              (" (high discrimination)" if cluster_id in high_discrim_clusters else ""))
    
    # Analyze high-discrimination clusters
    for cluster_id in high_discrim_clusters:
        print(f"\nCluster {cluster_id} characteristics:")
        cluster_data = train_data_with_clusters[train_data_with_clusters['cluster'] == cluster_id]
        
        # Outcome rate
        outcome_rate = cluster_data[outcome_column].mean()
        print(f"  Overall outcome rate: {outcome_rate:.4f}")
        print(f"  Size: {len(cluster_data)} samples")
        
        # Distribution by sensitive attributes
        for col in sensitive_columns:
            print(f"\n  Distribution by {col}:")
            value_counts = cluster_data[col].value_counts(normalize=True)
            
            for value, proportion in value_counts.items():
                # Get outcome rate for this subgroup
                subgroup = cluster_data[cluster_data[col] == value]
                subgroup_outcome_rate = subgroup[outcome_column].mean()
                divergence = subgroup_outcome_rate - outcome_rate
                
                print(f"    {col}={value}: {proportion:.4f} of cluster, " +
                      f"outcome rate: {subgroup_outcome_rate:.4f} ({divergence:+.4f})")
    
    # Statistical validation for the highest-discrimination cluster
    if high_discrim_clusters:
        target_cluster = high_discrim_clusters[0]
        
        # Extract data for this cluster
        cluster_data = train_data_with_clusters[train_data_with_clusters['cluster'] == target_cluster]
        cluster_size = len(cluster_data)
        
        print(f"\nValidating discrimination in cluster {target_cluster} (size: {cluster_size})...")
        
        # Set all samples to same cluster (since we're analyzing within a single cluster)
        cluster_assignments = np.zeros(cluster_size)
        
        # Perform permutation test
        print("Performing permutation test...")
        perm_results = permutation_test(
            cluster_data,
            cluster_assignments,
            sensitive_columns,
            outcome_column,
            nonsensitive_columns,
            num_permutations=100
        )
        
        # Plot permutation test results
        fig = plot_permutation_test(perm_results)
        save_figure(fig, f'cluster_{target_cluster}_permutation_test.png', 
                  os.path.join(args.output_dir, 'figures'))
        plt.close(fig)
        
        print(f"Permutation test p-value: {perm_results['p_value']:.4f}")
        
        # Mitigation (if requested)
        if args.mitigate:
            print("\nApplying mitigation strategies...")
            
            # Apply reweighting
            print("1. Reweighting strategy...")
            reweighted_data = reweighting(
                train_data_with_clusters, 
                clusters, 
                sensitive_columns, 
                outcome_column
            )
            
            # Train a fair model
            print("2. Training fairness-regularized model...")
            fair_model = FairnessRegularizedModel(lambda_param=1.0)
            fair_model.fit(
                X_train, 
                y_train, 
                sensitive_train, 
                clusters, 
                nonsensitive_columns
            )
            
            # Make predictions on test set
            y_pred_fair = fair_model.predict(X_test)
            
            # Train a regular model for comparison
            from sklearn.linear_model import LogisticRegression
            regular_model = LogisticRegression(max_iter=1000)
            regular_model.fit(X_train, y_train)
            y_pred_reg = regular_model.predict(X_test)
            
            # Calculate accuracy
            from sklearn.metrics import accuracy_score
            fair_acc = accuracy_score(y_test, y_pred_fair)
            reg_acc = accuracy_score(y_test, y_pred_reg)
            
            print(f"Regular model accuracy: {reg_acc:.4f}")
            print(f"Fair model accuracy: {fair_acc:.4f}")
            print(f"Accuracy change: {(fair_acc - reg_acc)*100:+.2f}%")
            
            # Calculate fairness metrics for both models
            for col in sensitive_columns:
                print(f"\nFairness metrics for {col}:")
                
                # Regular model
                reg_metrics = calculate_fairness_metrics(
                    y_test.values, 
                    y_pred_reg, 
                    sensitive_test[col].values
                )
                
                print("Regular model:")
                print(f"  Accuracy disparity: {reg_metrics['accuracy_disparity']:.4f}")
                print(f"  Recall disparity: {reg_metrics['recall_disparity']:.4f}")
                
                # Fair model
                fair_metrics = calculate_fairness_metrics(
                    y_test.values, 
                    y_pred_fair, 
                    sensitive_test[col].values
                )
                
                print("Fair model:")
                print(f"  Accuracy disparity: {fair_metrics['accuracy_disparity']:.4f}")
                print(f"  Recall disparity: {fair_metrics['recall_disparity']:.4f}")
                
                # Plot fairness metrics
                fig = plot_fairness_metrics(
                    fair_metrics, 
                    title=f"Fairness Metrics by {col} (Fair Model)"
                )
                save_figure(
                    fig, 
                    f'fairness_metrics_{col}.png',
                    os.path.join(args.output_dir, 'figures')
                )
                plt.close(fig)
    
    print("\nAnalysis complete. Results saved to", args.output_dir)

if __name__ == "__main__":
    main()