"""
Analysis of the Adult dataset using the discrimination detection framework.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import our modules
from src.preprocessing import load_adult_dataset, preprocess_adult_dataset
from src.clustering import MultiClusteringAlgorithm
from src.cmi import calculate_cmi, calculate_cmi_per_cluster, hierarchical_cmi_decomposition, interaction_information

def main():
    # Step 1: Load and preprocess the Adult dataset
    print("Loading and preprocessing the Adult dataset...")
    data = load_adult_dataset()
    processed_data, sensitive_columns, nonsensitive_columns, outcome_column = preprocess_adult_dataset(data)
    
    print(f"\nProcessed data shape: {processed_data.shape}")
    print(f"Sensitive columns: {sensitive_columns}")
    print(f"Outcome column: {outcome_column}")
    
    # Step 2: Split data into training and testing sets
    print("\nSplitting data into training and testing sets...")
    X = processed_data[nonsensitive_columns]
    y = processed_data[outcome_column]
    sensitive = processed_data[sensitive_columns]
    
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42
    )
    
    # Reconstruct DataFrames
    train_data = pd.concat([X_train, sensitive_train, y_train], axis=1)
    test_data = pd.concat([X_test, sensitive_test, y_test], axis=1)
    
    print(f"Training set: {len(train_data)} samples")
    print(f"Testing set: {len(test_data)} samples")
    
    # Step 3: Perform clustering on non-sensitive attributes
    print("\nPerforming clustering...")
    clustering = MultiClusteringAlgorithm()
    
    # Try different numbers of clusters
    k_values = [5, 7, 9]
    algorithms = ['kmeans', 'gmm', 'spectral', 'ensemble']
    
    best_cmi = -1
    best_config = None
    best_clusters = None
    
    results = []
    
    for algorithm in algorithms:
        for k in k_values:
            print(f"\nTrying {algorithm} with k={k}...")
            
            # Skip spectral clustering with high k due to memory constraints
            if algorithm == 'spectral' and k > 7:
                print(f"  Skipping spectral clustering with k={k} (memory intensive)")
                continue
                
            try:
                # Perform clustering
                if algorithm == 'ensemble':
                    clusters = clustering.ensemble_clustering(X_train.values, n_clusters=k, random_state=42)
                else:
                    clusters = clustering.fit(X_train.values, algorithm=algorithm, n_clusters=k, random_state=42)
                
                # Evaluate clustering quality
                metrics = clustering.evaluate_clusters(X_train.values, clusters)
                silhouette = metrics['silhouette_score']
                davies_bouldin = metrics['davies_bouldin_index']
                
                print(f"  Silhouette score: {silhouette:.3f}")
                print(f"  Davies-Bouldin index: {davies_bouldin:.3f}")
                
                # Calculate CMI for this clustering
                cmi = calculate_cmi(
                    train_data,
                    clusters,
                    sensitive_columns,
                    outcome_column,
                    nonsensitive_columns
                )
                
                print(f"  Overall CMI: {cmi:.4f}")
                
                # Store results
                results.append({
                    'algorithm': algorithm,
                    'k': k,
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin,
                    'cmi': cmi
                })
                
                # Check if this is the best configuration so far
                if cmi > best_cmi:
                    best_cmi = cmi
                    best_config = (algorithm, k)
                    best_clusters = clusters
            
            except Exception as e:
                print(f"  Error with {algorithm}, k={k}: {e}")
    
    # Step 4: Analyze the best clustering configuration
    print("\n--- Best Configuration ---")
    print(f"Algorithm: {best_config[0]}, k={best_config[1]}, CMI={best_cmi:.4f}")
    
    # Calculate CMI per cluster
    print("\nCalculating CMI per cluster...")
    cmi_per_cluster = calculate_cmi_per_cluster(
        train_data,
        best_clusters,
        sensitive_columns,
        outcome_column,
        nonsensitive_columns
    )
    
    # Identify high-discrimination clusters
    threshold = np.mean(list(cmi_per_cluster.values())) + 0.5 * np.std(list(cmi_per_cluster.values()))
    high_discrim_clusters = [c for c, v in cmi_per_cluster.items() if v > threshold]
    
    print("\nCMI per cluster:")
    for cluster_id, cluster_cmi in sorted(cmi_per_cluster.items(), key=lambda x: x[1], reverse=True):
        print(f"  Cluster {cluster_id}: {cluster_cmi:.4f}" + 
              (" (high discrimination)" if cluster_id in high_discrim_clusters else ""))
    
    # Step 5: Analyze cluster characteristics
    print("\nAnalyzing cluster characteristics...")
    train_data['cluster'] = best_clusters
    
    # Get distribution by sensitive attributes
    for cluster_id in high_discrim_clusters:
        print(f"\nCluster {cluster_id} characteristics:")
        cluster_data = train_data[train_data['cluster'] == cluster_id]
        
        # Outcome rate
        outcome_rate = cluster_data[outcome_column].mean()
        print(f"  Outcome rate: {outcome_rate:.2f}")
        print(f"  Size: {len(cluster_data)} samples")
        
        # Distribution by sensitive attributes
        for col in sensitive_columns:
            print(f"\n  Distribution by {col}:")
            value_counts = cluster_data[col].value_counts(normalize=True)
            
            for value, proportion in value_counts.items():
                # Get outcome rate for this subgroup
                subgroup = cluster_data[cluster_data[col] == value]
                subgroup_outcome_rate = subgroup[outcome_column].mean()
                
                print(f"    {col}={value}: {proportion:.2f} of cluster, outcome rate: {subgroup_outcome_rate:.2f}")
    
    # Step 6: Hierarchical decomposition
    print("\nHierarchical CMI decomposition for high-discrimination clusters:")
    for cluster_id in high_discrim_clusters:
        cluster_data = train_data[train_data['cluster'] == cluster_id]
        
        contributions = hierarchical_cmi_decomposition(
            cluster_data,
            np.zeros(len(cluster_data)),  # All in same cluster
            sensitive_columns,
            outcome_column,
            nonsensitive_columns
        )
        
        print(f"\nCluster {cluster_id} contributions:")
        cluster_cmi = cmi_per_cluster[cluster_id]
        for attr, value in contributions.items():
            print(f"  {attr}: {value:.4f} ({value/cluster_cmi*100:.1f}%)")
    
    # Step 7: Interaction information (if there are at least 2 sensitive attributes)
    if len(sensitive_columns) >= 2:
        print("\nInteraction information for high-discrimination clusters:")
        for cluster_id in high_discrim_clusters:
            cluster_data = train_data[train_data['cluster'] == cluster_id]
            
            interaction = interaction_information(
                cluster_data,
                np.zeros(len(cluster_data)),  # All in same cluster
                sensitive_columns[0],
                sensitive_columns[1],
                outcome_column,
                nonsensitive_columns
            )
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Interaction information: {interaction:.4f}")
            print(f"  {'Synergistic' if interaction > 0 else 'Redundant'} effect between {sensitive_columns[0]} and {sensitive_columns[1]}")
    
    # Step 8: Visualize results
    print("\nGenerating visualizations...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Plot CMI per cluster
    plt.figure(figsize=(10, 6))
    clusters = list(cmi_per_cluster.keys())
    cmi_values = list(cmi_per_cluster.values())
    
    # Sort by CMI value
    sorted_indices = np.argsort(cmi_values)[::-1]
    sorted_clusters = [clusters[i] for i in sorted_indices]
    sorted_values = [cmi_values[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_clusters, sorted_values)
    
    # Highlight high discrimination clusters
    for i, cluster in enumerate(sorted_clusters):
        if cluster in high_discrim_clusters:
            bars[i].set_color('red')
    
    plt.xlabel('Cluster')
    plt.ylabel('CMI')
    plt.title('Conditional Mutual Information by Cluster')
    plt.xticks(sorted_clusters)
    plt.grid(True, alpha=0.3)
    plt.savefig('results/cmi_by_cluster.png')
    plt.close()
    
    # Plot the clusters using PCA
    clustering.plot_clusters(
        X_train.values, 
        best_clusters, 
        pca_components=2, 
        save_path='results/cluster_visualization.png'
    )
    
    # Plot with sensitive attribute coloring
    if 'sex' in sensitive_columns:
        clustering.plot_clusters(
            X_train.values, 
            best_clusters, 
            pca_components=2, 
            sensitive_attr=sensitive_train['sex'].values,
            save_path='results/clusters_by_sex.png'
        )
    
    if 'race' in sensitive_columns:
        clustering.plot_clusters(
            X_train.values, 
            best_clusters, 
            pca_components=2, 
            sensitive_attr=sensitive_train['race'].values,
            save_path='results/clusters_by_race.png'
        )
    
    print("\nAnalysis complete. Results saved to the 'results' directory.")
    
    # Return results for potential further analysis
    return {
        'best_config': best_config,
        'best_cmi': best_cmi,
        'best_clusters': best_clusters,
        'cmi_per_cluster': cmi_per_cluster,
        'high_discrim_clusters': high_discrim_clusters,
        'results': results
    }

if __name__ == "__main__":
    main()
