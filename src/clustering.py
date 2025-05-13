"""
Clustering algorithms for discrimination detection.

This module implements multiple clustering algorithms:
1. K-means clustering
2. Gaussian Mixture Models (GMM)
3. Spectral clustering
4. Ensemble clustering approach

It also provides methods to find the optimal number of clusters
and evaluate clustering quality.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class MultiClusteringAlgorithm:
    """
    Class that implements multiple clustering algorithms and ensemble method
    as described in the paper.
    """
    
    def __init__(self):
        """Initialize the clustering algorithms."""
        self.algorithms = {
            'kmeans': KMeans,
            'gmm': GaussianMixture,
            'spectral': SpectralClustering
        }
        self.cluster_labels = None
        self.best_k = None
        self.best_algorithm = None
        self.models = {}
    
    def fit(self, data, algorithm='kmeans', n_clusters=7, random_state=42):
        """
        Fit the specified clustering algorithm.
        
        Parameters:
            data (np.ndarray): The data to cluster
            algorithm (str): The clustering algorithm to use ('kmeans', 'gmm', or 'spectral')
            n_clusters (int): The number of clusters
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: Cluster labels
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Algorithm {algorithm} not supported. Choose from {list(self.algorithms.keys())}")
        
        # Get the algorithm class
        Algorithm = self.algorithms[algorithm]
        
        # Initialize and fit the algorithm with appropriate parameters
        if algorithm == 'kmeans':
            # Use k-means++ initialization as described in the paper
            model = Algorithm(
                n_clusters=n_clusters, 
                init='k-means++',
                n_init=10,
                random_state=random_state
            )
            model.fit(data)
            self.cluster_labels = model.labels_
            self.models[algorithm] = model
            
        elif algorithm == 'gmm':
            # GMM with full covariance matrix
            model = Algorithm(
                n_components=n_clusters, 
                covariance_type='full',
                random_state=random_state
            )
            model.fit(data)
            self.cluster_labels = model.predict(data)
            self.models[algorithm] = model
            
        elif algorithm == 'spectral':
            # Use RBF kernel with median heuristic for sigma
            # Calculate pairwise distances
            distances = pdist(data)
            sigma = np.median(distances)
            
            model = Algorithm(
                n_clusters=n_clusters,
                affinity='rbf',
                gamma=1.0 / (2 * sigma**2),
                random_state=random_state
            )
            self.cluster_labels = model.fit_predict(data)
            self.models[algorithm] = model
        
        self.best_algorithm = algorithm
        self.best_k = n_clusters
        
        return self.cluster_labels
    
    def ensemble_clustering(self, data, n_clusters=7, random_state=42):
        """
        Perform ensemble clustering using all available algorithms.
        
        Parameters:
            data (np.ndarray): The data to cluster
            n_clusters (int): The number of clusters
            random_state (int): Random seed for reproducibility
            
        Returns:
            np.ndarray: Ensemble cluster labels
        """
        print("Running ensemble clustering...")
        
        # Get cluster labels from each algorithm
        kmeans_labels = self.fit(data, 'kmeans', n_clusters, random_state)
        gmm_labels = self.fit(data, 'gmm', n_clusters, random_state)
        spectral_labels = self.fit(data, 'spectral', n_clusters, random_state)
        
        # Create co-association matrix
        n_samples = len(data)
        co_association = np.zeros((n_samples, n_samples))
        
        print("Building co-association matrix...")
        # Fill co-association matrix
        for i in range(n_samples):
            for j in range(i, n_samples):  # Use symmetry for efficiency
                # Count how many algorithms place samples i and j in the same cluster
                same_cluster_count = 0
                if kmeans_labels[i] == kmeans_labels[j]:
                    same_cluster_count += 1
                if gmm_labels[i] == gmm_labels[j]:
                    same_cluster_count += 1
                if spectral_labels[i] == spectral_labels[j]:
                    same_cluster_count += 1
                
                co_association[i, j] = same_cluster_count / 3.0
                co_association[j, i] = co_association[i, j]  # Symmetric matrix
        
        print("Applying hierarchical clustering to co-association matrix...")
        # In the paper, they use hierarchical clustering with average linkage
        # For simplicity, we'll use K-means here, but you can extend this with actual
        # hierarchical clustering implementation if needed
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Convert co-association to distance matrix
        distance_matrix = 1 - co_association
        
        # Perform hierarchical clustering with average linkage
        condensed_distance = squareform(distance_matrix)
        Z = linkage(condensed_distance, method='average')
        
        # Cut the tree to get n_clusters
        self.cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-based indexing
        
        self.best_algorithm = 'ensemble'
        self.best_k = n_clusters
        
        print(f"Ensemble clustering complete. Found {len(np.unique(self.cluster_labels))} clusters.")
        return self.cluster_labels
    
    def evaluate_clusters(self, data, labels=None):
        """
        Evaluate clustering quality using silhouette score and Davies-Bouldin index.
        
        Parameters:
            data (np.ndarray): The data
            labels (np.ndarray, optional): Cluster labels. If None, use self.cluster_labels
            
        Returns:
            dict: Evaluation metrics
        """
        if labels is None:
            if self.cluster_labels is None:
                raise ValueError("No cluster labels available. Run fit() or ensemble_clustering() first.")
            labels = self.cluster_labels
        
        # Check if there's more than one cluster
        if len(np.unique(labels)) < 2:
            return {
                'silhouette_score': -1,  # Invalid score
                'davies_bouldin_index': float('inf')  # Worst possible score
            }
        
        # Calculate silhouette score
        silhouette = silhouette_score(data, labels)
        
        # Calculate Davies-Bouldin index
        davies_bouldin = davies_bouldin_score(data, labels)
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin
        }
    
    def find_optimal_k(self, data, algorithm='kmeans', k_range=range(2, 16), random_state=42):
        """
        Find optimal number of clusters using the Gap statistic as described in the paper.
        
        Parameters:
            data (np.ndarray): The data to cluster
            algorithm (str): The clustering algorithm to use
            k_range (range): Range of k values to try
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (optimal_k, gap_values)
        """
        print(f"Finding optimal number of clusters using Gap statistic...")
        gap_values = []
        
        for k in k_range:
            print(f"Testing k={k}...")
            # Fit model with current k
            self.fit(data, algorithm, k, random_state)
            
            # Calculate within-cluster dispersion
            wk = self._calculate_dispersion(data, self.cluster_labels)
            
            # Generate reference data and calculate expected dispersion
            reference_dispersions = []
            
            # Monte Carlo simulation with multiple reference datasets
            for i in range(10):  # Use 10 reference datasets as mentioned in the paper
                # Generate reference data from uniform distribution
                reference_data = self._generate_reference_data(data)
                
                # Cluster reference data
                ref_labels = self.fit(reference_data, algorithm, k, random_state + i)
                
                # Calculate dispersion for reference data
                ref_dispersion = self._calculate_dispersion(reference_data, ref_labels)
                reference_dispersions.append(np.log(ref_dispersion))
            
            # Calculate gap statistic
            expected_log_wk = np.mean(reference_dispersions)
            gap = expected_log_wk - np.log(wk)
            gap_values.append(gap)
            
            print(f"  k={k}, gap={gap:.4f}")
        
        # Find optimal k (maximum gap)
        optimal_k = k_range[np.argmax(gap_values)]
        self.best_k = optimal_k
        
        print(f"Optimal number of clusters: {optimal_k}")
        return optimal_k, gap_values
    
    def _calculate_dispersion(self, data, labels):
        """
        Calculate within-cluster dispersion as defined in the Gap statistic.
        
        Parameters:
            data (np.ndarray): The data
            labels (np.ndarray): Cluster labels
            
        Returns:
            float: Within-cluster dispersion
        """
        unique_labels = np.unique(labels)
        total_dispersion = 0
        
        for label in unique_labels:
            # Get points in this cluster
            cluster_points = data[labels == label]
            
            if len(cluster_points) <= 1:
                continue
            
            # Calculate pairwise distances within cluster
            cluster_distances = pdist(cluster_points)
            
            # Sum of squared distances
            total_dispersion += np.sum(cluster_distances**2) / (2 * len(cluster_points))
        
        return total_dispersion
    
    def _generate_reference_data(self, data):
        """
        Generate reference data from uniform distribution within data range.
        
        Parameters:
            data (np.ndarray): The original data
            
        Returns:
            np.ndarray: Reference data with same shape as original
        """
        # Find min and max for each feature
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # Generate uniform random data in the same range
        reference_data = np.random.uniform(
            low=min_vals,
            high=max_vals,
            size=data.shape
        )
        
        return reference_data
    
    def find_optimal_k_elbow(self, data, algorithm='kmeans', k_range=range(2, 16), random_state=42):
        """
        Find optimal number of clusters using the elbow method.
        
        Parameters:
            data (np.ndarray): The data to cluster
            algorithm (str): The clustering algorithm to use
            k_range (range): Range of k values to try
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (suggested_k, inertia_values)
        """
        inertia_values = []
        
        for k in k_range:
            if algorithm == 'kmeans':
                model = KMeans(n_clusters=k, random_state=random_state)
                model.fit(data)
                inertia_values.append(model.inertia_)
            else:
                # For other algorithms, use silhouette score
                self.fit(data, algorithm, k, random_state)
                score = silhouette_score(data, self.cluster_labels)
                inertia_values.append(-score)  # Negative because we want to minimize
        
        # Find elbow point (this is a simple heuristic, can be improved)
        diffs = np.diff(inertia_values)
        diffs_of_diffs = np.diff(diffs)
        suggested_k = k_range[np.argmax(diffs_of_diffs) + 2]
        
        return suggested_k, inertia_values
    
    def plot_gap_statistic(self, k_range, gap_values, save_path=None):
        """
        Plot Gap statistic values for different k.
        
        Parameters:
            k_range (range): Range of k values
            gap_values (list): Corresponding Gap values
            save_path (str, optional): Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, gap_values, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Gap statistic')
        plt.title('Gap Statistic by Number of Clusters')
        plt.grid(True)
        
        # Mark the optimal k
        optimal_k = k_range[np.argmax(gap_values)]
        max_gap = max(gap_values)
        plt.plot([optimal_k], [max_gap], 'ro', markersize=10)
        plt.annotate(f'Optimal k = {optimal_k}', 
                     xy=(optimal_k, max_gap),
                     xytext=(optimal_k + 1, max_gap),
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        if save_path:
            plt.savefig(save_path)
            print(f"Gap statistic plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_cluster_composition(self, data, sensitive_columns, outcome_column):
        """
        Analyze the composition of clusters with respect to sensitive attributes.
        
        Parameters:
            data (pd.DataFrame): The original data with sensitive attributes
            sensitive_columns (list): List of sensitive attribute columns
            outcome_column (str): Name of the outcome column
            
        Returns:
            pd.DataFrame: Analysis results
        """
        if self.cluster_labels is None:
            raise ValueError("No cluster labels available. Run fit() or ensemble_clustering() first.")
        
        # Add cluster labels to data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = self.cluster_labels
        
        # Initialize results
        results = []
        
        # For each cluster
        for cluster in np.unique(self.cluster_labels):
            cluster_data = data_with_clusters[data_with_clusters['cluster'] == cluster]
            
            # Overall cluster info
            cluster_size = len(cluster_data)
            cluster_outcome_rate = cluster_data[outcome_column].mean()
            
            cluster_info = {
                'cluster': cluster,
                'size': cluster_size,
                'outcome_rate': cluster_outcome_rate
            }
            
            # For each sensitive attribute
            for col in sensitive_columns:
                # Get distribution of this attribute in the cluster
                value_counts = cluster_data[col].value_counts(normalize=True)
                
                for value, proportion in value_counts.items():
                    # Get outcome rate for this subgroup
                    subgroup = cluster_data[cluster_data[col] == value]
                    subgroup_outcome_rate = subgroup[outcome_column].mean() if len(subgroup) > 0 else 0
                    
                    # Add to cluster info
                    cluster_info[f'{col}_{value}_proportion'] = proportion
                    cluster_info[f'{col}_{value}_outcome_rate'] = subgroup_outcome_rate
            
            results.append(cluster_info)
        
        return pd.DataFrame(results)
    
    def plot_clusters(self, data, labels=None, pca_components=2, sensitive_attr=None, save_path=None):
        """
        Visualize clusters using PCA for dimensionality reduction.
        
        Parameters:
            data (np.ndarray): The data
            labels (np.ndarray, optional): Cluster labels. If None, use self.cluster_labels
            pca_components (int): Number of PCA components for visualization
            sensitive_attr (np.ndarray, optional): Sensitive attribute values for color coding
            save_path (str, optional): Path to save the figure
        """
        if labels is None:
            if self.cluster_labels is None:
                raise ValueError("No cluster labels available. Run fit() or ensemble_clustering() first.")
            labels = self.cluster_labels
        
        # Reduce dimensionality for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=pca_components)
        reduced_data = pca.fit_transform(data)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # If we want to color by sensitive attribute
        if sensitive_attr is not None:
            # Plot points colored by sensitive attribute
            unique_attrs = np.unique(sensitive_attr)
            for attr in unique_attrs:
                mask = sensitive_attr == attr
                plt.scatter(
                    reduced_data[mask, 0], 
                    reduced_data[mask, 1],
                    label=f'Attr: {attr}',
                    alpha=0.6
                )
            plt.title('Data Colored by Sensitive Attribute')
            
        else:
            # Plot points colored by cluster
            unique_clusters = np.unique(labels)
            for cluster in unique_clusters:
                mask = labels == cluster
                plt.scatter(
                    reduced_data[mask, 0], 
                    reduced_data[mask, 1],
                    label=f'Cluster {cluster}',
                    alpha=0.6
                )
            plt.title(f'Cluster Visualization ({self.best_algorithm}, k={self.best_k})')
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Cluster visualization saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return reduced_data

# Example usage
if __name__ == "__main__":
    # Generate sample data with two clusters
    np.random.seed(42)
    
    # Create two clusters
    cluster1 = np.random.randn(100, 2) + np.array([2, 2])
    cluster2 = np.random.randn(100, 2) + np.array([-2, -2])
    
    # Combine into one dataset
    X = np.vstack([cluster1, cluster2])
    
    # Create instance
    clustering = MultiClusteringAlgorithm()
    
    # Try different algorithms
    print("\nK-means clustering:")
    kmeans_labels = clustering.fit(X, algorithm='kmeans', n_clusters=2)
    print("K-means evaluation:", clustering.evaluate_clusters(X, kmeans_labels))
    
    print("\nGMM clustering:")
    gmm_labels = clustering.fit(X, algorithm='gmm', n_clusters=2)
    print("GMM evaluation:", clustering.evaluate_clusters(X, gmm_labels))
    
    print("\nSpectral clustering:")
    spectral_labels = clustering.fit(X, algorithm='spectral', n_clusters=2)
    print("Spectral evaluation:", clustering.evaluate_clusters(X, spectral_labels))
    
    print("\nEnsemble clustering:")
    ensemble_labels = clustering.ensemble_clustering(X, n_clusters=2)
    print("Ensemble evaluation:", clustering.evaluate_clusters(X, ensemble_labels))
    
    # Find optimal k
    optimal_k, gap_values = clustering.find_optimal_k(X, algorithm='kmeans', k_range=range(1, 6))
    print(f"\nOptimal number of clusters (Gap statistic): {optimal_k}")
    
    # Plot Gap statistic
    clustering.plot_gap_statistic(range(1, 6), gap_values)
    
    # Visualize final clusters
    clustering.plot_clusters(X)
