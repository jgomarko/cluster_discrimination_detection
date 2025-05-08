"""
Clustering algorithms for discrimination detection framework.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

class MultiClusteringAlgorithm:
    """
    Implements multiple clustering algorithms for discrimination detection.
    """
    
    def __init__(self):
        """Initialize the clustering algorithms class."""
        self.clusters = None
        self.algorithm = None
        self.n_clusters = None
        
    def fit(self, X, algorithm='kmeans', n_clusters=8, random_state=42):
        """
        Fit a clustering algorithm to the data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The features to cluster
        algorithm : str
            Clustering algorithm to use ('kmeans', 'gmm', 'spectral', 'ensemble')
        n_clusters : int
            Number of clusters to create
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        numpy.ndarray
            Cluster assignments for each sample
        """
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        
        if algorithm == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            self.clusters = kmeans.fit_predict(X)
            
        elif algorithm == 'gmm':
            gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
            self.clusters = gmm.fit_predict(X)
            
        elif algorithm == 'spectral':
            # Spectral clustering can be slow for large datasets
            spectral = SpectralClustering(n_clusters=n_clusters, 
                                         random_state=random_state,
                                         n_jobs=-1,
                                         affinity='nearest_neighbors')
            self.clusters = spectral.fit_predict(X)
            
        elif algorithm == 'ensemble':
            self.clusters = self.ensemble_clustering(X, n_clusters, random_state)
            
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
            
        return self.clusters
    
    def ensemble_clustering(self, X, n_clusters=8, random_state=42):
        """
        Create an ensemble of multiple clustering algorithms.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The features to cluster
        n_clusters : int
            Number of clusters to create
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        numpy.ndarray
            Cluster assignments from the ensemble
        """
        # Run multiple clustering algorithms
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        kmeans_clusters = kmeans.fit_predict(X)
        
        gmm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gmm_clusters = gmm.fit_predict(X)
        
        # Create a co-occurrence matrix
        n_samples = X.shape[0]
        co_occurrence = np.zeros((n_samples, n_samples))
        
        # Update co-occurrence based on kmeans
        for i in range(n_clusters):
            idx = np.where(kmeans_clusters == i)[0]
            for j in idx:
                for k in idx:
                    co_occurrence[j, k] += 1
                    
        # Update co-occurrence based on gmm
        for i in range(n_clusters):
            idx = np.where(gmm_clusters == i)[0]
            for j in idx:
                for k in idx:
                    co_occurrence[j, k] += 1
        
        # Normalize
        co_occurrence /= 2
        
        # Use spectral clustering on the co-occurrence matrix
        spectral = SpectralClustering(n_clusters=n_clusters, 
                                     random_state=random_state,
                                     affinity='precomputed')
        ensemble_clusters = spectral.fit_predict(co_occurrence)
        
        return ensemble_clusters
    
    def evaluate_clusters(self, X, clusters=None):
        """
        Evaluate the quality of the clusters.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The features used for clustering
        clusters : numpy.ndarray, optional
            Cluster assignments. If None, use the stored clusters.
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        if clusters is None:
            if self.clusters is None:
                raise ValueError("No clusters found. Call fit() first.")
            clusters = self.clusters
            
        # Calculate silhouette score
        silhouette = silhouette_score(X, clusters)
        
        # Calculate Davies-Bouldin index
        davies_bouldin = davies_bouldin_score(X, clusters)
        
        return {
            'silhouette_score': silhouette,
            'davies_bouldin_index': davies_bouldin
        }
    
    def find_optimal_k(self, X, algorithm='kmeans', k_range=range(2, 11), random_state=42):
        """
        Find the optimal number of clusters using the Gap statistic.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The features to cluster
        algorithm : str
            Clustering algorithm to use
        k_range : range
            Range of k values to try
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (optimal_k, gap_values)
        """
        print("Finding optimal number of clusters using Gap statistic...")
        
        gap_values = []
        
        # Calculate dispersion for actual data
        wks = []
        for k in k_range:
            print(f"Testing k={k}...")
            
            # Fit clustering algorithm
            if algorithm == 'kmeans':
                kmeans = KMeans(n_clusters=k, random_state=random_state)
                clusters = kmeans.fit_predict(X)
            elif algorithm == 'gmm':
                gmm = GaussianMixture(n_components=k, random_state=random_state)
                clusters = gmm.fit_predict(X)
            else:
                raise ValueError(f"Unsupported algorithm for gap statistic: {algorithm}")
            
            # Calculate within-cluster dispersion
            wk = self._calculate_dispersion(X, clusters, k)
            wks.append(wk)
            
            # Generate reference datasets (uniform random)
            n_refs = 5  # Use more for production
            ref_dispersions = []
            
            for i in range(n_refs):
                # Create reference dataset with uniform random distribution
                X_ref = np.random.uniform(
                    low=np.min(X, axis=0),
                    high=np.max(X, axis=0),
                    size=X.shape
                )
                
                # Cluster the reference data
                if algorithm == 'kmeans':
                    kmeans_ref = KMeans(n_clusters=k, random_state=random_state+i)
                    clusters_ref = kmeans_ref.fit_predict(X_ref)
                elif algorithm == 'gmm':
                    gmm_ref = GaussianMixture(n_components=k, random_state=random_state+i)
                    clusters_ref = gmm_ref.fit_predict(X_ref)
                
                # Calculate dispersion for reference
                ref_dispersions.append(self._calculate_dispersion(X_ref, clusters_ref, k))
            
            # Calculate gap statistic
            gap = np.mean(np.log(ref_dispersions)) - np.log(wk)
            gap_values.append(gap)
            print(f"  k={k}, gap={gap:.4f}")
        
        # Find optimal k (highest gap)
        optimal_k = k_range[np.argmax(gap_values)]
        
        return optimal_k, gap_values
    
    def _calculate_dispersion(self, X, clusters, k):
        """
        Calculate within-cluster dispersion.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        clusters : numpy.ndarray
            Cluster assignments
        k : int
            Number of clusters
            
        Returns:
        --------
        float
            Within-cluster dispersion
        """
        dispersion = 0
        
        for i in range(k):
            cluster_points = X[clusters == i]
            
            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                d = np.sum(np.square(cluster_points - centroid))
                dispersion += d
        
        return dispersion
    
    def plot_gap_statistic(self, k_range, gap_values):
        """
        Plot the Gap statistic for different k values.
        
        Parameters:
        -----------
        k_range : range
            Range of k values
        gap_values : list
            Gap values for each k
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, gap_values, 'bo-')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Gap statistic')
        plt.title('Gap Statistic vs. Number of Clusters')
        plt.grid(True, alpha=0.3)
        
        # Mark the optimal k
        optimal_k = k_range[np.argmax(gap_values)]
        max_gap = max(gap_values)
        plt.plot(optimal_k, max_gap, 'ro', markersize=10)
        plt.annotate(f'Optimal k = {optimal_k}', 
                   (optimal_k, max_gap),
                   xytext=(5, -15),
                   textcoords='offset points')
        
        plt.savefig('gap_statistic.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_clusters(self, X, clusters, sensitive_attr=None):
        """
        Visualize the clusters using PCA for dimensionality reduction.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        clusters : numpy.ndarray
            Cluster assignments
        sensitive_attr : numpy.ndarray, optional
            Sensitive attribute values for coloring points
        """
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        if sensitive_attr is not None:
            # Color by sensitive attribute
            unique_values = np.unique(sensitive_attr)
            
            for value in unique_values:
                mask = sensitive_attr == value
                plt.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    label=f'Value={value}',
                    alpha=0.6
                )
                
            plt.title('PCA Visualization colored by Sensitive Attribute')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f}%)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f}%)')
            plt.legend()
            
        else:
            # Color by cluster
            unique_clusters = np.unique(clusters)
            
            for cluster_id in unique_clusters:
                mask = clusters == cluster_id
                plt.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    label=f'Cluster {cluster_id}',
                    alpha=0.6
                )
                
            plt.title('PCA Visualization of Clusters')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f}%)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f}%)')
            plt.legend()
            
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        if sensitive_attr is not None:
            plt.savefig('clusters_by_sensitive_attr.png', dpi=300, bbox_inches='tight')
        else:
            plt.savefig('clusters.png', dpi=300, bbox_inches='tight')
            
        plt.show()