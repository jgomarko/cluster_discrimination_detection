"""
Tests for the clustering module.
"""

import unittest
import numpy as np
from sklearn.datasets import make_blobs

from src.clustering import MultiClusteringAlgorithm

class TestClustering(unittest.TestCase):
    """Test cases for the clustering module."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic clustered data
        self.X, self.y = make_blobs(
            n_samples=300,
            n_features=5,
            centers=4,
            random_state=42
        )
        
        self.clustering = MultiClusteringAlgorithm()
    
    def test_kmeans_clustering(self):
        """Test k-means clustering."""
        clusters = self.clustering.fit(self.X, algorithm='kmeans', n_clusters=4)
        
        # Check that the number of clusters is correct
        self.assertEqual(len(np.unique(clusters)), 4)
        
        # Check that every sample is assigned a cluster
        self.assertEqual(len(clusters), len(self.X))
    
    def test_gmm_clustering(self):
        """Test Gaussian Mixture Model clustering."""
        clusters = self.clustering.fit(self.X, algorithm='gmm', n_clusters=4)
        
        # Check that the number of clusters is correct
        self.assertEqual(len(np.unique(clusters)), 4)
        
        # Check that every sample is assigned a cluster
        self.assertEqual(len(clusters), len(self.X))
    
    def test_ensemble_clustering(self):
        """Test ensemble clustering."""
        clusters = self.clustering.ensemble_clustering(self.X, n_clusters=4)
        
        # Check that the number of clusters is correct
        self.assertEqual(len(np.unique(clusters)), 4)
        
        # Check that every sample is assigned a cluster
        self.assertEqual(len(clusters), len(self.X))
    
    def test_find_optimal_k(self):
        """Test finding optimal k with the Gap statistic."""
        # Use a smaller subset for faster testing
        X_sample = self.X[:100]
        
        # Test with a range of k values
        optimal_k, gap_values = self.clustering.find_optimal_k(
            X_sample, 
            algorithm='kmeans', 
            k_range=range(2, 5),
            random_state=42
        )
        
        # Check that optimal_k is in the expected range
        self.assertTrue(2 <= optimal_k <= 4)
        
        # Check that gap_values has the right length
        self.assertEqual(len(gap_values), 3)  # for k = 2, 3, 4
    
    def test_evaluate_clusters(self):
        """Test cluster evaluation metrics."""
        # First cluster the data
        clusters = self.clustering.fit(self.X, algorithm='kmeans', n_clusters=4)
        
        # Evaluate the clusters
        metrics = self.clustering.evaluate_clusters(self.X, clusters)
        
        # Check that the metrics are present
        self.assertIn('silhouette_score', metrics)
        self.assertIn('davies_bouldin_index', metrics)
        
        # Check that the metrics are within reasonable ranges
        self.assertTrue(-1 <= metrics['silhouette_score'] <= 1)
        self.assertTrue(metrics['davies_bouldin_index'] >= 0)

if __name__ == '__main__':
    unittest.main()