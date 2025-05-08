"""
Tests for the CMI module.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.cmi import calculate_cmi, calculate_cmi_per_cluster, hierarchical_cmi_decomposition

class TestCMIFunctions(unittest.TestCase):
    """Test cases for the CMI module functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=1000,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            n_classes=2,
            random_state=42
        )
        
        # Create sensitive attributes with bias
        sex = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])
        race = np.random.choice([0, 1, 2], size=1000, p=[0.5, 0.3, 0.2])
        
        # Create biased outcome based on sensitive attributes
        # Higher probability of positive outcome for sex=1 and race=0
        bias_factors = 0.2 * (sex == 1) + 0.1 * (race == 0)
        y_biased = np.zeros_like(y)
        for i in range(len(y)):
            prob_one = 0.5 + bias_factors[i]
            y_biased[i] = np.random.choice([0, 1], p=[1-prob_one, prob_one])
        
        # Create DataFrame
        self.data = pd.DataFrame({
            'feature1': X[:, 0],
            'feature2': X[:, 1],
            'feature3': X[:, 2],
            'feature4': X[:, 3],
            'feature5': X[:, 4],
            'sex': sex,
            'race': race,
            'outcome': y_biased
        })
        
        # Create clusters
        self.clusters = np.random.choice([0, 1, 2, 3], size=1000, p=[0.4, 0.3, 0.2, 0.1])
        
        # Define columns
        self.sensitive_columns = ['sex', 'race']
        self.outcome_column = 'outcome'
        self.nonsensitive_columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    
    def test_calculate_cmi(self):
        """Test calculate_cmi function."""
        cmi = calculate_cmi(
            self.data,
            self.clusters,
            self.sensitive_columns,
            self.outcome_column,
            self.nonsensitive_columns
        )
        
        # CMI should be a non-negative float
        self.assertIsInstance(cmi, float)
        self.assertGreaterEqual(cmi, 0)
    
    def test_calculate_cmi_per_cluster(self):
        """Test calculate_cmi_per_cluster function."""
        cmi_per_cluster = calculate_cmi_per_cluster(
            self.data,
            self.clusters,
            self.sensitive_columns,
            self.outcome_column,
            self.nonsensitive_columns
        )
        
        # Result should be a dictionary with cluster IDs as keys
        self.assertIsInstance(cmi_per_cluster, dict)
        
        # Each cluster should have a CMI value
        for cluster_id in np.unique(self.clusters):
            self.assertIn(cluster_id, cmi_per_cluster)
            self.assertIsInstance(cmi_per_cluster[cluster_id], float)
            self.assertGreaterEqual(cmi_per_cluster[cluster_id], 0)
    
    def test_hierarchical_cmi_decomposition(self):
        """Test hierarchical_cmi_decomposition function."""
        contributions = hierarchical_cmi_decomposition(
            self.data,
            self.clusters,
            self.sensitive_columns,
            self.outcome_column,
            self.nonsensitive_columns
        )
        
        # Result should be a dictionary with sensitive attributes as keys
        self.assertIsInstance(contributions, dict)
        
        # Each sensitive attribute should have a contribution value
        for sensitive_attr in self.sensitive_columns:
            self.assertIn(sensitive_attr, contributions)
            self.assertIsInstance(contributions[sensitive_attr], float)
            self.assertGreaterEqual(contributions[sensitive_attr], 0)

if __name__ == '__main__':
    unittest.main()