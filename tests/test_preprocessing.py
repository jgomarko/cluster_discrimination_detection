"""
Tests for the preprocessing module.
"""

import unittest
import os
import pandas as pd
import numpy as np
from tempfile import NamedTemporaryFile

from src.preprocessing import load_adult_dataset, preprocess_adult_dataset

class TestPreprocessing(unittest.TestCase):
    """Test cases for the preprocessing module."""
    
    def setUp(self):
        """Set up test data."""
        # Create a temporary file with adult dataset format
        self.temp_file = NamedTemporaryFile(delete=False)
        
        # Write sample data to the temporary file
        with open(self.temp_file.name, 'w') as f:
            f.write("39, Private, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K\n")
            f.write("50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K\n")
            f.write("38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K\n")
            f.write("53, Private, 234721, 11th, 7, Married-civ-spouse, Handlers-cleaners, Husband, Black, Male, 0, 0, 40, United-States, <=50K\n")
            f.write("28, Private, 338409, Bachelors, 13, Married-civ-spouse, Prof-specialty, Wife, Black, Female, 0, 0, 40, Cuba, <=50K\n")
            f.write("37, Private, 284582, Masters, 14, Married-civ-spouse, Exec-managerial, Wife, White, Female, 0, 0, 40, United-States, <=50K\n")
            f.write("49, Private, 160187, 9th, 5, Married-spouse-absent, Other-service, Not-in-family, Black, Female, 0, 0, 16, Jamaica, <=50K\n")
            f.write("52, Self-emp-not-inc, 209642, HS-grad, 9, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 45, United-States, >50K\n")
    
    def tearDown(self):
        """Clean up after tests."""
        os.unlink(self.temp_file.name)
    
    def test_load_adult_dataset(self):
        """Test loading the Adult dataset."""
        data = load_adult_dataset(self.temp_file.name)
        
        # Check that the data is a DataFrame
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check that the data has the expected columns
        expected_columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        for col in expected_columns:
            self.assertIn(col, data.columns)
        
        # Check that the correct number of rows were loaded
        self.assertEqual(len(data), 8)
        
        # Check that the income column was processed correctly
        self.assertTrue(all(income in [0, 1] for income in data['income']))
    
    def test_preprocess_adult_dataset(self):
        """Test preprocessing the Adult dataset."""
        data = load_adult_dataset(self.temp_file.name)
        processed_data, sensitive_columns, nonsensitive_columns, outcome_column = preprocess_adult_dataset(data)
        
        # Check that the processed data is a DataFrame
        self.assertIsInstance(processed_data, pd.DataFrame)
        
        # Check that sensitive columns are identified correctly
        self.assertEqual(set(sensitive_columns), {'sex', 'race'})
        
        # Check that the outcome column is identified correctly
        self.assertEqual(outcome_column, 'income')
        
        # Check that nonsensitive columns don't include sensitive or outcome columns
        for col in sensitive_columns + [outcome_column]:
            self.assertNotIn(col, nonsensitive_columns)
        
        # Check that one-hot encoding was applied for categorical variables
        self.assertGreater(len(processed_data.columns), len(data.columns))
        
        # Check that numerical features were standardized
        age_column = processed_data['age']
        self.assertAlmostEqual(np.mean(age_column), 0, delta=0.5)
        self.assertAlmostEqual(np.std(age_column), 1, delta=0.5)

if __name__ == '__main__':
    unittest.main()