"""
Data loading and preprocessing functions for discrimination detection framework.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_adult_dataset(data_path):
    """
    Load the Adult dataset from a file path.
    
    Parameters:
    -----------
    data_path : str
        Path to the Adult dataset file
        
    Returns:
    --------
    pandas.DataFrame
        The loaded dataset
    """
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    data = pd.read_csv(data_path, header=None, names=column_names, 
                       sep=',\s*', engine='python', na_values='?')
    
    # Clean up data
    data = data.dropna()
    
    # Clean the income column
    data['income'] = data['income'].str.strip()
    data['income'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
    
    return data

def preprocess_adult_dataset(data):
    """
    Preprocess the Adult dataset for discrimination detection analysis.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The Adult dataset
        
    Returns:
    --------
    tuple
        (processed_data, sensitive_columns, nonsensitive_columns, outcome_column)
    """
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Define sensitive and outcome columns
    sensitive_columns = ['sex', 'race']
    outcome_column = 'income'
    
    # Convert categorical variables to one-hot encoding, excluding sensitive and outcome
    categorical_columns = [col for col in processed_data.columns 
                          if processed_data[col].dtype == 'object'
                          and col not in sensitive_columns + [outcome_column]]
    
    processed_data = pd.get_dummies(processed_data, columns=categorical_columns, drop_first=True)
    
    # Encode sensitive attributes
    for col in sensitive_columns:
        # Keep original column
        processed_data[f'{col}_orig'] = processed_data[col]
        
        # Encode categorical values
        unique_values = processed_data[col].unique()
        value_to_int = {val: i for i, val in enumerate(unique_values)}
        processed_data[col] = processed_data[col].map(value_to_int)
    
    # Normalize numerical features
    numerical_columns = [col for col in processed_data.columns 
                        if processed_data[col].dtype in ['int64', 'float64']
                        and col not in sensitive_columns + [outcome_column]]
    
    scaler = StandardScaler()
    processed_data[numerical_columns] = scaler.fit_transform(processed_data[numerical_columns])
    
    # Get all non-sensitive columns (excluding outcome and sensitive attributes)
    nonsensitive_columns = [col for col in processed_data.columns 
                           if col not in sensitive_columns + [outcome_column]]
    
    return processed_data, sensitive_columns, nonsensitive_columns, outcome_column