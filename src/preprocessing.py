"""
Data preprocessing functions for discrimination detection.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def download_adult_dataset(save_dir='data'):
    """
    Download the Adult dataset from UCI Machine Learning Repository.
    
    Parameters:
        save_dir (str): Directory to save the downloaded files
        
    Returns:
        str: Path to the saved data file
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # URLs for the Adult dataset
    data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    names_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
    
    # Define save paths
    data_path = os.path.join(save_dir, 'adult.data')
    names_path = os.path.join(save_dir, 'adult.names')
    
    # Download only if files don't already exist
    if not os.path.exists(data_path):
        print(f"Downloading Adult dataset to {data_path}...")
        urllib.request.urlretrieve(data_url, data_path)
        print("Download complete.")
    else:
        print(f"Adult dataset already exists at {data_path}")
    
    # Download the names file for reference (contains attribute information)
    if not os.path.exists(names_path):
        print(f"Downloading Adult dataset metadata to {names_path}...")
        urllib.request.urlretrieve(names_url, names_path)
        print("Metadata download complete.")
    
    return data_path

def load_adult_dataset(filepath=None):
    """
    Load the Adult dataset from a CSV file.
    If filepath is None, download the dataset first.
    
    Parameters:
        filepath (str, optional): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    if filepath is None:
        filepath = download_adult_dataset()
    
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Load the dataset
    print(f"Loading Adult dataset from {filepath}...")
    data = pd.read_csv(filepath, names=column_names, sep=',', skipinitialspace=True)
    
    # Replace '?' with NaN
    data = data.replace('?', np.nan)
    
    # Print basic info
    print(f"Loaded dataset with {len(data)} samples and {len(column_names)} attributes")
    print(f"Columns: {column_names}")
    
    return data

def preprocess_adult_dataset(data):
    """
    Preprocess the Adult dataset according to the methodology in the paper.
    
    Parameters:
        data (pd.DataFrame): The raw Adult dataset
        
    Returns:
        tuple: (preprocessed_data, sensitive_columns, nonsensitive_columns, outcome_column)
    """
    print("Starting preprocessing...")
    
    # Make a copy to avoid modifying the original data
    data_clean = data.copy()
    
    # Remove samples with missing values
    original_size = len(data_clean)
    data_clean = data_clean.dropna()
    removed_samples = original_size - len(data_clean)
    print(f"Removed {removed_samples} samples with missing values ({removed_samples/original_size:.1%})")
    
    # Define sensitive attributes as per the paper
    original_sensitive_columns = ['race', 'sex']
    outcome_column = 'income'
    
    # Define non-sensitive attributes (all columns except sensitive and outcome)
    nonsensitive_columns = [col for col in data_clean.columns 
                           if col not in original_sensitive_columns and col != outcome_column]
    
    # Identify categorical and numerical columns
    categorical_columns = [col for col in data_clean.columns 
                          if data_clean[col].dtype == 'object']
    numerical_columns = [col for col in data_clean.columns 
                        if data_clean[col].dtype != 'object']
    
    print(f"Categorical columns: {categorical_columns}")
    print(f"Numerical columns: {numerical_columns}")
    
    # Create a copy of the data with sensitive attributes preserved
    # This is important for later analysis
    processed_data = data_clean.copy()
    
    # One-hot encode categorical features except outcome and sensitive attributes
    categorical_for_encoding = [col for col in categorical_columns 
                               if col != outcome_column and col not in original_sensitive_columns]
    
    # Initialize encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    
    # Fit and transform
    encoded_data = encoder.fit_transform(data_clean[categorical_for_encoding])
    
    # Get feature names
    feature_names = encoder.get_feature_names_out(categorical_for_encoding)
    
    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=feature_names,
        index=data_clean.index
    )
    
    # Normalize numerical features using min-max scaling as described in the paper
    normalized_numerical = data_clean[numerical_columns].copy()
    
    for col in numerical_columns:
        min_val = normalized_numerical[col].min()
        max_val = normalized_numerical[col].max()
        normalized_numerical[col] = (normalized_numerical[col] - min_val) / (max_val - min_val)
    
    # Preserve the original sensitive columns
    sensitive_df = data_clean[original_sensitive_columns].copy()
    
    # Combine all features
    processed_data = pd.concat([sensitive_df, encoded_df, normalized_numerical], axis=1)
    
    # Convert outcome column to binary (1 for '>50K', 0 for '<=50K')
    processed_data[outcome_column] = (data_clean[outcome_column].str.contains('>50K')).astype(int)
    
    # Keep track of sensitive columns (original ones)
    sensitive_columns = original_sensitive_columns
    
    # Update nonsensitive columns to match the processed data
    new_nonsensitive_columns = list(encoded_df.columns) + list(normalized_numerical.columns)
    
    print(f"Preprocessing complete. Processed data shape: {processed_data.shape}")
    print(f"Sensitive columns: {sensitive_columns}")
    print(f"Outcome column: {outcome_column}")
    
    return processed_data, sensitive_columns, new_nonsensitive_columns, outcome_column

def split_data_by_sensitive_attributes(data, sensitive_columns, outcome_column):
    """
    Split data into groups based on sensitive attributes for analysis.
    
    Parameters:
        data (pd.DataFrame): The preprocessed data
        sensitive_columns (list): List of sensitive attribute columns
        outcome_column (str): Name of the outcome column
        
    Returns:
        dict: Dictionary of data subgroups, keyed by sensitive attribute values
    """
    # Verify that sensitive columns exist in the data
    for col in sensitive_columns:
        if col not in data.columns:
            raise ValueError(f"Sensitive column '{col}' not found in the data")
    
    # Create a combined sensitive attribute column
    if len(sensitive_columns) > 1:
        data['sensitive_group'] = data[sensitive_columns].apply(
            lambda row: '_'.join(row.astype(str)),
            axis=1
        )
        groupby_col = 'sensitive_group'
    else:
        groupby_col = sensitive_columns[0]
    
    # Group by the sensitive attribute(s)
    groups = {}
    for group_name, group_data in data.groupby(groupby_col):
        groups[group_name] = group_data
    
    # Print group information
    print(f"Split data into {len(groups)} groups based on sensitive attributes:")
    for group_name, group_data in groups.items():
        positive_rate = group_data[outcome_column].mean()
        print(f"  Group '{group_name}': {len(group_data)} samples, positive rate: {positive_rate:.2f}")
    
    return groups

# Example usage
if __name__ == "__main__":
    # Download and load the dataset
    data = load_adult_dataset()
    
    # Display some basic information
    print("\nData sample:")
    print(data.head())
    
    print("\nData info:")
    print(data.info())
    
    print("\nSummary statistics:")
    print(data.describe())
    
    # Preprocess the data
    processed_data, sensitive_columns, nonsensitive_columns, outcome_column = preprocess_adult_dataset(data)
    
    print("\nPreprocessed data sample:")
    print(processed_data.head())
    
    # Split data by sensitive attributes
    groups = split_data_by_sensitive_attributes(processed_data, sensitive_columns, outcome_column)
