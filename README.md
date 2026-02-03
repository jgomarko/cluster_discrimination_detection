# Cluster-Based Discrimination Detection Framework

A comprehensive framework for detecting and mitigating hidden discrimination in datasets using a clustering-based approach with Conditional Mutual Information (CMI).

## Overview

This framework identifies specific subgroups (clusters) in data where discrimination may be more pronounced than is evident from aggregate statistics. It then applies targeted mitigation strategies to reduce this discrimination while maintaining prediction accuracy.

## Key Features

- **Multi-algorithm clustering** to identify meaningful subgroups in data
- **Conditional Mutual Information (CMI)** measurement to quantify discrimination
- **Hierarchical decomposition** to identify main contributing factors
- **Statistical validation** with permutation tests and bootstrap confidence intervals
- **Multiple mitigation strategies** including:
  - Reweighting
  - Fairness-regularized modeling
  - Subgroup calibration
- **Comprehensive visualizations** for analysis and interpretation

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cluster_discrimination_detection.git
cd cluster_discrimination_detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command-line Interface

The framework includes a command-line script for easy use:

```bash
# Run the discrimination detection on the Adult dataset
python examples/detect_discrimination.py --dataset adult --n_clusters 4 --algorithm kmeans --mitigate

# Run on a custom dataset
python examples/detect_discrimination.py --dataset custom --data_path path/to/data.csv --sensitive attr1 attr2 --outcome target
```

### Python API

You can also use the framework in your own Python code:

```python
from src.preprocessing import load_adult_dataset, preprocess_adult_dataset
from src.clustering import MultiClusteringAlgorithm
from src.cmi import calculate_cmi, calculate_cmi_per_cluster
from src.validation import permutation_test, bootstrap_ci
from src.mitigation import reweighting, FairnessRegularizedModel
from src.utils import calculate_fairness_metrics, plot_fairness_metrics

# Load and preprocess data
data = load_adult_dataset("data/adult.data")
processed_data, sensitive_columns, nonsensitive_columns, outcome_column = preprocess_adult_dataset(data)

# Make sure we're using numerical features for clustering
numerical_cols = processed_data[nonsensitive_columns].select_dtypes(include=['number']).columns
X_numeric = processed_data[numerical_cols]

# Perform clustering
clustering = MultiClusteringAlgorithm()
clusters = clustering.fit(X_numeric.values)

# Calculate CMI to measure discrimination
cmi = calculate_cmi(processed_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns)
```

For a detailed demonstration, see the example notebooks:
- `notebooks/example_adult.ipynb`: Original example notebook with the Adult dataset
- `notebooks/fixed_adult_example.ipynb`: Fixed version that handles non-numeric data correctly

## Project Structure

- `src/` - Source code for the framework
  - `preprocessing/` - Data loading and preprocessing functions
  - `clustering/` - Clustering algorithms and evaluation metrics
  - `cmi/` - Conditional Mutual Information calculations
  - `validation/` - Statistical validation methods
  - `mitigation/` - Discrimination mitigation strategies
  - `visualization/` - Visualization utilities
  - `utils.py` - Utility functions for fairness metrics, plotting, and analysis
- `notebooks/` - Jupyter notebooks demonstrating the framework
- `examples/` - Example scripts for using the framework
  - `detect_discrimination.py` - Command-line script for discrimination detection
- `data/` - Datasets (not included in repository - downloaded on demand)
- `results/` - Output visualizations and analysis results
- `tests/` - Unit tests for framework components
- `docs/` - Additional documentation

## Dataset

The example uses the [Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult) from the UCI Machine Learning Repository, which contains census data for predicting whether income exceeds $50K/year.

## Common Issues and Solutions

### String Values in Clustering

If you encounter errors like `ValueError: could not convert string to float`, make sure to use only numerical features for clustering:

```python
# Filter to numerical columns before clustering
numerical_cols = df.select_dtypes(include=['number']).columns
X_numeric = df[numerical_cols]
clusters = clustering.fit(X_numeric.values)
```

### Memory Errors

For large datasets, you can reduce memory usage:

```python
# Use a sample for finding optimal number of clusters
sample_size = min(10000, len(X))
X_sample = X.sample(sample_size, random_state=42)
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{marko2025fairpath,
  title={Uncovering Algorithmic Inequity: A Conditional Mutual Information Framework for Detecting and Mitigating Hidden Discrimination},
  author={Marko, John Gabriel O. and Neagu, Ciprian Daniel and Anand, P.B},
  journal={AI and Ethics},
  year={2026}
}
```

## License

MIT License

## Contributors

- John Gabriel O. Marko - University of Bradford
- Ciprian Daniel Neagu - University of Bradford
- P.B Anand - University of Bradford
