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

See the example notebook for a detailed demonstration using the Adult income dataset:

```python
from src.preprocessing import load_adult_dataset, preprocess_adult_dataset
from src.clustering import MultiClusteringAlgorithm
from src.cmi import calculate_cmi, calculate_cmi_per_cluster
from src.validation import permutation_test, bootstrap_ci
from src.mitigation import reweighting, FairnessRegularizedModel

# Load and preprocess data
data = load_adult_dataset("data/adult.data")
processed_data, sensitive_columns, nonsensitive_columns, outcome_column = preprocess_adult_dataset(data)

# Perform clustering
clustering = MultiClusteringAlgorithm()
clusters = clustering.fit(processed_data[nonsensitive_columns].values)

# Calculate CMI to measure discrimination
cmi = calculate_cmi(processed_data, clusters, sensitive_columns, outcome_column, nonsensitive_columns)
```

## Project Structure

- `src/` - Source code for the framework
  - `preprocessing/` - Data loading and preprocessing functions
  - `clustering/` - Clustering algorithms and evaluation metrics
  - `cmi/` - Conditional Mutual Information calculations
  - `validation/` - Statistical validation methods
  - `mitigation/` - Discrimination mitigation strategies
  - `visualization/` - Visualization utilities
- `notebooks/` - Jupyter notebooks demonstrating the framework
- `data/` - Datasets (not included in repository - downloaded on demand)
- `results/` - Output visualizations and analysis results
- `tests/` - Unit tests for framework components
- `docs/` - Additional documentation

## Dataset

The example uses the [Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult) from the UCI Machine Learning Repository, which contains census data for predicting whether income exceeds $50K/year.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{marko2025fairpath,
  title={Uncovering Algorithmic Inequity: A Conditional Mutual Information Framework for Detecting and Mitigating Hidden Discrimination},
  author={Marko, John Gabriel O. and Neagu, Ciprian Daniel and Anand, P.B},
  journal={},
  year={2025}
}
```
## License

## Contributors

- John Gabriel O. Marko - University of Bradford
- Ciprian Daniel Neagu - University of Bradford
- P.B Anand - University of Bradford
