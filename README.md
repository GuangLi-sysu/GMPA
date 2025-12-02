# Implementation of "Group-Based Detection of Cryptocurrency Laundering Using Multi-Persona Analysis"

A novel unsupervised approach for detecting money laundering patterns in cryptocurrency transaction networks using multi-persona analysis and graph neural networks.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/GuangLi-sysu/GMPA.git

# Install dependencies
pip install torch torch-geometric networkx pandas numpy scikit-learn networkx
```

### Run Detection

```bash
python main.py
```

## 📊 What It Does

This method detects money laundering in cryptocurrency networks by:

1. **Persona Decomposition**: Splits users into multiple personas based on behavioral patterns
2. **Graph Analysis**: Analyzes connected components in transaction networks
3. **Pattern Detection**: Identifies suspicious laundering patterns using GraphSAGE embeddings
4. **Probability Scoring**: Ranks suspicious groups using Gaussian Mixture Models

## 🏗️ Project Structure

```
.
├── main.py                    # Main pipeline script
├── communities.py            # GraphSAGE implementation for embeddings
├── split_util.py            # Persona decomposition utilities
├── detector.py              # Detection and evaluation algorithms
├── data/                    # Dataset directory
└── README.md               # This file
```

## 📁 Datasets

The code expects the following datasets in the `data/` directory. You can download used datasets from Google driver (https://drive.google.com/drive/folders/1vrRfvVfl2fFv2kSTCxUEZMCU-3-JwIf5). 

Each dataset should be a PyTorch file containing:
- `x`: Node features tensor
- `edge_index`: Graph connectivity (COO format)
- `y`: Ground truth labels (1=illicit, 0=licit)

## 🔧 Usage

### Basic Usage

Run the detection pipeline on both included datasets:

```python
python main.py
```

### Custom Datasets

To use your own dataset:

1. Prepare your data in the same format as the provided datasets
2. Save it as a `.pt` file in the `data/` directory
3. Modify the dataset list in `main.py`:

```python
# In main.py, line ~140:
datasets = ['Harmony_origin', 'Upbithack_origin', 'your_dataset_name']
```

### Adjusting Parameters

Key parameters you can adjust in `main.py`:

- `embedding_dim`: GraphSAGE embedding dimension (default: 32)
- `epochs`: Training epochs for GraphSAGE (default: 100)
- `top_k`: Evaluation threshold (default: 5)


## 🧪 Pipeline Overview

```
Transaction Graph → GraphSAGE Embeddings → Persona Decomposition → 
Connected Components → Feature Extraction → GMM Scoring → 
Anomaly Detection → Evaluation
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{yourcitation2023,
  title={Group-Based Detection of Cryptocurrency Laundering Using Multi-Persona Analysis},
  author={Li, Guang and Mi, Yangtian and Zhou, Jieying and Zheng, Xianghan and Wu, Weigang},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2025},
  volume = {20},
  pages = {5992-6004},
  publisher = {IEEE}
}
```

## ⚠️ Disclaimer

This tool is for research purposes only. Use in production systems requires additional validation and compliance checks.