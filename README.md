# product-graph


[ogbn-products task](https://ogb.stanford.edu/docs/nodeprop/#ogbn-products) solution.

Written using PytorchLighthing & PytorchGeometric & Weights&Biases for experiment tracking.



## Task description

**Graph**: The ogbn-products dataset is an undirected and unweighted graph, representing an Amazon product co-purchasing network. Nodes represent products sold in Amazon, and edges between two products indicate that the products are purchased together. Node features are generated by extracting bag-of-words features from the product descriptions followed by a Principal Component Analysis to reduce the dimension to 100.

**Prediction task**: The task is to predict the category of a product in a multi-class classification setup, where the 47 top-level categories are used for target labels.

**Dataset splitting**: We use the sales ranking (popularity) to split nodes into training/validation/test sets. Specifically, we sort the products according to their sales ranking and use the top 8% for training, next top 2% for validation, and the rest for testing. This is a more challenging splitting procedure that closely matches the real-world application where labels are first assigned to important nodes in the network and ML models are subsequently used to make predictions on less important ones.


## Instructures

1. Setup environment (python >=3.10):

```
pip install -r requirements.txt
pip install -U rich

```

Special instructures for pytorch geometric, see here: https://github.com/pyg-team/pytorch_geometric:
```
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric

```

2. Download dataset:

```
python src/data/download.py
```

3. Start training (all hyperparameters must be specified in configuration file (e.g. `src/configs/config.py`)):

```
python src/train.py --config <path_to_config>
```


## Solutions

### 0. EDA

See [here](notebooks/eda.ipynb).


### 1. Baseline

Multi-layer perceptron trained only on vertices embeddings (169k parameters).

This solution does not take into account any information about relations between vertices so there is no graph-specific approaches are used. We can even consider this baseline as NLP task because vertices embeddings were extracted with BOW-method.

Scores:

![mlp_metrics](images/mlp-metrics.png)

Without using any information of the graph structure we can achieve 64.4% accuracy of predictions.

Config: `src/configs/config_mlp.py`


### 2. Graph Convolutional Network

Graph Convolutional Network (169k parameters). Scores:

TBA


Config: `src/configs/config_gcn.py`