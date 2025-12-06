import numpy as np
import torch
import scipy.sparse as sp
from dgl.data.utils import load_graphs

def load_data(path, dataset):
    data_path = "./" + path + dataset
    graphs, _ = load_graphs(data_path)
    return graphs, len(graphs)

