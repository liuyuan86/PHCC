import pickle
import os.path as osp
import numpy as np

import os.path as osp
import pickle
import numpy as np
from scipy.sparse import csr_matrix, diags

import os.path as osp
import numpy as np
import scipy.sparse as sp

import os.path as osp
import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz


def load_hapergraph_dataset(path=None, dataset="ModelNet40"):

    print(f"Loading {dataset} dataset ...")

    content_path = osp.join(path, dataset, f"{dataset}.content")
    idx_features_labels = np.genfromtxt(content_path, dtype=np.str_)

    feat = idx_features_labels[:, 1:-1].astype(np.float32)

    label = idx_features_labels[:, -1].astype(np.float64).astype(np.int64)


    idx = idx_features_labels[:, 0].astype(np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    

    edges_path = osp.join(path, dataset, f"{dataset}.edges")
    edges_unordered = np.genfromtxt(edges_path, dtype=np.int32)

    edges = np.array([idx_map.get(e, -1) for e in edges_unordered.flatten()], dtype=np.int32).reshape(edges_unordered.shape)

    edges = edges[np.all(edges != -1, axis=1)]

    node_indices = edges[:, 0]
    he_indices = edges[:, 1]

    he_unique = np.unique(he_indices)
    he_map = {he: i for i, he in enumerate(he_unique)}
    he_indices = np.array([he_map[he] for he in he_indices])
    num_he = len(he_unique)

    num_nodes = edges[0].max()

    H = sp.coo_matrix(
        (np.ones(len(node_indices)), (node_indices, he_indices)),
        shape=(num_nodes, num_he),
        dtype=np.float32
    )

    he_degree = np.array(H.sum(axis=0)).flatten() 
    he_degree_inv = 1.0 / he_degree
    he_degree_inv[np.isinf(he_degree_inv)] = 0.0 
    D_e_inv = sp.diags(he_degree_inv, dtype=np.float32)

    adj = H @ D_e_inv @ H.T

    feat = feat[:num_nodes]
    label = label[:num_nodes]

    return feat, label, adj


def load_graph_dataset(path=None, dataset="cora", show_details=False):

    load_path = path + dataset + "/" + dataset
    feat = np.load(load_path+"_feat.npy", allow_pickle=True)
    label = np.load(load_path+"_label.npy", allow_pickle=True)
    adj = np.load(load_path+"_adj.npy", allow_pickle=True)

    return feat, label, adj


# Uniformly load data
def load_dataset(dataset_name, root_path="data/", **kwargs):

    dataset_mapping = {
        # Hapergraph
        "ModelNet40": (load_hapergraph_dataset, {"path": root_path}),
        "zoo": (load_hapergraph_dataset, {"path": root_path}),
        "20newsW100": (load_hapergraph_dataset, {"path": root_path}),
        "NTU2012": (load_hapergraph_dataset, {"path": root_path}),
        # Graph
        "cora": (load_graph_dataset, {"path": root_path}),
        "citeseer": (load_graph_dataset, {"path": root_path}),
        "amap": (load_graph_dataset, {"path": root_path}),
        "pubmed": (load_graph_dataset, {"path": root_path}),
    }

    dataset_name = dataset_name.strip()
    dataset_name_upper = dataset_name.upper()
    dataset_name_lower = dataset_name.lower()

    matched_key = None
    for key in dataset_mapping.keys():
        if key == dataset_name or key.upper() == dataset_name_upper or key.lower() == dataset_name_lower:
            matched_key = key
            break

    if matched_key is None:
        raise ValueError(
            f"Unsupported dataset: {dataset_name}\n"
            f"Supported datasets are: {list(dataset_mapping.keys())}"
        )

    load_func, default_kwargs = dataset_mapping[matched_key]

    final_kwargs = {**default_kwargs, "dataset": matched_key, **kwargs}

    print(f"===== Loading dataset: {matched_key} ")
    feat, label, adj = load_func(**final_kwargs)

    print(f"feat shape: {feat.shape}")
    print(f"label shape: {label.shape}")
    print(f"adj shape: {adj.shape}")

    return feat, label, adj

# Example
if __name__ == "__main__":

    # feat_el, label_el, adj_el = load_hapergraph_dataset(path="data\\raw_data\\", dataset="zoo")
    # feat_el, label_el, adj_el = load_hapergraph_dataset(path="data\\raw_data\\", dataset="20newsW100")
    # feat_el, label_el, adj_el = load_hapergraph_dataset(path="data\\raw_data\\", dataset="Mushroom")
    # feat_el, label_el, adj_el = load_hapergraph_dataset(path="data\\raw_data\\", dataset="NTU2012")
    # feat_el, label_el, adj_el = load_hapergraph_dataset(path="data\\raw_data\\", dataset="zoo")
    # feat_el, label_el, adj_el = load_graph_dataset(path="data\\raw_data\\", dataset="cora")


    feat_cora, label_cora, adj_cora = load_dataset("cora")
    print(f"Cora: feat shape={feat_cora.shape}, label shape={label_cora.shape}, adj shape={adj_cora.shape}")

    feat_zoo, label_zoo, adj_zoo = load_dataset("zoo")
    print(f"Zoo: feat shape={feat_zoo.shape}, label shape={label_zoo.shape}, adj type={type(adj_zoo)}")
