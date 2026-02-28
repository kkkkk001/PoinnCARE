"""Data utils functions for pre-processing and data loading."""
import os
import pickle as pkl
import sys, random
import logging

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.aa_utils import aa_to_num, num_to_aa


def load_data(args, datapath):
    data = load_data_nc(datapath, adj1=args.adj1, adj2=args.adj2)

    args.preprocess_adj, args.ppr_alpha, args.diff_layers = args.preprocess_adj1, args.ppr_alpha1, args.diff_layers1
    data['adj_train1'] = preprocess_adj(data['adj_train1'], args)
    args.preprocess_adj, args.ppr_alpha, args.diff_layers = args.preprocess_adj2, args.ppr_alpha2, args.diff_layers2
    data['adj_train2'] = preprocess_adj(data['adj_train2'], args)
    data['adj_train_norm1'], data['features'] = process(
        data['adj_train1'], data['features'], args.normalize_adj, args.normalize_feats
    )
    data['adj_train_norm2'], data['features'] = process(
        data['adj_train2'], data['features'], args.normalize_adj, args.normalize_feats
    )
    return data


# ############### FEATURES PROCESSING ####################################

def preprocess_adj(adj, args):
    if args.preprocess_adj == None:
        return adj
    elif args.preprocess_adj == 'PPR':
        return ppr(adj, args.ppr_alpha, args.diff_layers)
    elif args.preprocess_adj == 'HKPR':
        return hkpr(adj, args.hk_t, args.diff_layers)
    return adj

def ppr(adj, alpha, diff_layers):
    A = adj.copy()
    N = A.shape[0]
    ppr_matrix = alpha * sp.eye(N)
    identity_matrix = sp.eye(N)
    for _ in range(diff_layers):
        ppr_matrix = (1 - alpha) * ppr_matrix @ A + alpha * identity_matrix
    return ppr_matrix

def hkpr(adj, t, diff_layers):
    from scipy.stats import poisson
    N = adj.shape[0]
    A_power = sp.eye(N)
    hkpr_matrix = poisson.pmf(0, t) * A_power # i=0
    for i in range(1, diff_layers+1):
        coeff = poisson.pmf(i, t)
        hkpr_matrix += coeff * A_power @ adj
    return hkpr_matrix



def process(adj, features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
    if normalize_feats:
        features = normalize(features)
    features = torch.Tensor(features)
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def augment(adj, features, normalize_feats=True):
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    return features


# ############### DATA SPLITS #####################################################


def mask_edges(adj, val_prop, test_prop, seed):
    np.random.seed(seed)  # get tp edges
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test + n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, torch.LongTensor(train_edges), torch.LongTensor(train_edges_false), torch.LongTensor(val_edges), \
           torch.LongTensor(val_edges_false), torch.LongTensor(test_edges), torch.LongTensor(
            test_edges_false)  


def split_data(labels, val_prop, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_val = round(val_prop * nb_pos_neg)
    nb_test = round(test_prop * nb_pos_neg)
    idx_val_pos, idx_test_pos, idx_train_pos = pos_idx[:nb_val], pos_idx[nb_val:nb_val + nb_test], pos_idx[
                                                                                                   nb_val + nb_test:]
    idx_val_neg, idx_test_neg, idx_train_neg = neg_idx[:nb_val], neg_idx[nb_val:nb_val + nb_test], neg_idx[
                                                                                                   nb_val + nb_test:]
    return idx_val_pos + idx_val_neg, idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


# ############### NODE CLASSIFICATION DATA LOADERS ####################################
def load_data_nc(data_path, adj1, adj2):
    adj1, adj2, features, labels, splits, \
    idx2ec, entry2index = load_protein_data(data_path, adj1, adj2)
    
    data = {'adj_train1': adj1, 'adj_train2': adj2, 'features': features, 'labels': labels, 
            'idx_train': splits['protein_train50'], 'idx_val': splits['val'],
            'idx_30_test': splits['30_protein_test'], 
            'idx_30_50_test': splits['30-50_protein_test'], 
            'idx_price_test': splits['price_protein_test'], 
            'idx_promiscuous_test': splits['promiscuous_protein_test']}
    
    data['idx2ec'] = idx2ec
    data['entry2index'] = entry2index
    return data
    




# ############### DATASETS ####################################
def load_protein_data(data_path, adj1, adj2):
    logging.info(f"Loading {adj1} and {adj2} adjacency matrices")
    adj1 = sp.load_npz(os.path.join(data_path, f"{adj1}.npz"))
    adj2 = sp.load_npz(os.path.join(data_path, f"{adj2}.npz"))
    
    logging.info(f"Loading features {os.path.join(data_path, 'feats.npy')}")
    features = np.load(os.path.join(data_path, "feats.npy"))
    features = torch.FloatTensor(features)

    logging.info(f"Loading labels {os.path.join(data_path, 'labels.npy')}")
    labels = np.load(os.path.join(data_path, "labels.npy"))
    labels = torch.FloatTensor(labels)

    entry2index = pkl.load(open(os.path.join(data_path, 'entry2index.pkl'), 'rb'))
    ec2index = pkl.load(open(os.path.join(data_path, 'ec2idx.pkl'), 'rb'))
    idx2ec = {v: k for k, v in ec2index.items()}


    # prepare splits
    splits = {}
    for split in ['protein_train50', '30_protein_test', '30-50_protein_test', 'price_protein_test', 'promiscuous_protein_test']:
        idx = np.loadtxt(os.path.join(data_path, "{}.csv".format(split)), dtype=str, delimiter='\t', usecols=(0,), skiprows=1)
        splits[split] = [entry2index[i] for i in idx]
    
    splits['val'] = splits['30_protein_test']

    return adj1, adj2, features, labels, splits, idx2ec, entry2index
