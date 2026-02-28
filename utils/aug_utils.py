import os
import pickle as pkl
import sys, random, math

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.aa_utils import aa_to_num, num_to_aa

import torch.nn.functional as F



class MyDataClass(Dataset):
    def __init__(self, data):
        super(MyDataClass, self).__init__()
        self.data = data
        self.index2entry = {idx:entry for entry, idx in data['entry2index'].items()}

    def __len__(self):
        return len(self.data['features'])

    def __getitem__(self, idx):
        entry = self.index2entry[idx]
        slice_data = {'features': self.data['features'][idx], 
                        'labels': self.data['labels'][idx],
                        'active_sites': self.data['active_sites'][entry],
                        'binding_sites': self.data['binding_sites'][entry],
                        'm_cas_sites': self.data['m_cas_sites'][entry],
                        'seq': self.data['seq'][entry]}
        return slice_data


def my_collate(batch):
    return {
        'features': torch.stack([item['features'] for item in batch]),
        'labels': torch.stack([item['labels'] for item in batch]),
        'active_sites': pad_sequence([torch.tensor(item['active_sites']) for item in batch], batch_first=True, padding_value=0),
        'binding_sites': pad_sequence([torch.tensor(item['binding_sites']) for item in batch], batch_first=True, padding_value=0),
        'm_cas_sites': pad_sequence([torch.tensor(item['m_cas_sites']) for item in batch], batch_first=True, padding_value=0),
        'seq': pad_sequence([torch.tensor([aa_to_num[i] for i in item['seq']]) for item in batch], batch_first=True, padding_value=0)
    }


def augment_data(data, args):
    total_act_sites = torch.cat([data['active_sites'], data['binding_sites'], data['m_cas_sites']], dim=1)
    seq = data['seq']

    pos_seq = seq.clone()
    neg_seq = seq.clone()
    mu, sigma = .10, .02
    pos_mut_rates = np.random.normal(mu, sigma, total_act_sites.shape[0])

    for i in range(total_act_sites.shape[0]):
        seq_len = (seq[i]!=0).sum()

        act_sites = total_act_sites[i][total_act_sites[i]!=0]-1 #??
        act_sites = act_sites[act_sites < seq_len]
        non_act_sites = [j for j in range(seq_len) if j not in act_sites.tolist()]
        
        ## gen neg ##
        act_res = seq[i][act_sites.to(int)]
        replace_candidate = set(range(1, 26)) - set(act_res.tolist())
        replace_res = [random.choice(list(replace_candidate)) for _ in act_sites]
        neg_seq[i][act_sites.to(int)] = torch.tensor(replace_res).to(neg_seq.dtype)

        ## gen pos ##
        num_mut = math.ceil(len(non_act_sites) * pos_mut_rates[i])
        pos_seq[i][random.sample(non_act_sites, num_mut)] = torch.tensor(aa_to_num['<mask>'], dtype=pos_seq.dtype)

    aug_seq = torch.cat([seq, pos_seq, neg_seq], dim=0)
    anchor_idx = torch.arange(len(seq))
    pos_idx = torch.arange(len(seq), len(seq) + len(pos_seq))
    neg_idx = torch.arange(len(seq) + len(pos_seq), len(seq) + len(pos_seq) + len(neg_seq))
    aug_label = torch.zeros((len(aug_seq), len(aug_seq)))
    aug_label[anchor_idx, anchor_idx] = 1
    aug_label[anchor_idx, pos_idx] = 1
    aug_label[pos_idx, anchor_idx] = 1
    aug_label[pos_idx, pos_idx] = 1
    aug_label[neg_idx, neg_idx] = 1
    return aug_seq, aug_label


def seq2str(seq):
    if isinstance(seq[0], torch.Tensor):
        return ''.join([num_to_aa[i.item()] for i in seq if i.item() != 0])
    elif isinstance(seq[0], str):
        return seq

import subprocess

def get_gpu_utilization():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return int(result.strip())
    except:
        return -1  # 返回-1表示获取失败


def get_gpu_memory_usage():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        return int(result.strip())
    except:
        return -1


class emsc_encoding():
    def __init__(self, esmc_model, batch_size=64):
        self.esmc_model = esmc_model
        self.tokenizer = self.esmc_model.tokenizer
        self.batch_size = batch_size


    def __call__(self, seq_list, batch_idx, logging):
        seq2str_dict = {i:seq2str(seq) for i, seq in enumerate(seq_list)}
        seq_set = list(seq2str_dict.values())

        embedding_dict = self.esmc_model.embed_dataset(
            sequences=seq_set,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size, # adjust for your GPU memory
            max_len=5000, # adjust for your needs
            full_embeddings=False, # if True, no pooling is performed
            embed_dtype=torch.float32, # cast to what dtype you want
            pooling_types=['mean'], # more than one pooling type will be concatenated together
            num_workers=4, # if you have many cpu cores, we find that num_workers = 4 is fast for large datasets
            sql=False, # if True, embeddings will be stored in SQLite database
            save=False, # if True, embeddings will be saved as a .pth file
        )
        logging.info(f'{batch_idx}th batch: {torch.cuda.memory_allocated() / 1024**2:.2f} MB, {get_gpu_memory_usage():.2f} MB, {get_gpu_utilization()}%')

        embedding = torch.zeros((len(seq_list), 960))
        for i, seq in enumerate(seq_list):
            embedding[i] = embedding_dict[seq2str_dict[i]]

        return embedding



