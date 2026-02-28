import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    if split == 'test' and 'test' not in metrics:
        return format_metrics_care(metrics)
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()])

def format_metrics_care(metrics, metric_name='accuracy_score'):
    res = '\n'
    for split in metrics:
        for i in range(1, 5):
            res += f'{split} level{i}: {metrics[split][metric_name]["level{}".format(i)]}\n'
    return res

def save_metrics(metrics, save_dir):
    with open(save_dir, 'w') as f:
        for split, metrics in metrics.items():
            for metric_name, metric_vals in metrics.items():
                for level, metric_val in metric_vals.items():
                    f.write(f'{split}\t{metric_name}\t{level}\t{metric_val:.4f}\n')


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser



def print_memory_usage(i):
    print(f"Chunk {i}")
    print(f"Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
    print(f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
import torch

def poincare_distance_matrix(U: torch.Tensor, V: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    U_norm_sq = torch.sum(U ** 2, dim=-1, keepdim=True)
    U_norm_sq = torch.clamp(U_norm_sq, max=1-eps)
    
    squared_dist = torch.cdist(U, V, p=2).pow(2)
    
    denominator = ((1 - U_norm_sq).view(-1, 1) * (1 - V_norm_sq).view(1, -1))
    denominator = torch.clamp(denominator, min=eps)
    arccosh_input = 1 + 2 * squared_dist / denominator
    print("arccosh_input min:", arccosh_input.min().item())
    res = torch.arccosh(torch.clamp(arccosh_input, min=1.0001))
    # res = torch.arccosh(1 + 2 * squared_dist / denominator)
    return res



def poincare_distance_matrix(U: torch.Tensor, V: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    U_norm_sq = torch.sum(U ** 2, dim=-1, keepdim=True)
    V_norm_sq = torch.sum(V ** 2, dim=-1, keepdim=True)
    U_norm_sq = torch.clamp(U_norm_sq, max=1-eps)
    V_norm_sq = torch.clamp(V_norm_sq, max=1-eps)
    
    squared_dist = torch.cdist(U, V, p=2).pow(2)
    
    denominator = ((1 - U_norm_sq).view(-1, 1) * (1 - V_norm_sq).view(1, -1))
    denominator = torch.clamp(denominator, min=eps)
    arccosh_input = 1 + 2 * squared_dist / denominator
    print("arccosh_input min:", arccosh_input.min().item())
    res = torch.arccosh(torch.clamp(arccosh_input, min=1.0001))
    # res = torch.arccosh(1 + 2 * squared_dist / denominator)
    return res


def info_nce_loss_chunked(embeddings1, embeddings2, chunk_size=1024, temperature=0.12, pert=False):
    total_loss = 0
    num_chunks = (len(embeddings1) + chunk_size - 1) // chunk_size
    if pert:
        idx = torch.randperm(len(embeddings1), device=embeddings1.device)  # 随机打乱索引
    else:
        idx = torch.arange(len(embeddings1), device=embeddings1.device)
    # import pdb; pdb.set_trace()
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(embeddings1))
        batch_idx = idx[start_idx:end_idx]  # 获取当前块的索引

        chunk1 = embeddings1[batch_idx]
        chunk2 = embeddings2[batch_idx]
        distance = poincare_distance_matrix(chunk1, chunk2)
     
        logits = -distance / temperature
        logits_t = logits.t()
        
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        chunk_loss = (F.cross_entropy(logits, labels) + 
                     F.cross_entropy(logits_t, labels)) / 2
        
        total_loss += chunk_loss * (end_idx - start_idx)

    return total_loss / len(embeddings1) / len(embeddings2)


def supcon_loss(features, labels, temperature=0.07, base_temperature=0.07):
    """
    features: 归一化后的特征 [batch_size, feature_dim]
    labels: 类别标签 [batch_size]
    """
    device = features.device
    batch_size = features.shape[0]
    
    # 计算相似度矩阵
    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.t()) / temperature  # [batch_size, batch_size]
    
    # 创建mask
    # 对角线mask（自己和自己的相似度）
    mask_self = torch.eye(batch_size, device=device)
    
    # 同类样本mask
    mask_pos = labels.expand(batch_size, batch_size).eq(labels.expand(batch_size, batch_size).t())
    mask_pos = mask_pos - mask_self  # 移除对角线（自己和自己）
    
    # 计算loss
    # 对exp(sim)进行数值稳定性处理
    exp_sim = torch.exp(sim_matrix)
    
    # 将自己和自己的相似度设为0
    exp_sim = exp_sim * (1 - mask_self)
    
    # 对每个样本，计算其与同类样本的相似度
    pos_sim = exp_sim * mask_pos
    pos_sim = pos_sim.sum(dim=1)
    
    # 计算分母（所有样本的相似度之和）
    neg_sim = exp_sim.sum(dim=1)
    
    # 计算正样本个数（每个anchor的同类样本数量）
    n_pos = mask_pos.sum(dim=1)
    
    # 处理没有正样本的情况
    pos_sim = pos_sim + 1e-8  # 避免除0
    n_pos = n_pos + 1e-8
    
    # 计算loss
    loss = -torch.log(pos_sim / neg_sim) / n_pos
    
    return loss.mean()
