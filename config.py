import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'seed': (1234, 'seed for training'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'repeat': (5, 'number of repeats for training'),

        'dropout': (0.0, 'dropout probability'),
        'optimizer_align': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'lr1_align': (0.01, 'learning rate for alignment'),
        'lr2_align': (0.01, 'learning rate for alignment'),

        'epochs': (10000, 'maximum number of epochs to train for'),
        'min-epochs': (3000, 'do not early stop before min-epochs'),
        'patience': (1000, 'patience for early stopping'),
        'weight-decay1_align': (0., 'l2 regularization strength for alignment'),
        'weight-decay2_align': (0., 'l2 regularization strength for alignment'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),

        'loss1_weight': (1.0, 'weight for loss1'),
        'loss2_weight': (1.0, 'weight for loss2'),
        'loss3_weight': (0.01, 'weight for loss3'),
        'loss4_weight': (0.0, 'weight for loss4'),
        'loss5_weight': (0.0, 'weight for loss5'),
        'output1_weight': (0.5, 'weight for output1'),
        'output2_weight': (0.5, 'weight for output2'),

        'log-freq': (5, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        
        'sweep-c': (0, ''),

    },
    'must_have_model_config': {
        'model': ('HGCN', 'which encoder to use, can be any of [Shallow, MLP, HNN, GCN, GAT, HyperGCN]'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'dim': (512, 'embedding dimension'),
    },
    'model_config': {
        # 'task': ('nc', 'which tasks to train on, can be any of [lp, nc, as]'),
        # 'loss_fn': ('CrossEntropyLoss', 'which loss function to use, can be any of [CrossEntropyLoss, BCEWithLogitsLoss]'),
        # 'concat-agg': (0, 'whether to concatenate the aggregated features with the input features'),
        # 'r': (2., 'fermi-dirac decoder parameter for lp'),
        # 't': (1., 'fermi-dirac decoder parameter for lp'),
        # 'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        # 'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'dual_graph_training_config': {

        # 'loss_type': ('cca', 'which loss function to use, can be any of [infonce, cca]'),
        # 'infonce-tau': (0.12, 'temperature for infonce loss'),
        # 'infonce-pert': (False, 'whether to use perturbation for infonce loss'),

    },
    'data_config': {
        'dataset': ('care_graph', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        # 'use-which-feats': ('feats', 'which features to use: feat, sup_emb, esm_c'),
        # 'use-which-adj': ('adj', 'which adjacency matrix to use: adj, adj_sup, adj_esm_c'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
        # 'preprocess-adj': (None, 'whether to preprocess the adjacency matrix'),
        # 'diff-layers': (1, 'number of layers of diffusion in preprocess-adj'),
        # 'ppr-alpha': (0.1, 'alpha for personalized page rank in preprocess-adj'),
        # 'hk-t': (2, 't for heat kernel in preprocess-adj'),
        # 'if-cache': (1, 'whether to cache chunks, 1 for yes, 0 for no'),
    },
    'dual_graph_data_config': {
        # 'align-dual-graphs': (0, 'whether to align dual graphs'),
        'adj1': ('adj_foldseek_nbit_03_inductive', 'name of first adjacency matrix'),
        'adj2': ('adj_disco_005_07_inductive', 'name of second adjacency matrix'),
        'preprocess-adj1': (None, 'whether to preprocess the first adjacency matrix'),
        'preprocess-adj2': (None, 'whether to preprocess the second adjacency matrix'),
        'ppr-alpha1': (0.1, 'alpha for personalized page rank in preprocess-adj1'),
        'ppr-alpha2': (0.1, 'alpha for personalized page rank in preprocess-adj2'),
        'diff-layers1': (1, 'number of layers of diffusion in preprocess-adj1'),
        'diff-layers2': (1, 'number of layers of diffusion in preprocess-adj2'),
        'hk-t1': (2, 't for heat kernel in preprocess-adj1'),
        'hk-t2': (2, 't for heat kernel in preprocess-adj2'),
    },
    # 'pretrain_config': {
    #     'pretrain_epochs': (0, 'number of pretraining epochs. default 0 means no pretraining'),
    #     'pretrain_batch_size': (5120, 'batch size for pretraining'),
    #     'pretrain_lr': (0.0001, 'learning rate for pretraining'),
    #     'pretrain_patience': (25, 'patience for pretraining'),
    #     'pretrain_esm_c_batchsize': (512, 'batch size for pretraining ESM-C'),
    #     'temperature': (0.1, 'temperature for supcon-hard loss'),
    #     'train_esmc': (0, 'whether to train ESM-C'),
    # },
    'inference_config': {
        'model_path': (None, 'path to models'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
