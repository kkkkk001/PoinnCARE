from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
# import pickle
import time
import numpy as np
import pandas as pd
import optimizers
import torch
from config import parser

from models.base_models import NCModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics, save_metrics, info_nce_loss_chunked
from utils.eval_utils import Evaluator

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

os.environ['NVIDIA_DRIVER_CAPABILITIES'] = 'compute,utility'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)

    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda) if int(args.cuda) >= 0 else ''
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
        os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])
    args.impr_based_on = 'acc'

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    logging.info(args)

    #################### Load data ####################
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    args.n_nodes, args.feat_dim = data['features'].shape
        
    for x, val in data.items():
        if torch.is_tensor(data[x]):
            data[x] = data[x].to(args.device)

    #################### Initialize evaluator ####################
    all_ecs = np.loadtxt('data/all_ec.txt', dtype=str)
    idx2ec = data['idx2ec']
    evaluator = Evaluator(all_ecs, idx2ec)

    #################### set model class ####################
    args.task = 'nc'
    Model = NCModel
    if len(data['labels'].shape) == 1:
        args.n_classes = int(data['labels'].max() + 1)
    else:
        args.n_classes = data['labels'].shape[1]
    logging.info(f'Num classes: {args.n_classes}')

    if args.save:
        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))


    #################### prepare total res container ####################
    col_names = pd.MultiIndex.from_product([['idx_30_test', 'idx_30_50_test', 'idx_price_test', 'idx_promiscuous_test'], 
                                            ['ori_acc','accuracy_score', 'precision_score', 'recall_score', 'f1_score',],
                                            ['level1', 'level2', 'level3', 'level4']])
    total_res = pd.DataFrame(columns=col_names)

    for repeat_idx in range(args.repeat):
        logging.info(f'#################### Repeat {repeat_idx+1} ####################')


        #################### Model and optimizer ####################
        model1 = Model(args)
        model2 = Model(args)
        model1.evaluator = evaluator
        model2.evaluator = evaluator

        logging.info(str(model1))
        logging.info(str(model2))

        optimizer_align = getattr(optimizers, args.optimizer_align)([
            {'params': model1.parameters(), 'lr': args.lr1_align, 'weight_decay': args.weight_decay1_align},
            {'params': model2.parameters(), 'lr': args.lr2_align, 'weight_decay': args.weight_decay2_align},
        ])


        tot_params = sum([np.prod(p.size()) for p in model1.parameters()]) + sum([np.prod(p.size()) for p in model2.parameters()])
        logging.info(f"Total number of parameters: {tot_params}")
        
        
        if args.cuda is not None and int(args.cuda) >= 0 :
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
            model1 = model1.to(args.device)
            model2 = model2.to(args.device)
            for x, val in data.items():
                if torch.is_tensor(data[x]):
                    data[x] = data[x].to(args.device)


        #################### Train model ####################
        t_total = time.time()
        counter = 0
        best_val_metrics = model1.init_metric_dict()
        best_test_metrics = None
        # best_emb = None
        all_index = list(range(len(data['labels'])))
        identity_matrix = torch.eye(args.dim, dtype=torch.float32, device=args.device)

        for epoch in range(args.epochs):
            t = time.time()
            model1.train()
            model2.train()
            optimizer_align.zero_grad()
            embeddings1 = model1.encode(data['features'], data['adj_train_norm1'])
            embeddings2 = model2.encode(data['features'], data['adj_train_norm2'])
            train_metrics1 = model1.compute_metrics(embeddings1, data, 'train', adj_name='adj_train_norm1')
            train_metrics2 = model2.compute_metrics(embeddings2, data, 'train', adj_name='adj_train_norm2')


            align_loss = torch.norm(embeddings1 - embeddings2, p=2, dim=-1).mean()
            deco_loss1 = torch.norm(torch.mm(embeddings1.T, embeddings1)-identity_matrix, p='fro')/args.dim**2
            deco_loss2 = torch.norm(torch.mm(embeddings2.T, embeddings2)-identity_matrix, p='fro')/args.dim**2
            loss_align = args.loss3_weight*align_loss + args.loss4_weight*deco_loss1 + args.loss5_weight*deco_loss2

            loss = args.loss1_weight*train_metrics1['loss'] + \
                    args.loss2_weight*train_metrics2['loss'] + loss_align
            loss.backward()


            for name, param in model1.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in grad of model1 {name}")
            for name, param in model2.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in grad of model2 {name}")

            optimizer_align.step()

            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                    '| model1 ' + format_metrics(train_metrics1, 'train'),
                                    '| model2 ' + format_metrics(train_metrics2, 'train'),
                                    '| align_loss: {:.4f}'.format(loss_align.item()),
                                    'time: {:.4f}s'.format(time.time() - t)
                                    ]))
            if (epoch + 1) % args.eval_freq == 0:
                model1.eval()
                model2.eval()
                embeddings1 = model1.encode(data['features'], data['adj_train_norm1'])
                embeddings2 = model2.encode(data['features'], data['adj_train_norm2'])
                output1 = model1.decode(embeddings1, data['adj_train_norm1'], all_index)
                output2 = model2.decode(embeddings2, data['adj_train_norm2'], all_index)
                output = args.output1_weight * output1 + args.output2_weight * output2
                val_metrics1 = model1.compute_metrics(embeddings1, data, 'val', output1)
                val_metrics2 = model2.compute_metrics(embeddings2, data, 'val', output2)
                val_metrics_align = model1.compute_metrics(embeddings1, data, 'val', output)

                if (epoch + 1) % args.log_freq == 0:
                    logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), '| model1 ' + format_metrics(val_metrics1, 'val'),
                                           '| model2 ' + format_metrics(val_metrics2, 'val'),
                                           '| align ' + format_metrics(val_metrics_align, 'val'),
                                           ]))


                if model1.has_improved(best_val_metrics, val_metrics_align, args.impr_based_on):
                    best_test_metrics = model1.compute_metrics(embeddings1, data, 'test', output)
                    best_emb1 = embeddings1.cpu()
                    best_emb2 = embeddings2.cpu()
                    best_val_metrics = val_metrics_align
                    counter = 0
                    if args.save:
                        np.save(os.path.join(save_dir, f'embeddings1_{repeat_idx}.npy'), best_emb1.cpu().detach().numpy())
                        np.save(os.path.join(save_dir, f'embeddings2_{repeat_idx}.npy'), best_emb2.cpu().detach().numpy())
                        torch.save(model1.state_dict(), os.path.join(save_dir, f'model1_{repeat_idx}.pth'))
                        torch.save(model2.state_dict(), os.path.join(save_dir, f'model2_{repeat_idx}.pth'))
                else:
                    counter += 1
                    if counter == args.patience and epoch > args.min_epochs:
                        logging.info("Early stopping")
                        break

        #################### Test model ####################
        logging.info("Optimization Finished!")
        logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        if not best_test_metrics:
            model1.eval()
            model2.eval()
            best_emb1 = model1.encode(data['features'], data['adj_train_norm1'])
            best_emb2 = model2.encode(data['features'], data['adj_train_norm2'])
            output1 = model1.decode(best_emb1, data['adj_train_norm1'], all_index)
            output2 = model2.decode(best_emb2, data['adj_train_norm2'], all_index)
            output = args.output1_weight * output1 + args.output2_weight * output2

            best_test_metrics = model1.compute_metrics(best_emb1, data, 'test', output)
            if args.save:
                np.save(os.path.join(save_dir, f'embeddings1_{repeat_idx}.npy'), best_emb1.cpu().detach().numpy())
                np.save(os.path.join(save_dir, f'embeddings2_{repeat_idx}.npy'), best_emb2.cpu().detach().numpy())
                torch.save(model1.state_dict(), os.path.join(save_dir, f'model1_{repeat_idx}.pth'))
                torch.save(model2.state_dict(), os.path.join(save_dir, f'model2_{repeat_idx}.pth'))
        logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
        logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
        save_metrics(best_test_metrics, save_dir+'/metrics_{}.txt'.format(repeat_idx))
        total_res = pd.concat([total_res, pd.DataFrame([[best_test_metrics[i[0]][i[1]][i[2]] for i in  col_names]], columns=col_names)])


    # logging.info(total_res)
    logging.info(f'#################### summary ####################')
    summary_df = pd.DataFrame({'Mean': total_res.mean(), 'Std': total_res.std()})
    summary_df = summary_df.reset_index()
    logging.info(summary_df.to_string(index=False))
    
if __name__ == '__main__':
    args = parser.parse_args()
    train(args)

    
