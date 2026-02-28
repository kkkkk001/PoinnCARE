from sklearn.metrics import average_precision_score, accuracy_score, f1_score
import numpy as np
# from CLEAN.infer import get_accuracy_level
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch, os
import itertools

def get_accuracy_level(predicted_ECs, true_ECs):
    """
    based on a list of predicted_ECs, calculates the highest level of accuracy achieved, against all true_ECs. Returns a list of the same length as true_ECs.
    """
    #convert true_EC to a list
    if type(predicted_ECs) == str:
        predicted_ECs = [predicted_ECs]
        
    if type(true_ECs) == str:
        true_ECs = [true_ECs]

    maxes = []
    err_c = 0

    for true_EC in true_ECs:

        true_split = true_EC.split('.')
        
        counters = []
        for predicted_EC in predicted_ECs:
            try:
                #if the output is not an EC number
                if predicted_EC.count('.') != 3:
                    err_c += 1 
                    predicted_EC = '0.0.0.0'
                    
                predicted_split = predicted_EC.split('.')
                counter = 0
    
                for predicted, true in zip(predicted_split, true_split):
                    if predicted == true:
                        counter += 1
                    else:
                        break
                counters.append(counter)
            except:
                print("ERROR:", predicted_EC)
        
        maxes.append(np.max(counters))
    return maxes


def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1



def acc_care(output, data_label, data):
    acc_all_levels = {'level{}'.format(i): [] for i in range(1, 5)}
    output = output.cpu().detach().numpy()
    data_label = data_label.cpu().detach().numpy()
    for pred_y, true_y in zip(output, data_label):
        predicted_idx = np.argpartition(-pred_y, int(true_y.sum()))[:int(true_y.sum())]
        true_idx = np.where(true_y==1)[0]
        predicted_ec_name = [data['idx2ec'][i] for i in predicted_idx]
        true_ec_name = [data['idx2ec'][i] for i in true_idx]
        acc_levels = get_accuracy_level(predicted_ec_name, true_ec_name)
        for i in range(1, 5):
            acc_all_levels['level{}'.format(i)].append((np.array(acc_levels)>=i).mean())
    for level in acc_all_levels.keys():
        acc_all_levels[level] = np.mean(acc_all_levels[level])
    return acc_all_levels


def keep_top_k_per_row(matrix, x):
    result = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        row = matrix[i]
        
        top_k = min(x[i], row.size)  # 确保 k 不超过行长度
        indices = np.argpartition(row, -top_k)[-top_k:]
        
        result[i, indices] = 1
    
    return result


class Evaluator:
    def __init__(self, all_ecs, idx2ec):
        self.all_ecs_level4 = all_ecs
        self.idx2ec = idx2ec
        self.all_ecs_level3 = list(set('.'.join(ec.split('.')[:3]) for ec in all_ecs))
        self.all_ecs_level2 = list(set('.'.join(ec.split('.')[:2]) for ec in all_ecs))
        self.all_ecs_level1 = list(set('.'.join(ec.split('.')[:1]) for ec in all_ecs))

        ec_level3_2idx_list = {ec: [] for ec in self.all_ecs_level3}
        ec_level2_2idx_list = {ec: [] for ec in self.all_ecs_level2}
        ec_level1_2idx_list = {ec: [] for ec in self.all_ecs_level1}
        for idx, ec in self.idx2ec.items():
            for level in [3, 2, 1]:
                level_ec = '.'.join(ec.split('.')[:level])
                eval(f'ec_level{level}_2idx_list')[level_ec].append(idx)

        self.ec_level3_2idx_list = ec_level3_2idx_list
        self.ec_level2_2idx_list = ec_level2_2idx_list
        self.ec_level1_2idx_list = ec_level1_2idx_list



    def level_reduce(self, pred_y, level):
        sample_num = pred_y.shape[0]
        class_num = len(getattr(self,f'ec_level{level}_2idx_list'))
        all_ecs_level = getattr(self,f'all_ecs_level{level}')
        ec_level_2idx_list = getattr(self,f'ec_level{level}_2idx_list')

        res = np.zeros((sample_num, class_num))
        for j in range(class_num):
            ec = all_ecs_level[j]
            res[:, j] = pred_y[:, ec_level_2idx_list[ec]].sum(1)
        res[res > 0] = 1
        return res


    def eval_care(self, output, data_label, metric_list=['accuracy_score', 'precision_score', 'recall_score', 'f1_score'], 
    split=None, pred_y_level4=None):
        if pred_y_level4 is None:
            output = output.cpu().detach().numpy()
            data_label = data_label.cpu().detach().numpy()
            pred_y_level4 = keep_top_k_per_row(output, data_label.sum(1).astype(int))

        data_label_level4 = data_label

        metrics = {}
        for m in metric_list:
            if m == 'roc_auc_score':
                metrics[m] = {}
                metrics[m]['level4'] = eval(m)(data_label, output)
                metrics[m]['level3'], metrics[m]['level2'], metrics[m]['level1'] = -1, -1, -1
            else:
                metrics[m] = self.eval_on_one_metric(pred_y_level4, data_label_level4, metric=m, split=split)
        return metrics
    

    def eval_on_one_metric(self, pred_y_level4, data_label_level4, metric='precision_score', split=None):
        res = {}
        pred_y_level3 = self.level_reduce(pred_y_level4, level=3)
        data_label_level3 = self.level_reduce(data_label_level4, level=3)
        pred_y_level2 = self.level_reduce(pred_y_level4, level=2)
        data_label_level2 = self.level_reduce(data_label_level4, level=2)
        pred_y_level1 = self.level_reduce(pred_y_level4, level=1)
        data_label_level1 = self.level_reduce(data_label_level4, level=1)
        assert metric in ['accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        for level in range(1, 5):
            if metric == 'accuracy_score':
                param = {'y_true': eval(f'data_label_level{level}'), 'y_pred': eval(f'pred_y_level{level}')}
                res[f'level{level}'] = eval(metric)(**param)
            else:
                if split is None or split.find('price') != -1 or split.find('promiscuous') != -1:
                    average = 'samples'
                    param = {'y_true': eval(f'data_label_level{level}'), 'y_pred': eval(f'pred_y_level{level}'), 'average': average}
                else:
                    average = 'macro'
                    y_true = eval(f'data_label_level{level}')
                    y_pred = eval(f'pred_y_level{level}')
                    mask = y_true.sum(0) > 0
                    y_true = y_true[:,mask]
                    y_pred = y_pred[:,mask]
                    param = {'y_true': y_true, 'y_pred': y_pred, 'average': average}
                res[f'level{level}'] = eval(metric)(**param)
        return res


# class GenTriplet:
#     def __init__(self, labels, num_nodes):
#         self.num_nodes = num_nodes 
#         self.ec_idx2node_idx_lst = {}
#         for i in range(labels.shape[1]):
#             self.ec_idx2node_idx_lst[i] = torch.nonzero(labels[:, i]).view(-1)


#     def __call__(self, labels):
#         pos_list = []
#         neg_list = []
#         for i in range(self.num_nodes):
#             i_label = torch.nonzero(labels[i]).view(-1)
#             node_list = []
#             for ec in i_label:
#                 node_list.extend(self.ec_idx2node_idx_lst[ec.item()])
#             node_list = torch.tensor(node_list)
#             # print(ec, node_list)
#             pos = np.random.choice(node_list)
#             neg = np.random.choice(np.setdiff1d(np.arange(len(labels)), node_list))
#             pos_list.append(pos)
#             neg_list.append(neg)
#         pos_idx = np.array(pos_list)
#         neg_idx = np.array(neg_list)
#         return pos_idx, neg_idx


import torch
import numpy as np


class GenTriplet:
    def __init__(self, labels, idx_lst):
        self.labels = labels[idx_lst].cpu().numpy()
        self.ec_idx2node_idx_lst = {}
        for i in range(labels.shape[1]):
            if len(np.where(self.labels[:, i]!=0)[0]):
                self.ec_idx2node_idx_lst[i] = np.where(self.labels[:, i]!=0)[0]



    def __call__(self, epoch):
        if self.pre_computed:
            pre_len = self.pre_computed_pos.shape[1]
            return self.pre_computed_pos[:, epoch%pre_len], self.pre_computed_neg[:, epoch%pre_len]
        pos_list = []
        neg_list = []
        for i in range(self.labels.shape[0]):
            i_label = np.where(self.labels[i]!=0)[0]
            select_pos_label = np.random.choice([ec.item() for ec in i_label])
            select_neg_label = np.random.choice(np.setdiff1d(list(self.ec_idx2node_idx_lst.keys()), [ec.item() for ec in i_label]))
            pos = np.random.choice(self.ec_idx2node_idx_lst[select_pos_label])
            neg = np.random.choice(self.ec_idx2node_idx_lst[select_neg_label])
            pos_list.append(pos)
            neg_list.append(neg)
        pos_idx = np.array(pos_list)
        neg_idx = np.array(neg_list)
        return pos_idx, neg_idx

    def pre_compute(self, save_path):
        # import pdb; pdb.set_trace()
        if os.path.exists(save_path):
            pos = np.load(save_path+'/pos.npy')
            neg = np.load(save_path+'/neg.npy')
            self.pre_computed = True
            self.pre_computed_pos = pos
            self.pre_computed_neg = neg
            return pos, neg
        pos_list = []
        neg_list = []
        for i in range(self.labels.shape[0]):
            i_label = np.where(self.labels[i]!=0)[0]
            pos = list(itertools.chain.from_iterable([self.ec_idx2node_idx_lst[p] for p in i_label]))
            # not_i_label = np.setdiff1d(list(self.ec_idx2node_idx_lst.keys()), [ec.item() for ec in i_label])
            # neg = list(itertools.chain.from_iterable([self.ec_idx2node_idx_lst[n] for n in not_i_label]))
            neg = np.setdiff1d(np.arange(self.labels.shape[0]), pos)
            pos = np.random.choice(pos, size=1000, replace=True)
            neg = np.random.choice(neg, size=1000, replace=True)
            pos_list.append(pos)
            neg_list.append(neg)
        pos = np.array(pos_list).astype(int)
        neg = np.array(neg_list).astype(int)
        os.makedirs(save_path, exist_ok=True)  
        np.save(save_path+'/pos.npy', pos)
        np.save(save_path+'/neg.npy', neg)
        self.pre_computed = True
        self.pre_computed_pos = pos
        self.pre_computed_neg = neg
        return pos, neg
