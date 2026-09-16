import random

import numpy as np
import torch
from torch.backends import cudnn
from sklearn import metrics
from scipy.optimize import linear_sum_assignment


def fix_random_seeds(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


def adjust_learning_config(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        import math
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if lr <= args.minlr:   
        lr = args.minlr 

    return lr

def get_dynamic_weight(epoch, max_weight=0.1, warmup_epochs=20,
                       dec_warmup_len=None):
    if epoch < warmup_epochs:
        return 0.0
    else:
        dec_epoch = epoch - warmup_epochs
        if dec_epoch < dec_warmup_len:
            return max_weight * (dec_epoch / dec_warmup_len)
        else:
            return max_weight

class FileLogger:
    def __init__(self, output_file):
        self.output_file = output_file

    def write(self, msg, p=True):
        with open(self.output_file, mode="a", encoding="utf-8") as log_file:
            log_file.writelines(msg + '\n')
        if p:
            print(msg)


def evaluate(label, pred,n_cluster=None):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, n_cluster)
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, f, acc


def _make_cost_matrix(confusion_matrix):
    if not isinstance(confusion_matrix, np.ndarray):
        confusion_matrix = np.array(confusion_matrix)
    return (np.max(confusion_matrix) - confusion_matrix).tolist()


def get_y_preds(y_true, y_pred, n_clusters):
    max_label = int(max(y_true.max(), y_pred.max()) + 1)
    actual_size = max(n_clusters, max_label)
    w = np.bincount(y_pred * actual_size + y_true, minlength=actual_size**2).reshape(actual_size, actual_size)

    cost_matrix = _make_cost_matrix(w)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    map_dict = dict(zip(row_ind, col_ind))
    
    y_adjusted = np.array([map_dict[i] for i in y_pred])
    return y_adjusted
