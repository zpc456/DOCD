import numpy as np
import os
import sys
import math
import torch
import json
import random
import pickle
from torch import nn
from torch.utils.data import TensorDataset, Dataset, DataLoader
from sklearn.metrics import (average_precision_score, auc, roc_curve, confusion_matrix, roc_auc_score, precision_recall_curve, recall_score, balanced_accuracy_score, precision_score, accuracy_score, f1_score)


def build_tree_with_padding(treeFile, max_len_ancestors=6):
    treeMap = pickle.load(open(treeFile, 'rb'))
    if len(treeMap) == 0:
        return [], [], []
    ancestors = np.array(list(treeMap.values())).astype('int32')
    ancSize = ancestors.shape[1]
    leaves = []
    for k in treeMap.keys():
        leaves.append([k] * ancSize)
    leaves = np.array(leaves).astype('int32')
    if ancSize < max_len_ancestors:
        ones = np.ones((ancestors.shape[0], ancSize)).astype('int32')
        zeros = np.zeros((ancestors.shape[0], max_len_ancestors - ancSize)).astype('int32')
        leaves = np.concatenate([leaves, zeros], axis=1)
        ancestors = np.concatenate([ancestors, zeros], axis=1)
        mask = np.concatenate([ones, zeros], axis=1)
    else:
        mask = np.ones((ancestors.shape[0], max_len_ancestors))
    return leaves, ancestors, mask


class CustomizedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def priors_collate_fn(batch):
    new_batch = []
    for i, item in enumerate(batch):
        num_indices = item[0].shape[-1]
        new_indices = torch.cat((torch.tensor([i] * num_indices).reshape(1, -1), item[0]), axis=0)
        new_batch.append((new_indices, item[1]))
    indices = torch.cat([t[0] for t in new_batch], axis=1)
    values = torch.cat([t[1] for t in new_batch], axis=-1)
    return indices, values


def get_extended_attention_mask(attention_mask):
    if attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    elif attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def prepare_data(data, priors_data, device):
    features = {}
    features['dx_ints'] = data[0]
    features['dx_masks'] = data[1]
    features['proc_ints'] = data[2]
    features['proc_masks'] = data[3]
    features['demographics_ints'] = data[4]
    features['vital_signs_ints'] = data[5]
    features['expired'] = data[6]
    features['dx_ccs_cat1'] = data[7]
    features['proc_ccs_cat1'] = data[8]
    for k, v in features.items():
        features[k] = v.to(device)

    priors = {}
    priors['indices'] = priors_data[0].to(device)
    priors['values'] = priors_data[1].to(device)
    return features, priors


def get_rootCode(treeFile):
    tree = pickle.load(open(treeFile, 'rb'))
    rootCode = list(tree.values())[0][1]
    return rootCode


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def nested_concat(tensors, new_tensors, dim=0):
    assert type(tensors) == type(new_tensors), f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, dim) for t, n in zip(tensors, new_tensors))
    return torch.cat((tensors, new_tensors), dim=dim)


def nested_numpify(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    return tensors.cpu().numpy()


def nested_detach(tensors):
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    return tensors.detach()


def compute_metrics(preds, ground_labels):
    metrics = {}

    preds_labels = np.argmax(preds, axis=1)

    preds = torch.tensor(preds)
    preds = nn.Softmax(dim=-1)(preds)
    preds_prob = preds[:, 1]

    confusion = confusion_matrix(ground_labels, preds_labels)
    metrics['TP'] = confusion[1][1]
    metrics['FP'] = confusion[0][1]
    metrics['FN'] = confusion[1][0]
    metrics['TN'] = confusion[0][0]

    metrics['accuracy'] = round(accuracy_score(ground_labels, preds_labels), 4)
    metrics['balanced_accuracy'] = round(balanced_accuracy_score(ground_labels, preds_labels), 4)
    metrics['precision'] = round(precision_score(ground_labels, preds_labels, zero_division=1), 4)
    metrics['average_precision'] = round(average_precision_score(ground_labels, preds_prob), 4)
    metrics['recall'] = round(recall_score(ground_labels, preds_labels), 4)
    metrics['f1_score'] = round(f1_score(ground_labels, preds_labels), 4)

    fpr, tpr, thresholds = roc_curve(ground_labels, preds_prob)
    auc_roc = auc(fpr, tpr)
    metrics['AUC-ROC'] = round(auc_roc, 4)

    precisions, recalls, thresholds = precision_recall_curve(ground_labels, preds_prob)
    auc_pr = auc(recalls, precisions)
    metrics['AUC-PR'] = round(auc_pr, 4)

    return metrics
