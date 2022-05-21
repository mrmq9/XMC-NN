# coding=utf-8
# @Author  : Mohammadreza Qaraei
# @Email   : mohammadreza.mohammadniaqaraei@aalto.fi


import os
import numpy as np


# def compute_inv_prop(data_path, inv_prop_path, a=0.55, b=1.5):
    
#     if os.path.exists(inv_prop_path):
#         print("Loading inverse propensity scores")
#         inv_prop = np.load(inv_prop_path)
#         return inv_prop

#     f = open(data_path, "r")
#     header = f.readline().split(" ")
#     num_samples = int(header[0])
#     num_labels = int(header[2])
    
#     inv_prop = np.zeros(num_labels)
#     print("Computing inverse propensity scores")
#     for _ in range(num_samples):
#         sample = f.readline().rstrip('\n')
#         labels = sample.split(" ",1)[0]
#         if labels == "":
#             continue
#         labels = [int(label) for label in labels.split(",")]
#         inv_prop[labels] += 1.0
#     f.close()
    

#     c = (np.log(num_samples) - 1) * np.power(b+1, a)
#     inv_prop = 1 + c * np.power(inv_prop + b, -a)
#     np.save(inv_prop_path, inv_prop)

#     return inv_prop

def fix_tr_pred(pred, tr, k):
    if tr is not np:
        tr = np.array(tr)  
    if len(tr.shape) > 1:
        tr = tr.squeeze(0)
    pred = pred[:k]
    return pred, tr

def precision(pred_mat, true_mat, k):
    assert len(pred_mat) == len(true_mat)
    correct_count = np.zeros(k, dtype=np.int)
    for pred, tr in zip(pred_mat, true_mat):
        pred, tr = fix_tr_pred(pred, tr, k)
        match= np.isin(pred, tr, assume_unique=True)
        correct_count += np.cumsum(match)

    precision = correct_count * 100.0 / (len(pred_mat) * np.arange(1, k+1))
    return precision



def ndcg(pred_mat, true_mat, k):
    assert len(pred_mat) == len(true_mat)
    correct_count = np.zeros(k)
    for pred, tr in zip(pred_mat, true_mat):
        pred, tr = fix_tr_pred(pred, tr, k)
        num = np.isin(pred, tr, assume_unique=True).astype(float)

        num[num>0] = 1.0/np.log((num>0).nonzero()[0]+2)
        
        den = np.zeros(k)
        den_size = min(tr.size, k)
        den[:den_size] = 1.0 / np.log(np.arange(1, den_size+1)+1)

        correct_count += np.cumsum(num) / np.cumsum(den)

    ndcg = correct_count * 100.0 / len(pred_mat)
    return ndcg



def psp(pred_mat, true_mat, inv_prop, k):
    assert len(pred_mat) == len(true_mat)
    num = np.zeros(k)
    den = np.zeros(k)
    for pred, tr in zip(pred_mat, true_mat):
        pred, tr = fix_tr_pred(pred, tr, k)

        match = np.isin(pred, tr, assume_unique=True).astype(float)
        match[match>0] = inv_prop[pred[match>0]]
        num += np.cumsum(match)

        inv_prop_sample = inv_prop[tr]
        inv_prop_sample = np.sort(inv_prop_sample)[::-1]

        match = np.zeros(k)
        match_size = min(tr.size, k)
        match[:match_size] = inv_prop_sample[:match_size]
        den += np.cumsum(match)

    psp = num * 100 / den
    return psp


def psndcg(pred_mat, true_mat, inv_prop, k):
    assert len(pred_mat) == len(true_mat)
    den = np.zeros(k)
    num = np.zeros(k)
    for pred, tr in zip(pred_mat, true_mat):
        pred, tr = fix_tr_pred(pred, tr, k)

        match = np.isin(pred, tr, assume_unique=True).astype(float)
        match[match>0] = inv_prop[pred[match>0]] / np.log2((match>0).nonzero()[0]+2)
        num += np.cumsum(match)

        match = np.zeros(k)
        match_size = min(tr.size, k)
        ind_large = np.argsort(inv_prop[tr])[::-1]
        temp_match = inv_prop[tr[ind_large]] / np.log2(np.arange(ind_large.size)+2) 
        match[:match_size] = temp_match[:match_size]
        den += np.cumsum(match)

    psndcg = num * 100.0 / den
    return psndcg