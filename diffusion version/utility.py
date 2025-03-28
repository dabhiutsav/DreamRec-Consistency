import numpy as np
import torch

def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

def calculate_hit(sorted_list,topk,true_items,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)