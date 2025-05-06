import numpy as np
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment
from sklearn.metrics import roc_curve

def re_new(y_true, y_score, ratio):
    fp = 0
    tp = 0
    p = sum(y_true)
    n = len(y_true) - p
    num = ratio * n
    sort_index = np.argsort(y_score)[::-1]
    for i in range(len(sort_index)):
        index = sort_index[i]
        if y_true[index] == 1:
            tp += 1
        else:
            fp += 1
            if fp >= num:
                break
    return (tp * n) / (p * fp)


def calc_re(y_true, y_score, ratio_list):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    # print(fpr, tpr)
    res = {}
    res2 = {}
    total_active_compounds = sum(y_true)
    total_compounds = len(y_true)

    # for ratio in ratio_list:
    #     for i, t in enumerate(fpr):
    #         if t > ratio:
    #             #print(fpr[i], tpr[i])
    #             if fpr[i-1]==0:
    #                 res[str(ratio)]=tpr[i]/fpr[i]
    #             else:
    #                 res[str(ratio)]=tpr[i-1]/fpr[i-1]
    #             break

    for ratio in ratio_list:
        res2[str(ratio)] = re_new(y_true, y_score, ratio)

    # print(res)
    # print(res2)
    return res2


def cal_metrics(y_score, y_true, alpha=80.5):
    """
    Calculate BEDROC score.

    Parameters:
    - y_true: true binary labels (0 or 1)
    - y_score: predicted scores or probabilities
    - alpha: parameter controlling the degree of early retrieval emphasis

    Returns:
    - BEDROC score
    """

    # concate res_single and labels
    scores = np.expand_dims(y_score, axis=1)
    y_true = np.expand_dims(y_true, axis=1)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    count = 0
    # sort y_score, return index
    index = np.argsort(y_score)[::-1]
    for i in range(int(len(index) * 0.005)):
        if y_true[index[i]] == 1:
            count += 1
    auc = CalcAUC(scores, 1)
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.05])
    return {
        "BEDROC": bedroc,
        "AUC": auc,
        "EF0.5": ef_list[0],
        "EF1": ef_list[1],
        "EF5": ef_list[2]
    }


# import torch
# torch.multiprocessing.set_start_method('spawn', force=True)
# def mycollator(input_batch):
#     for data in input_batch:
#         node, neighbors = data
#         node["pocket_data"] = torch.tensor(node["pocket_data"]).cuda()
#         node["lig_data"] = torch.tensor(node["lig_data"]).cuda()
#         for neighbor in neighbors:
#             neighbor["pocket_data"] = torch.tensor(node["pocket_data"]).cuda()
#             neighbor["lig_data"] = torch.tensor(node["lig_data"]).cuda()
#     return input_batch