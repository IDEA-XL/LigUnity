import sys
import os
import json
import copy
import numpy as np
import scipy.stats as stats
import math
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcAUC, CalcEnrichment

def cal_metrics(y_score, y_true):
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
    ef_list = CalcEnrichment(scores, 1, [0.005, 0.01, 0.02, 0.05])

    return {
        "BEDROC": bedroc,
        "AUROC": auc,
        "EF0.5": ef_list[0],
        "EF1": ef_list[1],
        "EF5": ef_list[3]
    }

def print_avg_metric(metric_dict, name):
    metric_lst = list(metric_dict.values())
    ret_metric = copy.deepcopy(metric_lst[0])
    for m in metric_lst[1:]:
        for k in m:
            ret_metric[k] += m[k]

    for k in ret_metric:
        ret_metric[k] = ret_metric[k] / len(metric_lst)
    print(name, ret_metric)

def read_zeroshot_res(res_dir):
    targets = sorted(list(os.listdir(res_dir)))
    res_dict = {}
    for target in targets:
        real_dg = np.load(f"{res_dir}/{target}/saved_labels.npy")
        if os.path.exists(f"{res_dir}/{target}/saved_preds.npy"):
            pred_dg = np.load(f"{res_dir}/{target}/saved_preds.npy")
        else:
            mol_reps = np.load(f"{res_dir}/{target}/saved_mols_embed.npy")
            pocket_reps = np.load(f"{res_dir}/{target}/saved_target_embed.npy")
            res = pocket_reps @ mol_reps.T
            pred_dg = res.max(axis=0)
        res_dict[target] = {
            "pred": pred_dg,
            "exp": real_dg
        }
    return res_dict

def get_ensemble_res(res_list, begin=0, end=-1):
    if end == -1:
        end = len(res_list)
    ret = copy.deepcopy(res_list[begin])
    for res in res_list[begin+1:end]:
        for k in ret.keys():
            ret[k]["pred"] = np.array(ret[k]["pred"]) + np.array(res[k]["pred"])

    for k in ret.keys():
        ret[k]["pred"] = np.array(ret[k]["pred"]) / (end-begin)

    return ret

def avg_metric(metric_lst_all):
    ret_metric_dict = {}
    for metric_lst in metric_lst_all:
        ret_metric = copy.deepcopy(metric_lst[0])
        for m in metric_lst[1:]:
            for k in ["pearsonr", "spearmanr", "r2"]:
                ret_metric[k] += m[k]
        for k in ["spearmanr", "pearsonr", "r2"]:
            ret_metric[k] = ret_metric[k] / len(metric_lst)
        ret_metric_dict[ret_metric["target"]] = ret_metric
    return ret_metric_dict

def get_metric(res):
    metric_dict = {}
    for k in sorted(list(res.keys())):
        pred = res[k]["pred"]
        exp = res[k]["exp"]
        spearmanr = stats.spearmanr(exp, pred).statistic
        pearsonr = stats.pearsonr(exp, pred).statistic
        if math.isnan(pearsonr):
            pearsonr = 0
        if math.isnan(spearmanr):
            spearmanr = 0
        metric_dict[k] = {
            "pearsonr":pearsonr,
            "spearmanr":spearmanr,
            "r2":max(pearsonr, 0)**2,
            "target":k
        }
    return metric_dict


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == "zeroshot":
        test_sets = sys.argv[2:]
        for test_set in test_sets:
            if test_set in ["DUDE", "PCBA", "DEKOIS"]:
                metrics = {}
                target_id_list = sorted(list(os.listdir(f"./result/pocket_ranking/{test_set}")))
                for target_id in target_id_list:
                    lig_act = np.load(f"./result/pocket_ranking/{test_set}/{target_id}/saved_labels.npy")
                    score_1 = np.load(f"./result/pocket_ranking/{test_set}/{target_id}/GNN_res_epoch9.npy")
                    score_2 = np.load(f"./result/protein_ranking/{test_set}/{target_id}/GNN_res_epoch9.npy")

                    score = (score_1 + score_2)/2
                    metrics[target_id] = cal_metrics(score, lig_act)

                json.dump(metrics, open(f"./result/pocket_ranking/{test_set}_metrics.json", "w"))
                print_avg_metric(metrics, "Ours")
            elif test_set in ["FEP"]:
                target_id_list = sorted(list(os.listdir(f"./result/pocket_ranking/{test_set}")))
                res_all_pocket, res_all_protein = [], []
                for repeat in range(1, 6):
                    res_pocket = read_zeroshot_res(f"./result/pocket_ranking/{test_set}/repeat_{repeat}")
                    res_protein = read_zeroshot_res(f"./result/protein_ranking/{test_set}/repeat_{repeat}")
                    res_all_pocket.append(res_pocket)
                    res_all_protein.append(res_protein)
                res_all_fusion = get_ensemble_res(res_all_pocket + res_all_protein)
                metrics = get_metric(res_all_fusion)
                json.dump(metrics, open(f"./result/pocket_ranking/{test_set}_metrics.json", "w"))
                print_avg_metric(metrics, "Ours")
    elif mode == "fewshot":
        test_set = sys.argv[2]
        support_num = sys.argv[3]
        begin = 15
        end = 20
        metric_fusion_all = []
        for seed in range(1, 11):
            res_repeat_pocket = []
            res_repeat_seq = []

            if test_set in ["TIME", "OOD"]:
                res_file_pocket = f"./result/pocket_ranking/{test_set}/random_{seed}_sup{support_num}.jsonl"
                res_file_seq = f"./result/pocket_ranking/{test_set}/random_{seed}_sup{support_num}.jsonl"
                if not os.path.exists(res_file_pocket):
                    continue
                res_repeat_pocket = [json.loads(line) for line in open(res_file_pocket)][1:]
                res_repeat_seq = [json.loads(line) for line in open(res_file_seq)][1:]
            elif test_set in ["FEP_fewshot"]:
                for repeat in range(1, 6):
                    res_file_pocket = f"./result/pocket_ranking/{test_set}/repeat_{repeat}/random_{seed}_sup{support_num}.jsonl"
                    res_file_seq = f"./result/pocket_ranking/{test_set}/repeat_{repeat}/random_{seed}_sup{support_num}.jsonl"
                    if not os.path.exists(res_file_pocket):
                        continue
                    res_pocket = [json.loads(line) for line in open(res_file_pocket)][1:]
                    res_seq = [json.loads(line) for line in open(res_file_seq)][1:]
                    res_pocket = get_ensemble_res(res_pocket, begin, end)
                    res_seq = get_ensemble_res(res_seq, begin, end)
                    res_repeat_pocket.append(res_pocket)
                    res_repeat_seq.append(res_seq)

            res_repeat_fusion = get_ensemble_res(res_repeat_pocket + res_repeat_seq)
            metric_fusion_all.append(get_metric(res_repeat_fusion))
        metric_fusion_all = avg_metric(list(map(list, zip(*metric_fusion_all))))
        json.dump(metric_fusion_all, open(f"./result/pocket_ranking/{test_set}_metrics.json", "w"))
        print_avg_metric(metric_fusion_all, "Ours")
