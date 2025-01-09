# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json

import math
import torch
import torch.nn.functional as F
import pandas as pd
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
from unicore.losses.cross_entropy import CrossEntropyLoss
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
import warnings
from sklearn.metrics import top_k_accuracy_score
from rdkit.ML.Scoring.Scoring import CalcBEDROC
import random
import scipy.stats as stats
from scipy import stats


def calculate_bedroc(y_true, y_score, alpha):
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
    # print(scores.shape, y_true.shape)
    scores = np.concatenate((scores, y_true), axis=1)
    # inverse sort scores based on first column
    scores = scores[scores[:, 0].argsort()[::-1]]
    bedroc = CalcBEDROC(scores, 1, 80.5)
    return bedroc


@register_loss("rank_softmax")
class RSLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(
            **sample["pocket"]["net_input"],
            **sample["lig"]["net_input"],
            protein_sequences=sample["protein"],
            features_only=True,
            fix_encoder=fix_encoder,
            is_train=self.training
        )

        if self.training:
            loss_dict_accum = None
            for pocket_emb in net_output[0]:
                for mol_emb in net_output[1]:
                    logit_output = torch.matmul(pocket_emb, torch.transpose(mol_emb, 0, 1))
                    logit_output = logit_output * model.logit_scale.exp().detach()
                    loss_dict = self.compute_loss(model, logit_output, sample, reduce=reduce)
                    if loss_dict_accum is None:
                        loss_dict_accum = loss_dict
                    else:
                        for k in loss_dict_accum.keys():
                            loss_dict_accum[k] = loss_dict_accum[k] + loss_dict[k]
        else:
            pocket_emb = net_output[0][0]
            mol_emb = net_output[1][0]
            logit_output = torch.matmul(pocket_emb, torch.transpose(mol_emb, 0, 1))
            logit_output = logit_output * model.logit_scale.exp().detach()
            loss_dict_accum = {"loss": torch.tensor(0., device=logit_output.device)}

        if not self.training:
            # For training
            if self.args.valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
                sample_size = logit_output.size(0)
                logging_output = {
                    "loss": loss_dict_accum["loss"].data,
                    "logit_output": logit_output,
                    "act_list": sample["act_list"],
                    "batch_list": sample["batch_list"],
                    "smi_name": sample["lig"]["smi_name"],
                    "sample_size": sample_size,
                    "assay_id_list": sample["assay_id_list"],
                    "bsz": logit_output.size(0),
                    "scale": model.logit_scale.data
                }
            else:
                sample_size = logit_output.size(0)
                targets = torch.arange(sample_size, dtype=torch.long).cuda()
                assert logit_output.size(1) == sample_size
                logit_output = logit_output[:, :sample_size]
                probs = F.softmax(logit_output.float(), dim=-1).view(
                    -1, logit_output.size(-1)
                )
                logging_output = {
                    "loss": loss_dict_accum["loss"].data,
                    "prob": probs.data,
                    "target": targets,
                    "smi_name": sample["lig"]["smi_name"],
                    "sample_size": sample_size,
                    "bsz": logit_output.size(0),
                    "scale": model.logit_scale.data
                }
        else:
            sample_size = logit_output.size(0)
            logging_output = {
                "loss": loss_dict_accum["loss"].data,
                "loss_pocket": loss_dict_accum["loss_pocket"].data,
                "loss_mol": loss_dict_accum["loss_mol"].data,
                "loss_rank": loss_dict_accum["loss_rank"].data,
                "sample_size": sample_size,
                "bsz": logit_output.size(0),
                "scale": model.logit_scale.data
            }
        return loss_dict_accum["loss"], sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        batch_list = sample["batch_list"]
        act_list = sample["act_list"]
        smi_list = sample["lig"]["smi_name"]
        uniprotid_list = sample["uniprot_list"]
        net_output = net_output.float()
        num_pocket = net_output.shape[0]
        num_lig = net_output.shape[1]
        idx2pocket = []

        uniprotid_mask = torch.zeros_like(net_output)
        mol_mask = torch.zeros_like(net_output)
        for i in range(num_pocket):
            range_i = batch_list[i]
            idx2pocket += [i] * (range_i[1] - range_i[0])
            smi_pocket_i = smi_list[range_i[0]: range_i[1]]
            for j in range(num_pocket):
                if j == i:
                    continue
                range_j = batch_list[j]

                # mols of other pocket with the same uniprot id should be ignored
                if uniprotid_list[i] == uniprotid_list[j]:
                    uniprotid_mask[i, range_j[0]:range_j[1]] = -1e9

                # mols of other pocket with the same smi should be ignored
                for k in range(range_j[0], range_j[1]):
                    if smi_list[k] in smi_pocket_i:
                        mol_mask[i, k] = -1e9

        net_output = net_output + uniprotid_mask + mol_mask

        def pcc(x, y):
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        # pocket retrieve mol
        loss_mol = []
        loss_rank = []
        loss_corr = []
        for i in range(num_pocket):
            range_i = batch_list[i]
            act_list_i = act_list[i]
            for k in range(range_i[0], range_i[1]):
                mask = torch.zeros_like(net_output[i])
                mask[range_i[0]: range_i[1]] = -1e9
                mask[k] = 0.
                lprobs_mol = F.log_softmax(mask + net_output[i], dim=-1)
                loss_tmp = F.nll_loss(
                    lprobs_mol,
                    torch.tensor(k).cuda(),
                    reduction="sum" if reduce else "none",
                )
                if range_i[1] - range_i[0] > 1 and act_list_i[k - range_i[0]] < 5:
                    continue
                loss_mol.append(loss_tmp / math.sqrt(range_i[1] - range_i[0]))

            if range_i[1] - range_i[0] > 2:
                output_i = net_output[i, range_i[0]:range_i[1]]
                act_list_i = act_list[i]
                for k in range(range_i[1] - range_i[0] - 1):
                    mask = torch.zeros_like(output_i)
                    # mask[:k] = -1e9
                    for idx in range(0, range_i[1] - range_i[0]):
                        if idx == k:
                            continue
                        if act_list_i[k] - math.log10(3) <= act_list_i[idx]:  # three times
                            mask[idx] = -1e9
                    lprobs_mol = F.log_softmax(mask + output_i, dim=-1)
                    loss_tmp = F.nll_loss(
                        lprobs_mol,
                        torch.tensor(k).cuda(),
                        reduction="sum" if reduce else "none",
                    )
                    loss_rank.append(loss_tmp / (math.log(k + 2) * math.sqrt(range_i[1] - range_i[0])))

                corr = pcc(output_i, torch.tensor(act_list_i).to(output_i.device))
                if not torch.isnan(corr):
                    loss_corr.append(1. - corr)

        loss_mol = torch.stack(loss_mol).sum()

        lprobs_pocket = F.log_softmax(torch.transpose(net_output, 0, 1), dim=-1)
        lprobs_pocket = lprobs_pocket.view(-1, lprobs_pocket.size(-1))
        targets = torch.tensor(idx2pocket, dtype=torch.long).view(-1).cuda()
        loss_pocket_ = F.nll_loss(
            lprobs_pocket,
            targets,
            reduction="none",
        )
        loss_pocket = []
        for i in range(num_pocket):
            range_i = batch_list[i]
            if range_i[1] - range_i[0] > 0:
                loss_pocket.append(loss_pocket_[range_i[0]:range_i[1]].sum() / math.sqrt(range_i[1] - range_i[0]))
        loss_pocket = torch.stack(loss_pocket).sum()

        if self.args.few_shot:
            loss_rank = loss_corr
            self.args.contras_weight = 0.

        loss = self.args.contras_weight * (loss_pocket + loss_mol)

        if len(loss_rank) > 0:
            loss_rank = torch.stack(loss_rank).sum()
            loss = loss + self.args.rank_weight * loss_rank
        else:
            loss_rank = 0. * loss

        return {"loss": loss,
                "loss_pocket": loss_pocket,
                "loss_mol": loss_mol,
                "loss_rank": loss_rank}

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid", args=None) -> None:
        """Aggregate logging outputs from data parallel training."""
        metrics.log_scalar("scale", logging_outputs[0].get("scale"), round=3)
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        valid_set = args.valid_set
        split_method = args.split_method
        if "train" in split:
            for key in ["loss_mol", "loss_pocket", "loss_rank"]:
                loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(
                    key, loss_sum / sample_size, sample_size, round=3
                )
        elif valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
            corrs = []
            pearsons = []
            r2s = []
            res_dict = {}
            info_dict = {}
            for log in logging_outputs:
                logit_output = log["logit_output"].detach().cpu().numpy()
                true_act = log["act_list"]
                lig_smi = log["smi_name"]
                for i, (assay_id, span, acts) in enumerate(zip(log["assay_id_list"], log["batch_list"], true_act)):
                    acts = np.array(acts)
                    if len(acts) >= 3:
                        pred_score = logit_output[i, span[0]:span[1]]
                        corr = stats.spearmanr(acts, pred_score).statistic
                        pearson = stats.pearsonr(acts, pred_score).statistic
                        if math.isnan(corr):
                            corr = 0.
                        if math.isnan(pearson):
                            pearson = 0.
                        assay_smi = lig_smi[span[0]:span[1]]
                        res_dict[assay_id] = {
                            "assay_id": assay_id,
                            "pred": [round(x, 3) for x in pred_score.tolist()],
                            "exp": [round(x, 3) for x in acts.tolist()],
                            "spearmanr": corr,
                            "pearson": pearson
                        }
                        info_dict[assay_id] = {
                            "assay_id": assay_id,
                            "smiles": assay_smi,
                        }
                        corrs.append(corr)
                        pearsons.append(pearson)
                        r2s.append(max(pearson, 0) ** 2)
                        # print(pearson, len(acts))

            metrics.log_scalar(f"{split}_mean_corr", np.mean(corrs), sample_size, round=3)
            metrics.log_scalar(f"{split}_mean_pearson", np.mean(pearsons), sample_size, round=3)
            metrics.log_scalar(f"{split}_mean_r2", np.mean(r2s), sample_size, round=3)
            sup_num = float(args.sup_num)
            if args.sup_num > 1:
                sup_num = int(args.sup_num)

            import os
            rank = int(os.environ["LOCAL_RANK"])
            if rank == 0 and args.few_shot:
                if args.results_path.endswith(".jsonl"):
                    write_file = args.results_path
                else:
                    write_file = f"{args.results_path}/{split_method}_{args.seed}_sup{sup_num}.jsonl"
                    if args.active_learning_resfile != "":
                        write_file = f"{args.results_path}/{args.active_learning_resfile}"
                import os
                if not os.path.exists(write_file):
                    with open(write_file, "a") as f:
                        f.write(json.dumps(info_dict) + "\n")
                with open(write_file, "a") as f:
                    f.write(json.dumps(res_dict) + "\n")
                print(f"saving to {write_file}")

        else:
            acc_sum = sum(sum(log.get("prob").argmax(dim=-1) == log.get("target")) for log in logging_outputs)

            prob_list = []
            if len(logging_outputs) == 1:
                prob_list.append(logging_outputs[0].get("prob"))
            else:
                for i in range(len(logging_outputs) - 1):
                    prob_list.append(logging_outputs[i].get("prob"))
            probs = torch.cat(prob_list, dim=0)

            metrics.log_scalar(f"{split}_acc", acc_sum / sample_size, sample_size, round=3)
            metrics.log_scalar("valid_neg_loss", -loss_sum / sample_size / math.log(2), sample_size, round=3)
            targets = torch.cat([log.get("target", 0) for log in logging_outputs], dim=0)
            # print(targets.shape, probs.shape)

            targets = targets[:len(probs)]
            bedroc_list = []
            auc_list = []
            for i in range(len(probs)):
                prob = probs[i]
                target = targets[i]
                label = torch.zeros_like(prob)
                label[target] = 1.0
                cur_auc = roc_auc_score(label.cpu(), prob.cpu())
                auc_list.append(cur_auc)
                bedroc = calculate_bedroc(label.cpu(), prob.cpu(), 80.5)
                bedroc_list.append(bedroc)
            bedroc = np.mean(bedroc_list)
            auc = np.mean(auc_list)

            top_k_acc = top_k_accuracy_score(targets.cpu(), probs.cpu(), k=3, normalize=True)
            metrics.log_scalar(f"{split}_auc", auc, sample_size, round=3)
            metrics.log_scalar(f"{split}_bedroc", bedroc, sample_size, round=3)
            metrics.log_scalar(f"{split}_top3_acc", top_k_acc, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train


@register_loss("mseloss")
class MSELoss(RSLoss):
    def __init__(self, task):
        super().__init__(task)

    def forward(self, model, sample, reduce=True, fix_encoder=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        net_output = model(
            **sample["pocket"]["net_input"],
            **sample["lig"]["net_input"],
            protein_sequences=sample["protein"],
            batch_list=None,
            features_only=True,
            fix_encoder=fix_encoder,
            is_train=self.training
        )

        pred_actvity = net_output + 6.
        if self.training:
            loss = self.compute_loss(model, pred_actvity, sample, reduce=reduce)
        else:
            loss = torch.tensor(0., device=pred_actvity.device)

        if not self.training:
            # For training
            # print(pred_actvity)
            sample_size = 1
            if self.args.valid_set in ["FEP", "TIME", "TYK2", "OOD", "DEMO"]:
                logging_output = {
                    "loss": loss.data,
                    "logit_output": pred_actvity,
                    "act_list": sample["act_list"],
                    "batch_list": sample["batch_list"],
                    "smi_name": sample["lig"]["smi_name"],
                    "sample_size": sample_size,
                    "assay_id_list": sample["assay_id_list"],
                    "bsz": sample_size,
                    "scale": model.logit_scale.data
                }
            else:
                sample_size = pred_actvity.size(0)
                targets = torch.arange(sample_size, dtype=torch.long).cuda()
                assert pred_actvity.size(0) == pred_actvity.size(1)
                probs = F.softmax(pred_actvity.float(), dim=-1).view(
                    -1, pred_actvity.size(-1)
                )
                # print(probs)
                logging_output = {
                    "loss": loss.data,
                    "prob": probs.data,
                    "target": targets,
                    "smi_name": sample["lig"]["smi_name"],
                    "sample_size": sample_size,
                    "bsz": sample_size,
                    "scale": model.logit_scale.data
                }
        else:
            sample_size = 1
            logging_output = {
                "loss": loss.data,
                "sample_size": sample_size,
                "bsz": sample_size,
                "scale": model.logit_scale.data
            }
        return loss, sample_size, logging_output

    def compute_loss(self, model, pred_act, sample, reduce=True):
        batch_list = sample["batch_list"]
        act_list = sample["act_list"]
        smi_list = sample["lig"]["smi_name"]
        uniprotid_list = sample["uniprot_list"]
        net_output = pred_act.float()
        num_pocket = net_output.shape[0]
        num_lig = net_output.shape[1]
        idx2pocket = []

        uniprotid_mask = torch.zeros_like(net_output)
        mol_mask = torch.zeros_like(net_output)
        for i in range(num_pocket):
            range_i = batch_list[i]
            idx2pocket += [i] * (range_i[1] - range_i[0])
            smi_pocket_i = smi_list[range_i[0]: range_i[1]]
            for j in range(num_pocket):
                if j == i:
                    continue
                range_j = batch_list[j]

                # mols of other pocket with the same uniprot id should be ignored
                if uniprotid_list[i] == uniprotid_list[j]:
                    uniprotid_mask[i, range_j[0]:range_j[1]] = -1e9

                # mols of other pocket with the same smi should be ignored
                for k in range(range_j[0], range_j[1]):
                    if smi_list[k] in smi_pocket_i:
                        mol_mask[i, k] = -1e9

        # pocket retrieve mol
        loss_mse_all = []
        for i in range(num_pocket):
            range_i = batch_list[i]
            act_list_i = act_list[i]
            min_act = np.min(act_list_i)
            if len(act_list_i) > 1:
                min_act = max(min_act, 5)
            num_lig_i = range_i[1] - range_i[0]
            for j in range(num_pocket):
                for k in range(range_i[0], range_i[1]):
                    if i == j:
                        real_act = act_list_i[k - range_i[0]]
                        loss_mse = (net_output[j, k] - real_act) ** 2
                        loss_mse_all.append(loss_mse / math.sqrt(num_lig_i))
                    else:
                        if mol_mask[j, k] < 0:
                            continue
                        else:
                            # sample 10% of negative sample for speeding up training
                            # pos:neg = 1:2.4 after sampling
                            if random.random() > 0.1:
                                continue
                            neg_upper = min_act - self.args.neg_margin
                            loss_mse = (torch.clamp(net_output[j, k], min=neg_upper, max=None) - neg_upper) ** 2
                            loss_mse_all.append(loss_mse / math.sqrt(num_lig_i))

        loss = torch.stack(loss_mse_all).sum() / num_pocket
        return loss

