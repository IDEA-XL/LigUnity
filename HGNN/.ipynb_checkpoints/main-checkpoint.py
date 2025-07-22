import json
import random

import numpy as np
import torch
import torch.nn as nn
from PL_Encoder import PLEncoder
from PL_Aggregator import PLAggregator
from PP_Encoder import PPEncoder
from PP_Aggregator import PPAggregator
from screen_dataset import *
import torch.nn.functional as F
import torch.utils.data
import argparse
import os
from util import cal_metrics


class HGNN(nn.Module):

    def __init__(self, enc_u, enc_v, r2e):
        super(HGNN, self).__init__()
        self.enc_u = enc_u
        self.enc_v = enc_v
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)

        self.logit_scale = nn.Parameter(torch.ones([1], device="cuda") * np.log(14))

    def trainable_parameters(self):
        for name, param in self.named_parameters(recurse=True):
            if param.requires_grad:
                yield param

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u, nodes_v)
        embeds_v = self.enc_v(nodes_v)
        return embeds_u, embeds_v

    def criterion(self, x_u, x_v, labels):

        netout = torch.matmul(x_u, torch.transpose(x_v, 0, 1))
        score = netout * self.logit_scale.exp().detach()
        score = (labels - torch.eye(len(labels)).to(labels.device)) * -1e6 + score

        lprobs_pocket = F.log_softmax(score.float(), dim=-1)
        lprobs_pocket = lprobs_pocket.view(-1, lprobs_pocket.size(-1))
        sample_size = lprobs_pocket.size(0)
        targets = torch.arange(sample_size, dtype=torch.long).view(-1).cuda()

        # pocket retrieve mol
        loss_pocket = F.nll_loss(
            lprobs_pocket,
            targets,
            reduction="mean"
        )

        lprobs_mol = F.log_softmax(torch.transpose(score.float(), 0, 1), dim=-1)
        lprobs_mol = lprobs_mol.view(-1, lprobs_mol.size(-1))
        lprobs_mol = lprobs_mol[:sample_size]

        # mol retrieve pocket
        loss_mol = F.nll_loss(
            lprobs_mol,
            targets,
            reduction="mean"
        )

        loss = 0.5 * loss_pocket + 0.5 * loss_mol

        ef_all = []
        for i in range(len(netout)):
            act_pocket = labels[i]
            affi_pocket = netout[i]
            top1_index = torch.argmax(affi_pocket)
            top1_act = act_pocket[top1_index]
            ef_all.append(cal_metrics(affi_pocket.detach().cpu().numpy(), act_pocket.detach().cpu().numpy()))
        ef_mean = {k: np.mean([x[k] for x in ef_all]) for k in ef_all[0].keys()}

        return loss, ef_mean, netout

    def loss(self, nodes_u, nodes_v, labels):
        x_u, x_v = self.forward(nodes_u, nodes_v)
        loss, ef_mean, netout = self.criterion(x_u, x_v, labels)
        return loss, ef_mean


def train(model, device, train_loader, optimizer, epoch, valid_idxes, valid_molidxes, valid_labels):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels = data
        optimizer.zero_grad()
        loss, _ = model.loss(batch_nodes_u[0].to(device), batch_nodes_v[0].to(device), labels[0].to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 0:
            print('[%d, %5d] loss: %.3f '%(epoch, i, running_loss / 200))
            running_loss = 0.0
            avg_loss, avg_acc = valid(model,
                                     device,
                                     torch.tensor(valid_idxes).to(device),
                                     torch.tensor(valid_molidxes).to(device),
                                     torch.tensor(valid_labels).to(device))
            print('Valid set results:', avg_loss.item(), avg_acc)
    return 0


def valid(model, device, valid_idxes, valid_molidxes, valid_labels):
    model.eval()
    with torch.no_grad():
        loss, ef = model.loss(valid_idxes.to(device), valid_molidxes.to(device), valid_labels.to(device))
    model.train()
    return loss, ef


def test_dekois(model, device, epoch, result_root, dekois_pocket_name, dekois_idxes):
    model.eval()
    loss_all, ef_all = [], []
    loss_raw_all, ef_raw_all = [], []
    dekois_dir = f"{result_root}/DEKOIS"
    with torch.no_grad():
        for dekois_id, pocket_node_id in zip(dekois_pocket_name, dekois_idxes):
            embeds_pocket = model.enc_u([pocket_node_id], None, max_sample=-1)
            embeds_lig = torch.tensor(np.load(f"{dekois_dir}/{dekois_id}/saved_mols_embed.npy")).to(device).float()
            labels = np.load(f"{dekois_dir}/{dekois_id}/saved_labels.npy")
            embeds_pocket_raw = model.enc_u.aggregator.u2e(torch.tensor([pocket_node_id]).to(device))

            score = torch.matmul(embeds_pocket, torch.transpose(embeds_lig, 0, 1)).squeeze().detach().cpu().numpy()
            score_raw = torch.matmul(embeds_pocket_raw, torch.transpose(embeds_lig, 0, 1)).squeeze().detach().cpu().numpy()
            np.save(f"{dekois_dir}/{dekois_id}/GNN_res_epoch{epoch}.npy", score)
            np.save(f"{dekois_dir}/{dekois_id}/noGNN_res.npy", score_raw)
            metric = cal_metrics(score, labels)
            metric_raw = cal_metrics(score_raw, labels)
            # print(dekois_id, metric["EF1"], metric["BEDROC"], metric["AUC"])
            ef_all.append(metric)
            ef_raw_all.append(metric_raw)

    model.train()
    ef_all = {k: np.mean([x[k] for x in ef_all]) for k in ef_all[0].keys()}
    ef_raw_all = {k: np.mean([x[k] for x in ef_raw_all]) for k in ef_raw_all[0].keys()}
    print('Test on dekois:', ef_all)
    print('No HGNN on dekois:', ef_raw_all)

def test_dude(model, device, epoch, result_root, dude_pocket_name, dude_idxes):
    model.eval()
    loss_all, ef_all = [], []
    loss_raw_all, ef_raw_all = [], []
    dude_dir = f"{result_root}/DUDE"
    with torch.no_grad():
        for dude_id, pocket_node_id in zip(dude_pocket_name, dude_idxes):
            embeds_pocket = model.enc_u([pocket_node_id], None, max_sample=-1)
            embeds_lig = torch.tensor(np.load(f"{dude_dir}/{dude_id}/saved_mols_embed.npy")).to(device).float()
            labels = np.load(f"{dude_dir}/{dude_id}/saved_labels.npy")
            embeds_pocket_raw = model.enc_u.aggregator.u2e(torch.tensor([pocket_node_id]).to(device))

            score = torch.matmul(embeds_pocket, torch.transpose(embeds_lig, 0, 1)).squeeze().detach().cpu().numpy()
            score_raw = torch.matmul(embeds_pocket_raw, torch.transpose(embeds_lig, 0, 1)).squeeze().detach().cpu().numpy()
            np.save(f"{dude_dir}/{dude_id}/GNN_res_epoch{epoch}.npy", score)
            np.save(f"{dude_dir}/{dude_id}/noGNN_res.npy", score_raw)
            metric = cal_metrics(score, labels)
            metric_raw = cal_metrics(score_raw, labels)
            # print(dude_id, metric["EF1"], metric["BEDROC"], metric["AUC"])
            ef_all.append(metric)
            ef_raw_all.append(metric_raw)

    model.train()
    ef_all = {k: np.mean([x[k] for x in ef_all]) for k in ef_all[0].keys()}
    ef_raw_all = {k: np.mean([x[k] for x in ef_raw_all]) for k in ef_raw_all[0].keys()}
    print('Test on dude:', ef_all)
    print('No HGNN on dude:', ef_raw_all)

def test_pcba(model, device, epoch, result_root, pcba_idxes):
    model.eval()
    loss_all, ef_all = [], []
    loss_raw_all, ef_raw_all = [], []
    pcba_dir = f"{result_root}/PCBA"
    with torch.no_grad():
        pocket_idx = 0
        for pcba_id in sorted(list(os.listdir(pcba_dir))):
            pocket_names = []
            for names in json.load(open(f"{pcba_dir}/{pcba_id}/saved_pocket_names.json")):
                pocket_names += names
            embeds_lig = torch.tensor(np.load(f"{pcba_dir}/{pcba_id}/saved_mols_embed.npy")).to(device).float()
            labels = np.load(f"{pcba_dir}/{pcba_id}/saved_labels.npy")
            score_all_pocket = []
            score_raw_pocket = []

            for i, pocket_name in enumerate(pocket_names):
                pcba_test_idx = pcba_idxes[pocket_idx]
                embeds_pocket = model.enc_u([pcba_test_idx], None, max_sample=-1)
                netout = torch.matmul(embeds_pocket, torch.transpose(embeds_lig, 0, 1))
                embeds_pocket_raw = model.enc_u.aggregator.u2e(torch.tensor([pcba_test_idx]).to(device))
                netout_raw = torch.matmul(embeds_pocket_raw, torch.transpose(embeds_lig, 0, 1))
                score_all_pocket.append(netout.squeeze().detach().cpu().numpy())
                score_raw_pocket.append(netout_raw.squeeze().detach().cpu().numpy())
                pocket_idx += 1

            score_max = np.stack(score_all_pocket, axis=0).mean(axis=0)
            score_raw_max = np.stack(score_raw_pocket, axis=0).max(axis=0)
            metric = cal_metrics(score_max, labels)
            print(pcba_id, metric["EF1"], metric["BEDROC"], metric["AUC"])
            np.save(f"{pcba_dir}/{pcba_id}/GNN_res_epoch{epoch}.npy", score_max)
            np.save(f"{pcba_dir}/{pcba_id}/noGNN_res.npy", score_raw_max)
            ef_all.append(cal_metrics(score_max, labels))
            ef_raw_all.append(cal_metrics(score_raw_max, labels))

    model.train()
    print(f"saving to {pcba_dir}")
    ef_all = {k: np.mean([x[k] for x in ef_all]) for k in ef_all[0].keys()}
    ef_raw_all = {k: np.mean([x[k] for x in ef_raw_all]) for k in ef_raw_all[0].keys()}
    print('Test on pcba:', ef_all)
    print('No HGNN on pcba:', ef_raw_all)
    return ef_all["EF1"]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='HGNN model training')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--embed_dim', type=int, default=128, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train')
    parser.add_argument("--test_ckpt", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--result_root", type=str, default="../result/pocket_ranking")
    args = parser.parse_args()
    data_root = args.data_root

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    print("begin load dataset")
    assayinfo_lst, pocket_feat, mol_feat, assayid_lst_all, mol_smi_lst, \
        assayid_lst_train, assayid_lst_test, dude_pocket_name, pcba_pocket_name, dekois_pocket_name, valid_molidxes = load_datas(data_root, result_root)
    print("begin load pocket-pocket graph")
    pocket_graph = load_pocket_pocket_graph(data_root, assayid_lst_all, assayid_lst_train)

    screen_dataset = ScreenDataset(args.batch_size, pocket_graph, assayinfo_lst, assayid_lst_all, mol_smi_lst, assayid_lst_train)
    num_pockets = len(assayid_lst_all)
    num_ligs = mol_feat.shape[0]

    embed_dim = args.embed_dim
    pocket2e = nn.Embedding(num_pockets, embed_dim).to(device)
    pocket2e.weight.data.copy_(torch.tensor(pocket_feat).to(device))
    for param in pocket2e.parameters():
        param.requires_grad = False

    lig2e = nn.Embedding(num_ligs, embed_dim).to(device)
    for param in lig2e.parameters():
        param.requires_grad = False
    type2e = nn.Embedding(10, embed_dim).to(device)

    agg_pocket = PLAggregator(lig2e, type2e, pocket2e, embed_dim, cuda=device, uv=True)
    enc_pocket = PLEncoder(embed_dim, pocket_graph, agg_pocket, assayid_lst_all, assayid_lst_train, mol_smi_lst, assayinfo_lst, cuda=device, uv=True)
    # neighobrs
    agg_pocket_sim = PPAggregator(pocket2e, embed_dim, cuda=device)
    enc_pocket = PPEncoder(enc_pocket, embed_dim, pocket_graph, agg_pocket_sim, assayid_lst_all, assayid_lst_train,
                           base_model=enc_pocket, cuda=device)
    enc_lig = lig2e
    # model
    graphrec = HGNN(enc_pocket, enc_lig, type2e).to(device)
    print("trainable parameters")
    for name, param in graphrec.named_parameters(recurse=True):
        if param.requires_grad:
            print(name, param.shape)
    optimizer = torch.optim.RMSprop(graphrec.trainable_parameters(), lr=args.lr, alpha=0.9)

    begin = len(assayid_lst_train+assayid_lst_test)
    end = begin + len(dude_pocket_name)
    dude_idxes = range(begin, end)
    begin = end
    end += len(pcba_pocket_name)
    pcba_idxes = range(begin, end)
    begin = end
    end += len(dekois_pocket_name)
    dekois_idxes = range(begin, end)

    if args.test_ckpt is not None:
        graphrec.load_state_dict(torch.load(args.test_ckpt, weights_only=True))
        test_dude(graphrec, device, 0, result_root, dude_pocket_name, dude_idxes)
        test_dekois(graphrec, device, 0, result_root, dekois_pocket_name, dekois_idxes)
        test_pcba(graphrec, device, 0, result_root, pcba_idxes)
    else:
        for epoch in range(args.epochs):
            screen_dataset.set_epoch(epoch)
            train_loader = torch.utils.data.DataLoader(screen_dataset, batch_size=1, shuffle=True, num_workers=8)
            lig2e.weight.data.copy_(torch.tensor(mol_feat).to(device))
            valid_labels = load_valid_label(assayid_lst_test)
            valid_idxes = range(len(assayid_lst_train), len(assayid_lst_train+assayid_lst_test))
            train(graphrec, device, train_loader, optimizer, epoch, valid_idxes, valid_molidxes, valid_labels)
            test_dude(graphrec, device, epoch+1, result_root, dude_pocket_name, dude_idxes)
            test_dekois(graphrec, device, epoch+1, result_root, dekois_pocket_name, dekois_idxes)
            test_pcba(graphrec, device, epoch+1, result_root, pcba_idxes)

            os.system(f"mkdir -p {result_root}/HGNN_save")
            torch.save(graphrec.state_dict(),f"{result_root}/HGNN_save/model_{epoch}.pt")

if __name__ == "__main__":
    main()
