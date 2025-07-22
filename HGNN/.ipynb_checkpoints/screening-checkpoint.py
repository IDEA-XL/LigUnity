import json
import random

import numpy as np
import torch
import torch.nn as nn
from PL_Encoder import PLEncoder
from PL_Aggregator import PLAggregator
from PP_Encoder import PPEncoder
from PP_Aggregator import PPAggregator
import torch.nn.functional as F
import torch.utils.data
import argparse
import os
from util import cal_metrics
from read_fasta import read_fasta_from_pocket, read_fasta_from_protein
from align import get_neighbor_pocket


class HGNN(nn.Module):

    def __init__(self, enc_u, enc_v=None, r2e=None):
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

    def refine_pocket(self, pocket_embed, neighbor_pocket_list):
        embeds_u = self.enc_u.refine_pocket(pocket_embed, neighbor_pocket_list)
        return embeds_u



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='HGNN model inference')
    parser.add_argument('--embed_dim', type=int, default=128, metavar='N', help='embedding size')
    parser.add_argument("--test_ckpt", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--result_root", type=str, default="../result/pocket_ranking")
    parser.add_argument("--pocket_embed", type=str, default="../example/pocket_embed.npy")
    parser.add_argument("--save_file", type=str, default="../example/refined_pocket.npy")
    parser.add_argument("--pocket_pdb", type=str, default=None)
    parser.add_argument("--protein_pdb", type=str, default="../example/protein.pdb")
    parser.add_argument("--ligand_pdb", type=str, default="../example/ligand.pdb")
    
    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    embed_dim = args.embed_dim
    type2e = nn.Embedding(10, embed_dim).to(device)

    # load model
    agg_pocket = PLAggregator(r2e=type2e, embed_dim=embed_dim, cuda=device, uv=True)
    enc_pocket = PLEncoder(embed_dim=embed_dim, aggregator=agg_pocket, cuda=device, uv=True)
    agg_pocket_sim = PPAggregator(embed_dim=embed_dim, cuda=device)
    enc_pocket = PPEncoder(enc_pocket, embed_dim=embed_dim, aggregator=agg_pocket_sim, cuda=device)
    
    model = HGNN(enc_pocket).to(device)
    model.load_state_dict(torch.load(args.test_ckpt, weights_only=True), strict=False)
    model.eval()

    # load pocket embedding and fasta
    pocket_embed = torch.tensor(np.load(args.pocket_embed)).to(device)

    if args.pocket_pdb is not None:
        pocket_fasta = read_fasta_from_pocket(args.pocket_pdb)
    else:
        pocket_fasta = read_fasta_from_protein(args.protein_pdb, args.ligand_pdb)
    
    # get neighbor pocket
    neighbor_pocket_list = get_neighbor_pocket(pocket_fasta, args.data_root, args.result_root, device) # [(pocket_embed, ligand_embed, similarity)]

    # get refined pocket
    if len(neighbor_pocket_list) > 0:
        with torch.no_grad():
            refined_pocket = model.refine_pocket(pocket_embed, neighbor_pocket_list)
            refined_pocket = refined_pocket.cpu().numpy()
    else:
        refined_pocket = pocket_embed.cpu().numpy()

    print("finished, saving refined pocket embedding into:", args.save_file)
    np.save(args.save_file, refined_pocket)
    

if __name__ == "__main__":
    main()
