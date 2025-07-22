import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import random
import copy

class PPEncoder(nn.Module):

    def __init__(self, pocket_encoder, embed_dim, pocket_graph=None, aggregator=None, assayid_lst_all=[], assayid_lst_train=[], base_model=None, cuda="cpu"):
        super(PPEncoder, self).__init__()

        self.pocket_encoder = pocket_encoder
        self.pocket_graph = pocket_graph
        self.aggregator = aggregator
        if base_model != None:
            self.base_model = base_model
        self.embed_dim = embed_dim
        self.device = cuda
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.assayid_lst_all, self.assayid_set_train = assayid_lst_all, set(assayid_lst_train)
        self.assayid2idxes = {}
        for idx, assayid in enumerate(assayid_lst_all):
            if assayid not in self.assayid2idxes:
                self.assayid2idxes[assayid] = []
            self.assayid2idxes[assayid].append(idx)

    def forward(self, nodes_pocket, nodes_lig=None, max_sample=10):
        to_neighs = []

        for node in nodes_pocket:
            assayid = self.assayid_lst_all[node]
            neighbors = []
            nbr_pockets = self.pocket_graph.get(assayid, [])
            for n_assayid, score in nbr_pockets:
                if n_assayid == assayid:
                    continue
                if n_assayid not in self.assayid_set_train:
                    continue
                neighbors.append((random.choices(self.assayid2idxes[n_assayid])[0], score))
            to_neighs.append(neighbors)

        neigh_feats = self.aggregator.forward(nodes_pocket, to_neighs)  # user-user network
        self_feats = self.pocket_encoder(nodes_pocket, nodes_lig, max_sample)

        return (self_feats + neigh_feats) / 2
    
    def refine_pocket(self, pocket_embed, neighbor_list=None):
        neigh_feats = self.aggregator.forward_inference(pocket_embed, neighbor_list)
        self_feats = self.pocket_encoder.refine_pocket(pocket_embed, neighbor_list)
        return (self_feats + neigh_feats) / 2