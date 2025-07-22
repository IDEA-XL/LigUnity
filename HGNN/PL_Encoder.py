import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import random

class PLEncoder(nn.Module):

    def __init__(self, embed_dim, pocket_graph=None, aggregator=None, idx2assayid={}, assayid_lst_train=[], mol_smi={}, train_label_lst=[], cuda="cpu", uv=True):
        super(PLEncoder, self).__init__()

        self.uv = uv
        self.pocket_graph = pocket_graph
        self.aggregator = aggregator
        self.embed_dim = embed_dim
        self.device = cuda
        smi2idx = {smi:idx for idx, smi in enumerate(mol_smi)}
        self.idx2assayid, self.assayid_lst_train, self.smi2idx, self.mol_smi, self.train_label_lst = idx2assayid, assayid_lst_train, smi2idx, mol_smi, train_label_lst
        self.assayid_set_train = set(assayid_lst_train)
        self.label_dicts = {x["assay_id"]: x for x in self.train_label_lst}
        self.linear1 = nn.Linear(2 * self.embed_dim, self.embed_dim)  #

    def forward(self, nodes_pocket, nodes_lig=None, max_sample=10):
        to_neighs = []
        if nodes_lig is None:
            lig_smi_lst = ["----"] * len(nodes_pocket)
        else:
            lig_smi_lst = [self.mol_smi[lig_id] for lig_id in nodes_lig]

        for node, smi in zip(nodes_pocket, lig_smi_lst):
            assayid = self.idx2assayid[node]
            neighbors = []
            nbr_pockets = self.pocket_graph.get(assayid, [])
            # random.shuffle(nbr_pockets)
            # breakpoint()
            for n_assayid, score in nbr_pockets:
                nbr_smi = self.label_dicts[n_assayid]["ligands"][0]["smi"]
                if assayid == n_assayid:
                    continue
                if smi == nbr_smi:
                    continue
                if n_assayid not in self.assayid_set_train:
                    continue
                neighbors.append((self.smi2idx[nbr_smi], int((score - 0.5) * 10)))
            to_neighs.append(neighbors)

        neigh_feats = self.aggregator.forward(nodes_pocket, to_neighs)  # user-item network
        return neigh_feats
    
    def refine_pocket(self, pocket_embed, neighbor_list=None):
        return self.aggregator.forward_inference(pocket_embed, neighbor_list)
