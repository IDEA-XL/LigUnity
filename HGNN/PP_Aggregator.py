import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random
from Attention import Attention


class PPAggregator(nn.Module):
    """
    Social Aggregator: for aggregating embeddings of social neighbors.
    """

    def __init__(self, u2e, embed_dim, cuda="cpu"):
        super(PPAggregator, self).__init__()
        self.device = cuda
        self.u2e = u2e
        self.embed_dim = embed_dim
        self.att = Attention(self.embed_dim)

    def forward(self, nodes, to_neighs):
        embed_matrix = torch.zeros(len(nodes), self.embed_dim, dtype=torch.float).to(self.device)
        self_feats = self.u2e.weight[nodes]
        for i in range(len(nodes)):
            tmp_adj = to_neighs[i]

            num_neighs = len(tmp_adj)

            if num_neighs > 0:
                e_u = self.u2e.weight[[x[0] for x in tmp_adj]] # fast: user embedding
                u_rep = self.u2e.weight[nodes[i]]
                att_w = self.att(e_u, u_rep, num_neighs)
                att_history = torch.mm(e_u.t(), att_w).t()
                embed_matrix[i] = (att_history + self_feats[i]) / 2
            else:
                embed_matrix[i] = self_feats[i]
        return embed_matrix
