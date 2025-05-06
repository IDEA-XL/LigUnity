import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random
from Attention import Attention

class PLAggregator(nn.Module):
    """
    item and user aggregator: for aggregating embeddings of neighbors (item/user aggreagator).
    """

    def __init__(self, v2e, r2e, u2e, embed_dim, cuda="cpu", uv=True):
        super(PLAggregator, self).__init__()
        self.uv = uv
        self.v2e = v2e
        self.r2e = r2e
        self.u2e = u2e
        self.device = cuda
        self.embed_dim = embed_dim
        self.w_r1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_r2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att = Attention(self.embed_dim)
        self.v2e.requires_grad = False
        self.u2e.requires_grad = False


    def forward(self, nodes_u, input_hist):
        embed_matrix = torch.zeros(len(input_hist), self.embed_dim, dtype=torch.float).to(self.device)

        for i in range(len(input_hist)):
            history = []
            label = []
            for idx in range(len(input_hist[i])):
                vid_hist = input_hist[i][idx][0]
                vlabel_hist = input_hist[i][idx][1]
                history.append(vid_hist)
                label.append(vlabel_hist)

            num_histroy_item = len(history)

            if num_histroy_item > 0:
                e_uv = self.v2e.weight[history]
                uv_rep = self.u2e.weight[nodes_u[i]]

                e_r = self.r2e.weight[label]
                x = torch.cat((e_uv, e_r), 1)
                x = F.relu(self.w_r1(x))
                o_history = F.relu(self.w_r2(x))

                att_w = self.att(o_history, uv_rep, num_histroy_item)
                # print([(a,b) for a,b in zip(label, att_w)])
                att_history = torch.mm(o_history.t(), att_w)
                att_history = att_history.t()

                embed_matrix[i] = (att_history + uv_rep) / 2
            else:
                embed_matrix[i] = self.u2e.weight[nodes_u[i]]

        return embed_matrix
