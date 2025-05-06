import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import random
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embedding_dims):
        super(Attention, self).__init__()
        self.embed_dim = embedding_dims
        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, 1)
        self.att1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.att2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.att3 = nn.Linear(self.embed_dim, 1)

        # self.linear_q = nn.Linear(self.embed_dim, self.embed_dim)
        # self.linear_k = nn.Linear(self.embed_dim, self.embed_dim)
        self.softmax = nn.Softmax(0)

    def forward(self, node1, u_rep, num_neighs):
        uv_reps = u_rep.repeat(num_neighs, 1)
        x = torch.cat((node1, uv_reps), 1)
        x = F.relu(self.att1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.att2(x))
        x = F.dropout(x, training=self.training)
        x = self.att3(x)
        att = F.softmax(x, dim=0)

        # u_rep = self.linear_q(u_rep).repeat(num_neighs, 1)
        # node1 = self.linear_k(node1)
        # att = torch.sum(u_rep * node1, dim=1)
        # att = F.softmax(att, dim=0).unsqueeze(1)
        return att
