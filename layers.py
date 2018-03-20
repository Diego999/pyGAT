import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True, residual=False):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.residual = residual

        self.seq_transformation = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1, bias=False)
        if self.residual:
            self.proj_residual = nn.Conv1d(in_features, out_features, kernel_size=1, stride=1)
        self.f_1 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.f_2 = nn.Conv1d(out_features, 1, kernel_size=1, stride=1)
        self.bias = nn.Parameter(torch.zeros(out_features).type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        seq = torch.transpose(input, 0, 1).unsqueeze(0)
        seq_fts = self.seq_transformation(seq)

        f_1 = self.f_1(seq_fts)
        f_2 = self.f_2(seq_fts)
        logits = (torch.transpose(f_1, 2, 1) + f_2).squeeze(0)
        coefs = F.softmax(self.leakyrelu(logits) + adj, dim=1)

        seq_fts = F.dropout(torch.transpose(seq_fts.squeeze(0), 0, 1), self.dropout, training=self.training)
        coefs = F.dropout(coefs, self.dropout, training=self.training)

        ret = torch.mm(coefs, seq_fts) + self.bias

        if self.residual:
            if seq.size()[-1] != ret.size()[-1]:
                ret += self.proj_residual(seq).squeeze(2)
            else:
                ret += seq

        if self.concat:
            return F.elu(ret)
        else:
            return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'