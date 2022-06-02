import torch
import torch.nn as nn

class GCN (nn.Module):
    def __init__(self,in_dim,out_dim):
        super(GCN, self).__init__()

        self.W = nn.Parameter(torch.rand(in_dim, out_dim, requires_grad=True))

    def forward(self, adj,X):
        rowsum = torch.sum(adj, 1)
        for i in range(0, len(adj)):
            adj[i] = adj[i] / rowsum[i]
        out = torch.mm(torch.mm(adj, X), self.W)
        return out


class GCNNet(torch.nn.Module):
    def __init__(self, nfeat, nout):
        super(GCNNet, self).__init__()
        self.conv1 = GCN( nfeat, nout)

    def forward(self, A,X):
        result = self.conv1(A,X)
        return result

