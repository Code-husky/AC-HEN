import torch
import math
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class Muti_H_Atten(nn.Module):
    '''
    multi-head  attention
    '''
    def __init__(self, in_dim, hidden_dim,feature_dim,activation, num_heads,exist_col, cuda=False):
        super(Muti_H_Atten, self).__init__()

        self.attentions = [Attention(in_dim, hidden_dim, hidden_dim ,hidden_dim,feature_dim ,exist_col) for _ in range(num_heads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, emb,feature):

        x = torch.cat([att(emb,feature).unsqueeze(0) for att in self.attentions], dim=0)

        return torch.mean(x, dim=0, keepdim=False)



class PositionalEncoding(nn.Module):
    "positional encoding"

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Attention(nn.Module):
    def __init__(self,d_input, d_q, d_k, d_v,feature_dim,exist_col):
        super(Attention,self).__init__()
        self.d_k=d_k
        self.pad_index=0
        self.feature_dim=feature_dim

        self.TransToQ = nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(d_input,self.d_k).type(torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)
        self.TransToK=nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(d_input, self.d_k).type(torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)
        self.TransToV=nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(self.feature_dim, d_v).type(torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)
        self.pos_emb = PositionalEncoding(d_model=d_input, dropout=0)
        self.exist_col=exist_col

    def forward(self,emb,feature):
        '''
        get attention-based high order embedding
        '''

        enc_output=self.pos_emb(emb)


        Q_S=torch.matmul(enc_output,self.TransToQ)#求取Q，K，V
        K_S=torch.matmul(enc_output,self.TransToK)
        V_S=torch.matmul(feature,self.TransToV)
        scores=torch.matmul(Q_S,K_S.transpose(-1,-2))/np.sqrt(self.d_k)
        atten=nn.Softmax(dim=-1)(scores)
        atten = F.dropout(atten, 0.5, training=self.training)

        context=torch.matmul(atten,V_S)

        result=context[0][0]

        for i in range(1,len(context)):
            result=torch.cat((result,context[i][0]))
        result=result.view(-1,self.d_k)

        return result