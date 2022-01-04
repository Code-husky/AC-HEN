import torch
import torch.nn as nn
from Model.GCN import *
from Model.Attention import *
import torch.nn.functional as F

class FeatureComplete(nn.Module):
    def __init__(self,node_dim,hid_dim,out_dim,drop_col,top_k,exist_col,num_heads,batch_node_num,context_node_num,source):
        super(FeatureComplete,self).__init__()
        self.node_dim=node_dim
        self.hid_dim=hid_dim
        self.out_dim=out_dim
        self.drop_col=drop_col
        self.exist_col=exist_col
        self.top_k=top_k
        self.context_node_num=context_node_num
        self.batch_node_num=batch_node_num
        self.source=source
        self.gcn=GCNNet(self.node_dim,self.hid_dim)
        self.gcnE=GCNNet(self.node_dim,self.hid_dim)
        self.Muti_H_Atten=Muti_H_Atten(self.hid_dim,self.hid_dim,self.node_dim,
                                       activation=F.elu,num_heads=num_heads,exist_col=self.exist_col)

        self.linear=nn.Linear(self.hid_dim*3,self.out_dim)

        self.translate=nn.Linear(self.node_dim,self.hid_dim)
        self.paraForCos=nn.Parameter(nn.init.xavier_normal_(
            torch.Tensor(self.top_k,1).type(torch.FloatTensor),
            gain=np.sqrt(2.0)), requires_grad=True)

    def forward(self,adj,target_emb,node_emb_gcn,node_emb_rd,batch_node_idx,batch_simi_node_feature,graph,trainfeature,node_rd,feature):
        #weight adjust

        batch_simi_node_feature = self.translate(batch_simi_node_feature)
        weight_feature=torch.zeros(len(batch_simi_node_feature),self.top_k,self.hid_dim)

        for i in range(0,len(batch_simi_node_feature)):
            weight_feature[i]=batch_simi_node_feature[i]*self.paraForCos

        simi_feature=torch.mean(weight_feature,axis=1)
        #neighbor aggregation
        emb_gcn=self.gcn(adj,node_emb_gcn)
        #high order aggregation
        emb_gcnE=self.gcnE(graph,trainfeature)
        for i in range(0,self.batch_node_num):
            for j in range(0,self.context_node_num+1):
                if j==0:
                    node_emb_rd[i][j]=emb_gcnE[node_rd[i][j]+self.source]
                    feature[i][j] = trainfeature[node_rd[i][j] + self.source]
                else:
                    node_emb_rd[i][j]=emb_gcnE[node_rd[i][j]]
                    feature[i][j] = trainfeature[node_rd[i][j] ]
        emb_gcn=emb_gcn[batch_node_idx]

        emb_attention=self.Muti_H_Atten(node_emb_rd,feature)
        #embedding fusion
        emb=torch.cat([emb_gcn,emb_attention],dim=1)
        final_emb=torch.cat([emb,simi_feature],dim=1)
        result=self.linear(final_emb)
        loss_ac = F.mse_loss(target_emb[:,self.drop_col], result[:,self.drop_col])

        return result,loss_ac

