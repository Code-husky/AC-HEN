import numpy as np
import torch
class batch_generator():
    def __init__(self,context_pair,batch_size,missingidx,neig_len,allFeature,cos_matrix,missingFeature,shuffle=True):
        self.context_pair=context_pair#所有random walk的节点
        self.batch_size=batch_size# 每个batch 选几个节点
        self.last_batch_size=batch_size
        self.missingidx=missingidx # 被去除特征的用户节点
        self.neig_len=neig_len # batch 中的一个节点，选取多少random walk 中的节点
        self.allFeature=allFeature
        self.iter_counter = 0
        self.shuffle = shuffle
        self.cos_matrix=cos_matrix
        self.missingFeature=missingFeature
        self.src_missingidx=missingidx.copy()
        if shuffle:
            np.random.shuffle(self.missingidx)

    def next(self):
        if self.num_iterations_left() <= 0:# 将所有训练集shuffle 并且遍历一边
            self.reset()
        self.iter_counter += 1
        batch_node=self.missingidx[(self.iter_counter - 1) * self.batch_size:
                                    min(self.iter_counter * self.batch_size,len(self.missingidx))]
        index=self.get_index(batch_node)
        return self.get_batch_rdnode(batch_node),index,self.cos_matrix[index]# 返回对应batch的节点

    def num_iterations(self):

        return int(np.ceil(len(self.missingidx) / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
             np.random.shuffle(self.missingidx)
        self.iter_counter = 0

    def get_index(self,batch_node):
        '''
        get accuracy index
        '''
        index=[]
        all_missing=list(self.src_missingidx)#原来的missingidx 已经被打乱顺序，所以需要未被打乱的

        for i in batch_node:
            index.append(all_missing.index(i))
        if(len(index)<self.batch_size):
            self.last_batch_size=len(index)
        return  index

    def get_context_node(self,context_node, source_node):
        '''
        get high order nodes
        '''
        count = 0
        node_index = 0
        temp = [source_node]
        while count < self.neig_len:
            flag = True
            for i in self.missingidx:
                try:
                    if i == context_node[node_index]:
                        flag = False
                except:
                    node_index = node_index - 1
                    temp.append(context_node[node_index])
                    count=count+1
            if flag:
                temp.append(context_node[node_index])
                count = count + 1
            node_index = node_index + 1
        return temp


    def get_batch_rdnode(self,batch_node_idx):
        '''
        drop nodes
        '''

        result = []
        for i in batch_node_idx:
            context_node = self.context_pair.get(i).copy()
            temp = self.get_context_node(context_node, i)
            result.append(temp)
        result = np.array(result)

        return result

    def get_batch_feature(self, feature_dim, batch_rdnode):
        '''
        get feature of nodes
        '''

        batch_feature = np.zeros((min(self.batch_size,self.last_batch_size), self.neig_len, feature_dim))

        for i in range(0, len(batch_feature)):
            for j in range(0, len(batch_feature[0])):
                batch_feature[i][j] = self.allFeature[batch_rdnode[i][j]]

        return torch.tensor(batch_feature,dtype=torch.float)

    def get_simi_batch_feature(self,feature_dim,batch_simi_node,top_k):
        '''
        get feature of nodes
        '''
        batch_feature = np.zeros((min(self.batch_size, self.last_batch_size), top_k, feature_dim))

        for i in range(0, len(batch_feature)):
            for j in range(0, len(batch_feature[0])):
                batch_feature[i][j] = self.missingFeature[batch_simi_node[i][j]]

        return torch.tensor(batch_feature, dtype=torch.float)





