import torch
from utils.utils import *
import torch.nn.functional as F
from Model.model import *
from utils.tool import *
from utils.evaluate import *
from utils.earlyStop import *
import warnings
warnings.filterwarnings("ignore")
feature_dim=128
hid_dim=256
out_put_dim=128
batch_node_num=16
context_node_num=4
lr = 0.001
weight_decay = 0.001
epoch=40
np.random.seed(1234)
np.set_printoptions(precision = 3)
top_k=4
num_heads=8
source=0


#load data
graph,Adj,compleltefeature,TrainFeature,missingidx,context_pair,source_feature,missingFeature=loaddata()
val_Adj,val_Feature,val_missingidx,val_completefeature,val_source_feature,valid_missingFeature=loadvaldata()
test_Adj,test_Feature,test_missingidx,test_completefeature,test_source_feature,final_result=loadtestdata()
test_missingFeature=final_result.copy()

neighbor=np.load('Neighborhood.npy',allow_pickle=True)
new_context_pair=get_new_context_pair(context_pair,neighbor,context_node_num,len(missingFeature))

#get similiar nodes
cos_matrix,drop_col=get_CosSimilarity(missingFeature,missingidx,top_k,'Train')
valid_cos_matrix,valid_drop_col=get_CosSimilarity(valid_missingFeature,val_missingidx,top_k,'Valid')
test_cos_matrix,test_drop_col=get_CosSimilarity(final_result,test_missingidx,top_k,'Test')


all_col=np.arange(0,feature_dim)
exist_col=[]
exist_col=np.delete(all_col,drop_col)
graph=torch.tensor(graph,dtype=torch.float)
TrainFeature=torch.tensor(TrainFeature,dtype=torch.float)
all_feature=np.load("data/target_feature_low.npy")
#get  batch generator
batch_iterator=batch_generator(new_context_pair,batch_node_num,missingidx,context_node_num,TrainFeature,cos_matrix,missingFeature)
missingFeature=torch.tensor(missingFeature,dtype=torch.float)
valid_missingFeature=torch.tensor(valid_missingFeature,dtype=torch.float)
test_missingFeature=torch.tensor(test_missingFeature,dtype=torch.float)
train_Y,train_svm_idx,test_label=loadSVM()
train_X=all_feature[train_svm_idx]
Adj=torch.tensor(Adj,dtype=torch.float)
val_Adj=torch.tensor(val_Adj,dtype=torch.float)
test_Adj=torch.tensor(test_Adj,dtype=torch.float)
#define model
early_stop=EarlyStopping(patience=10)
net=FeatureComplete(feature_dim,hid_dim,out_put_dim,drop_col,top_k,exist_col,num_heads,batch_node_num,
                    context_node_num,source)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
early_stopping = EarlyStopping(patience=10, verbose=True)

val_Feature_tensor=torch.tensor(val_Feature,dtype=torch.float)
test_Feature_tensor=torch.tensor(test_Feature,dtype=torch.float)

# training model
for i in range(0,epoch):
    total_loss=0

    for j in range(0, batch_iterator.num_iterations()):
        batch_node, index ,batch_simi_node= batch_iterator.next()
        # batch_node_feature = batch_iterator.get_batch_feature(feature_dim, batch_node)
        batch_simi_node_feature=batch_iterator.get_simi_batch_feature(feature_dim,batch_simi_node,top_k)
        node_emb_rd = torch.zeros((batch_node_num, context_node_num + 1, hid_dim), dtype=torch.float)
        feature = torch.zeros((batch_node_num, context_node_num + 1, feature_dim),dtype=torch.float)
        compleltefeature = torch.tensor(compleltefeature, dtype=torch.float)
        target = source_feature[index]
        target = torch.tensor(target, dtype=torch.float)
        result ,loss_ac= net(Adj,target,compleltefeature, node_emb_rd, index,batch_simi_node_feature,
                             graph,TrainFeature,batch_node,feature)

        optimizer.zero_grad()
        loss_ac.backward()
        optimizer.step()
        total_loss += loss_ac.item()

    total_loss/=batch_iterator.num_iterations()
    total_loss=np.round(total_loss,4)

    print("training epoch:{}, loss: {}\n".format(i,total_loss))
    #val model
    with torch.no_grad():
        val_loss = 0
        val_batch_iterator = batch_generator(new_context_pair, batch_node_num,
                                             val_missingidx, context_node_num
                                             , val_Feature,valid_cos_matrix,valid_missingFeature)

        for k in range(0, val_batch_iterator.num_iterations()):
            val_batch_node, val_index,val_batch_simi_node= val_batch_iterator.next()
            val_batch_node_feature = val_batch_iterator.get_batch_feature(feature_dim, val_batch_node)
            val_batch_simi_node_feature = val_batch_iterator.get_simi_batch_feature(feature_dim, val_batch_simi_node, top_k)
            val_completefeature = torch.tensor(val_completefeature, dtype=torch.float)
            val_target = val_source_feature[val_index]
            val_target = torch.tensor(val_target, dtype=torch.float)
            val_node_emb_rd = torch.zeros((batch_node_num, context_node_num + 1, hid_dim), dtype=torch.float)
            val_feature_batch = torch.zeros((batch_node_num, context_node_num + 1, feature_dim), dtype=torch.float)
            val_result ,val_loss_ac= net(val_Adj,val_target ,val_completefeature,val_node_emb_rd,
                                      val_index,val_batch_simi_node_feature,graph,val_Feature_tensor,val_batch_node,val_feature_batch)


            val_loss += val_loss_ac.item()

        val_loss /= val_batch_iterator.num_iterations()

        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
        print("val loss：{}\n".format(val_loss))

#test model
net.load_state_dict(torch.load('checkpoint.pt'))
net.eval()
test_X = []
test_Y = []
with torch.no_grad():
    test_loss = 0
    test_batch_iterator = batch_generator(new_context_pair, batch_node_num,
                                          test_missingidx, context_node_num
                                          , test_Feature,test_cos_matrix,test_missingFeature,shuffle=False)

    for m in range(0, test_batch_iterator.num_iterations()):
        test_batch_node, test_index,test_batch_simi_node = test_batch_iterator.next()
        test_batch_node_feature = test_batch_iterator.get_batch_feature(feature_dim, test_batch_node)
        test_batch_simi_node_feature = test_batch_iterator.get_simi_batch_feature(feature_dim, test_batch_simi_node, top_k)
        test_completefeature = torch.tensor(test_completefeature, dtype=torch.float)
        test_target = test_source_feature[test_index]
        test_target = torch.tensor(test_target, dtype=torch.float)
        test_node_emb_rd = torch.zeros((batch_node_num, context_node_num + 1, hid_dim), dtype=torch.float)
        test_feature_batch = torch.zeros((batch_node_num, context_node_num + 1, feature_dim), dtype=torch.float)

        test_result ,test_loss_ac = net(test_Adj, test_target,test_completefeature,
                          test_node_emb_rd, test_index,test_batch_simi_node_feature,graph,test_Feature_tensor,test_batch_node,
                                        test_feature_batch)


        test_loss += test_loss_ac.item()
        test_result = test_result.detach().numpy()


        if m == 0:
            test_X = test_result
            test_Y = test_label[test_index]
        else:
            test_X = np.concatenate((test_X, test_result), axis=0)
            test_Y = np.concatenate((test_Y, test_label[test_index]), axis=0)

    for row_num in range(0,len(test_missingidx)):
        for col in drop_col:
           final_result[test_missingidx[row_num]][col]=test_X[row_num][col]


    test_loss /= test_batch_iterator.num_iterations()
    test_loss = np.round(test_loss, 4)

    distance=get_batch_Heat_Kernel(test_X[:,drop_col],all_feature[test_missingidx][:,drop_col])
    correlation=get_correlation(test_X[:,drop_col],all_feature[test_missingidx][:,drop_col])
    print("test loss：{},distance:{},correlation: {}\n".format(test_loss,distance,correlation))


np.save("CompleteFeatureMovie.npy", final_result)

