from sklearn import metrics
import numpy as np
from math import *
from sklearn.svm import LinearSVC
import pandas as pd

def getAccuracy(y_true,y_pred,split_index):
    result=[]
    y_pred=y_pred.detach().numpy()
    for i in range(0,len(split_index)):#将节点的特征分类
        if i==0:
            begin=0
        else:
            begin=split_index[i-1]
        end=split_index[i]
        denoiseFeature=Denoise(y_pred[:,begin:end])
        result.append(metrics.f1_score(y_true[:,begin:end],denoiseFeature,average='macro'))

    return result

def Denoise(feature):
    '''
    去除最大值，就是预测值
    '''
    result=np.zeros((len(feature),len(feature[0])))
    for i in range(0,len(feature)):
        temp_max=feature[i][0]
        index=0
        for j in range(1,len(feature[0])):
            if feature[i][j]>temp_max:
                temp_max=feature[i][j]
                index=j
        result[i][index]=1
    return result

def SVMTest(train_x,train_y,test_x,test_y):
    svm = LinearSVC(dual=False)
    svm.fit(train_x, train_y)
    y_pred = svm.predict(test_x)
    macro_f1 = metrics.f1_score(test_y, y_pred, average='macro')
    micro_f1 = metrics.f1_score(test_y, y_pred, average='micro')

    return macro_f1,micro_f1

def dot_product(v1, v2):
    """Get the dot product of the two vectors.
    if A = [a1, a2, a3] && B = [b1, b2, b3]; then
    dot_product(A, B) == (a1 * b1) + (a2 * b2) + (a3 * b3)
    true
    Input vectors must be the same length.
    """
    return sum(a * b for a, b in zip(v1, v2))



def Heat_Kernel(x,y,hyper):
    core=sqrt(dot_product(x-y,x-y))
    result=exp(-core/hyper)

    return  result

def get_batch_Heat_Kernel(predict,ground):
    total_len=len(predict)
    total_loss=0
    for i in range(0,total_len):
        total_loss+=Heat_Kernel(predict[i],ground[i],2)
    avg=total_loss/total_len

    return avg

def get_correlation(predict,ground):
    result = 0

    for i in range(0, len(ground)):
        ground_temp = pd.Series(ground[i])
        predict_temp = pd.Series(predict[i])
        corr = predict_temp.corr(ground_temp, method='pearson')
        result += corr

    return result/len(ground)