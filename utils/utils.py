import numpy as np
import dill
from utils.node2vec1 import *
import networkx as nx
from collections import defaultdict
from math import sqrt
import itertools

def loaddata():
    graph = nx.read_gpickle("data/IMDB_movie_actor_matrix.pkl")
    TrainMatrix = np.load('data/TrainDropMovieActorMatrix.npy')
    missingFeature = np.load("data/TrainMissingMovieEmb.npy")
    missingidx = np.load("data/TrainDropMovieidx.npy")
    completeFeature = np.load("data/actor_feature_low.npy")
    completeidx = np.load("data/TrainDropActoridx.npy")
    sourceFeature = np.load("data/movie_feature_low.npy")


    try:
        context_pair=dill.load(open("train_pair.pkl",'rb'))
    except:
        context_pair=run_random_walks_n2v(graph,graph.nodes())
        dill.dump(context_pair, open("train_pair.pkl", 'wb'))

    return nx.to_numpy_array(graph),TrainMatrix,completeFeature[completeidx],np.vstack((missingFeature,completeFeature)),\
           missingidx,context_pair,sourceFeature[missingidx],missingFeature

def loadvaldata():
    TrainMatrix = np.load('data/valid/ValidDropMovieActorMatrix.npy')
    missingFeature = np.load("data/valid/ValidMissingMovieEmb.npy")
    missingidx = np.load("data/valid/ValidDropMovieidx.npy")
    completeidx = np.load("data/valid/ValidDropActoridx.npy")
    completeFeature = np.load("data/actor_feature_low.npy")
    sourceFeature = np.load("data/movie_feature_low.npy")


    return TrainMatrix,np.vstack((missingFeature,completeFeature)),\
           missingidx,completeFeature[completeidx],sourceFeature[missingidx],missingFeature



def loadtestdata():
    TrainMatrix = np.load('data/test/TestDropMovieActorMatrix.npy')
    missingFeature = np.load("data/test/TestMissingMovieEmb.npy")
    missingidx = np.load("data/test/TestDropMovieidx.npy")
    completeidx = np.load("data/test/TestDropActoridx.npy")
    completeFeature = np.load("data/actor_feature_low.npy")
    sourceFeature = np.load("data/movie_feature_low.npy")



    return TrainMatrix,np.vstack((missingFeature,completeFeature)),\
           missingidx,completeFeature[completeidx],sourceFeature[missingidx],missingFeature

def loadSVM():
    train_svm_label=np.load("data/svm/train_svm_label.npy",allow_pickle=True)
    train_svm_idx=np.load("data/svm/train_svm_idx.npy",allow_pickle=True)
    test_label=np.load("data/svm/test_label.npy",allow_pickle=True)

    return train_svm_label,train_svm_idx,test_label




def run_random_walks_n2v(graph, nodes, num_walks=4, walk_len=20):
    """ In: Graph and list of nodes
        Out: (target, context) pairs from random walk sampling using the sampling strategy of node2vec (deepwalk)"""
    walk_len = 20

    nx_G = nx.Graph()

    adj = nx.adjacency_matrix(graph,range(len(graph.nodes())))#邻接矩阵行标与图节点名称对应
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])

    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)# 初始化randomwalk 对象
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 8
    pairs = defaultdict(lambda: [])
    pairs_cnt = 0


    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs

def get_batch_rdnode(context_pair,batch_size,start_index,usedropidx):
    '''
    drop nodes
    '''
    batch_node_idx = usedropidx[start_index:start_index+batch_size]
    result = []
    for i in batch_node_idx:

        context_node = context_pair.get(i).copy()
        temp = get_context_node(context_node, usedropidx, 10, i)
        result.append(temp)
    result = np.array(result)

    return result

def get_context_node(context_node,missingidx,len,source_node):
    '''
    get high order nodes
    '''
    count=0
    node_index=0
    temp=[source_node]
    while count<len:
        flag=True
        for i in missingidx:
            try:
                if i==context_node[node_index]:
                    flag=False
            except:
                node_index=node_index-1
                temp.append(context_node[node_index])

        if flag:
            temp.append(context_node[node_index])
            count=count+1
        node_index=node_index+1
    return temp


def get_batch_feature( batch_node_num, context_node_num, feature_dim,
                       allFeature, batch_rdnode):
    '''
    get feature
    '''
    batch_feature = np.zeros((batch_node_num, context_node_num, feature_dim))
    for i in range(0, len(batch_feature)):
        for j in range(0, len(batch_feature[0])):
            batch_feature[i][j] = allFeature[batch_rdnode[i][j]]

    return batch_feature


def get_CosSimilarity(missingFeature,miss_idx,k,name):
    '''
    select top-k similar nodes
    '''
    try:
        final_matrix=np.load(name+'_Cos_Similarty_Matrix.npy')
        drop_col=np.load('drop_col.npy')

    except:
        all_idx=np.arange(len(missingFeature))
        exist_idx=np.delete(all_idx,miss_idx)
        drop_len=len(miss_idx)
        exist_len=len(exist_idx)
        cos_Simi_matrix=np.zeros((drop_len,exist_len))
        exist_col,drop_col=get_dropcol(missingFeature[miss_idx[0]])
        feature=missingFeature[:,exist_col]


        for i in range(0,drop_len):
            for j in range(0,exist_len):
                cos_Simi_matrix[i][j]=similarity(feature[miss_idx[i]],feature[exist_idx[j]])
            if i%10==0:
                print('process: {}'.format(i/drop_len))




        final_matrix=np.zeros((drop_len,k),dtype=np.int)
        for i in range(0, drop_len):
            for j in range(0,k):
                max_idx=np.argmax(cos_Simi_matrix[i])
                final_matrix[i][j]=max_idx
                cos_Simi_matrix[i][max_idx]=-1
        np.save(name+'_Cos_Similarty_Matrix.npy',final_matrix)
        np.save('drop_col',drop_col)

    return final_matrix,drop_col


def get_dropcol(missingFeature):
    '''
    select exist feature drop feature
    '''
    exist_col=[]
    drop_col=[]
    for i in range(0,len(missingFeature)):
        if missingFeature[i]!=1:
            exist_col.append(i)
        else:
            drop_col.append(i)
    return exist_col,drop_col

def dot_product(v1, v2):
    """Get the dot product of the two vectors.
    if A = [a1, a2, a3] && B = [b1, b2, b3]; then
    dot_product(A, B) == (a1 * b1) + (a2 * b2) + (a3 * b3)
    true
    Input vectors must be the same length.
    """
    return sum(a * b for a, b in zip(v1, v2))


def magnitude(vector):
    """Returns the numerical length / magnitude of the vector."""
    return sqrt(dot_product(vector, vector))


def similarity(v1, v2):
    """Ratio of the dot product & the product of the magnitudes of vectors."""
    return dot_product(v1, v2) / (magnitude(v1) * magnitude(v2) + .00000000001)

def get_new_context_pair(context_pair,neighbor,context_node_num,total_len):
    new_context_pair = defaultdict(lambda: [])

    for i in range(0, total_len):
        pair = context_pair.get(i)
        temp = []
        for j in pair:  # 遍历邻接
            flag = False
            for k in range(0, len(neighbor[i])):  # 去除邻居
                if j == neighbor[i][k]:
                    flag = True
                    break
            if flag is False:
                temp.append(j)
        while len(temp) < context_node_num:
            temp.append(context_pair[i][len(temp) - 1])
        new_context_pair[i] = temp

    return new_context_pair