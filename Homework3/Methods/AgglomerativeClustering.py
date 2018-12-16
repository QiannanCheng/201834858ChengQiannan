# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:25:24 2018
Agglomerative Clustering
@author: Qiannan Cheng
"""
import DataProcessing
import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

#定义函数执行AgglomerativeClustering聚类算法
# @param X 样本特征数据
# @param k 簇的数目
# @return y_pred
def AgglomerativeClusteringAlgorithm(X,k):
    #参数linkage:  {“ward”, “complete”, “average”, “single”}, optional 
    #average uses the average of the distances of each observation of the two sets.
    #参数n_clusters: int, default=2
    #The number of clusters to find.
    ac=AgglomerativeClustering(linkage='average',
                               n_clusters=k)
    ac.fit(X)
    y_pred=ac.labels_
    return y_pred

if __name__=='__main__':
    #调用数据处理函数，得到sklearn函数可以直接调用的数据
    X,labels=DataProcessing.getAvailableData("../Tweets.txt")
    X=X.toarray() #type:<class 'numpy.ndarray'>
    true_k=np.unique(labels).shape[0] #cluster的数目
    
    #调用AgglomerativeClusteringAlgorithm函数，得到聚类标签
    pred_labels=AgglomerativeClusteringAlgorithm(X,true_k)
    
    #使用NMI(Normalized Mutual Information)作为评价指标进行评估
    NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
    print("Normalized Mutual Information: %0.3f" % NMI)

