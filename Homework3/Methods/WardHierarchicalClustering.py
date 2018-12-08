# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 22:38:57 2018
Ward Hierarchical Clustering
@author: Qiannan Cheng
"""
import DataProcessing
import numpy as np
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering

#调用数据处理函数，得到sklearn函数可以直接调用的数据
X,labels=DataProcessing.getAvailableData("../Tweets.txt")
X=X.toarray() #type:<class 'numpy.ndarray'>
true_k=np.unique(labels).shape[0] #cluster的数目

#定义函数执行WardHierarchicalClustering聚类算法
# @param X 样本特征数据
# @param k 簇的数目
# @return y_pred
def WardHierarchicalClusteringAlgorithm(X,k):
    #参数linkage: optional (default=”ward”)
    #ward minimizes the variance of the clusters being merged.
    #参数n_clusters: int, default=2
    #The number of clusters to find.
    whc=AgglomerativeClustering(linkage='ward',
                                n_clusters=k)
    whc.fit(X)
    y_pred=whc.labels_
    return y_pred

#调用WardHierarchical_Clustering函数，得到聚类标签
pred_labels=WardHierarchicalClusteringAlgorithm(X,true_k)

#使用NMI(Normalized Mutual Information)作为评价指标进行评估
NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
print("Normalized Mutual Information: %0.3f" % NMI)


