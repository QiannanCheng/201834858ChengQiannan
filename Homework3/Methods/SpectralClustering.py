# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:31:55 2018
Spectral Clustering
@author: Qiannan Cheng
"""
import DataProcessing
import numpy as np
from sklearn import metrics
from sklearn.cluster import SpectralClustering

#调用数据处理函数，得到sklearn函数可以直接调用的数据
X,labels=DataProcessing.getAvailableData("../Tweets.txt")
X=X.toarray() #type:<class 'numpy.ndarray'>
true_k=np.unique(labels).shape[0] #cluster的数目

#定义函数执行SpectralClustering聚类算法
# @param X 样本特征数据
# @param k 簇的数目
# @return y_pred 
def Spectral_Clustering(X,k):
    #参数n_clusters: integer, optional
    #The dimension of the projection subspace.
    sc=SpectralClustering(n_clusters=k)
    sc.fit(X)
    y_pred=sc.labels_
    return y_pred

#调用Spectral_Clustering函数，得到聚类标签
pred_labels=Spectral_Clustering(X,true_k)

#使用NMI(Normalized Mutual Information)作为评价指标进行评估
NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
print("Normalized Mutual Information: %0.3f" % NMI)

