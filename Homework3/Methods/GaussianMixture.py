# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 16:45:15 2018
Gaussian Mixture聚类算法
@author: Qiannan Cheng
"""
import DataProcessing
import numpy as np
from sklearn import metrics
from sklearn.mixture import GaussianMixture

#调用数据处理函数，得到sklearn函数可以直接调用的数据
X,labels=DataProcessing.getAvailableData("../Tweets.txt")
X=X.toarray() #type:<class 'numpy.ndarray'>
true_k=np.unique(labels).shape[0] #cluster的数目

#定义函数执行GaussianMixture聚类算法
# @param X 样本特征数据
# @param k 簇的数目
# @return y_pred 
def GaussianMixture_Clustering(X,k):
    #参数n_components: int, defaults to 1
    #The number of mixture components.
    gm=GaussianMixture(n_components=k)
    gm.fit(X)
    y_pred=gm.predict(X)
    return y_pred

#调用GaussianMixture_Clustering函数，得到聚类标签
pred_labels=GaussianMixture_Clustering(X,true_k)

#使用NMI(Normalized Mutual Information)作为评价指标进行评估
NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
print("Normalized Mutual Information: %0.3f" % NMI)
    
    
    