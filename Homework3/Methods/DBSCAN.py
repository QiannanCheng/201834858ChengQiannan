# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:59:38 2018
DBSCAN聚类算法
@author: Qiannan Cheng
"""
import DataProcessing
from sklearn import metrics
from sklearn.cluster import DBSCAN

#调用数据处理函数，得到sklearn函数可以直接调用的数据
X,labels=DataProcessing.getAvailableData("../Tweets.txt")
X=X.toarray() #type:<class 'numpy.ndarray'>

#定义函数执行DBSCAN聚类算法
# @param X 样本特征数据
# @return y_pred
def DBSCANAlgorithm(X):
    #参数eps: float, optional
    #The maximum distance between two samples for them to be considered as in the same neighborhood.
    #参数min_samples: int, optional
    #The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
    #This includes the point itself.
    ds=DBSCAN(eps=0.5,
              min_samples=5)
    ds.fit(X)
    y_pred=ds.labels_
    return y_pred

#调用DBSCAN_Clustering函数，得到聚类标签
pred_labels=DBSCANAlgorithm(X)

#使用NMI(Normalized Mutual Information)作为评价指标进行评估
NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
print("Normalized Mutual Information: %0.3f" % NMI)


