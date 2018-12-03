# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:18:20 2018
Affinity Propagation聚类算法
@author: Qiannan Cheng
"""
import DataProcessing
from sklearn import metrics
from sklearn.cluster import AffinityPropagation

#调用数据处理函数，得到sklearn函数可以直接调用的数据
X,labels=DataProcessing.getAvailableData("../Tweets.txt")

#定义函数实现AffinityPropagation聚类算法
# @param X 样本特征数据
# @return y_pred 
def AffinityPropagation_Clustering(X):
    #参数damping: float, optional, default: 0.5
    #Damping factor (between 0.5 and 1) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping). 
    #This in order to avoid numerical oscillations when updating these values (messages).
    #参数preference: array-like, shape (n_samples,) or float, optional
    #The number of exemplars, ie of clusters, is influenced by the input preferences value.
    #If the preferences are not passed as arguments, they will be set to the median of the input similarities.
    ap=AffinityPropagation(damping=0.5, preference=None)
    ap.fit(X)
    y_pred=ap.labels_
    return y_pred

#调用AffinityPropagation聚类函数，得到聚类标签
pred_labels=AffinityPropagation_Clustering(X)

#使用NMI(Normalized Mutual Information)作为评价指标进行评估
NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
print("Normalized Mutual Information: %0.3f" % NMI)


