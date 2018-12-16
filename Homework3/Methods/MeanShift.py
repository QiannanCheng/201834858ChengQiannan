# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:27:31 2018
Mean-shift聚类算法
@author: Qiannan Cheng
"""
import DataProcessing
from sklearn import metrics
from sklearn.cluster import MeanShift

#定义函数执行MeanShift聚类算法
# @param X 样本特征数据
# @return y_pred 
def MeanShiftAlgorithm(X):
    #参数bandwidth: float, optional
    #Bandwidth used in the RBF kernel.
    #If not given, the bandwidth is estimated using sklearn.cluster.estimate_bandwidth;
    ms=MeanShift(bandwidth=0.9)
    ms.fit(X)
    y_pred=ms.labels_
    return y_pred

if __name__=='__main__':
    #调用数据处理函数，得到sklearn函数可以直接调用的数据
    X,labels=DataProcessing.getAvailableData("../Tweets.txt")
    X=X.toarray() #type:<class 'numpy.ndarray'>
    
    #调用MeanShiftAlgorithm函数，得到聚类标签
    pred_labels=MeanShiftAlgorithm(X)
    
    #使用NMI(Normalized Mutual Information)作为评价指标进行评估
    NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
    print("Normalized Mutual Information: %0.3f" % NMI)

