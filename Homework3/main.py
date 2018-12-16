# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:59:54 2018
主函数
@author: Qiannan Cheng
"""
import DataProcessing
from Methods.AffinityPropagation import *
from Methods.AgglomerativeClustering import *
from Methods.GaussianMixture import *
from Methods.KMeans import *
from Methods.MeanShift import *
from Methods.SpectralClustering import *
from Methods.WardHierarchicalClustering import *
from Methods.DBSCAN import *
import numpy as np
from sklearn import metrics
from matplotlib import pyplot as plt

#调用数据处理函数，得到sklearn函数可以直接调用的数据
X,labels=DataProcessing.getAvailableData("Tweets.txt")
X=X.toarray() #type:<class 'numpy.ndarray'>
true_k=np.unique(labels).shape[0] #cluster的数目

#建立聚类算法列表和评估分数列表
ClusteringAlgorithmList=['AffinityPropagation','AgglomerativeClustering', \
                     'DBSCAN','GaussianMixture','KMeans','MeanShift', \
                     'SpectralClustering','WardHierarchicalClustering']
EvaluationList=[]

#循环调用8种聚类算法对Tweets文本进行聚类，得到聚类标签，保存评估分数
for ClusteringAlgorithm in ClusteringAlgorithmList:
    ClusteringFunc=ClusteringAlgorithm+'Algorithm'
    param_num=eval(ClusteringFunc).__code__.co_argcount #返回聚类函数的参数数目
    if param_num==3:
        pred_labels,_=eval(ClusteringFunc)(X,true_k,True)
    elif param_num==2:
        pred_labels=eval(ClusteringFunc)(X,true_k)
    else:
        pred_labels=eval(ClusteringFunc)(X)
    #使用NMI(Normalized Mutual Information)作为评价指标进行评估
    NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
    NMI=float('%.3f' % NMI)
    EvaluationList.append(NMI)

#绘制条形图，直观比较8种聚类算法的效果
plt.bar(range(8),EvaluationList,align='center',color='steelblue',alpha =0.8)
plt.ylabel('NMI') #添加y轴标签
plt.title('Comparing about eight clustering algorithms') #添加标题
plt.xticks(range(8),ClusteringAlgorithmList) #添加x轴刻度标签
plt.ylim([0,1]) #设置y轴刻度范围
for x,y in enumerate(EvaluationList):
    plt.text(x,y+0.02,'%s' % y,ha='center')
plt.show()

#排序输出聚类算法及其对应的评估分数
AlgorithmScore={}
for i in range(8):
    AlgorithmScore[ClusteringAlgorithmList[i]]=EvaluationList[i]
AlgorithmScoreSorted=sorted(AlgorithmScore.items(),key = lambda x:x[1],reverse = True) #按评估分数进行排序
print('Clustering algorithm and corresponding NMI score:')
for i in range(8):
    print(str(i+1)+'. '+AlgorithmScoreSorted[i][0]+': '+str(AlgorithmScoreSorted[i][1]))

