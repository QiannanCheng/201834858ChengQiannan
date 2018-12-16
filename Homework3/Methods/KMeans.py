# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 20:48:14 2018
K-Means聚类算法
@author: Qiannan Cheng
"""
import json
import numpy as np
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

#定义函数实现KMeans聚类算法
# @param X 数据
# @param k 簇的数目
# @param minibatch 布尔型(True:MiniBatchKMeans False:KMeans)
# @return y_pred,km
# MinibatchKMeans:
# Alternative online implementation that does incremental updates of the centers positions using mini-batches. 
# For large scale learning (say n_samples > 10k) MiniBatchKMeans is probably much faster than the default batch implementation.
def KMeansAlgorithm(X, k, minibatch):
    if minibatch:
        km=MiniBatchKMeans(n_clusters=k, #形成的簇数以及生成的中心数
                           init='k-means++', #用智能的方式选择初始聚类中心以加速收敛
                           n_init=3, #随机初始化的次数
                           batch_size=100, #Size of the mini batches
                           )
    else:
        km=KMeans(n_clusters=k, 
                  init='k-means++',
                  max_iter=300, #一次单独运行的最大迭代次数
                  n_init=10, #使用不同的聚类中心进行初始化的次数
                  )
    km.fit(X) 
    y_pred=km.labels_
    return y_pred,km

if __name__=='__main__':
    #读取json数据文件最终得到：
    #data=['text1','text2',...] <class 'list'>
    #labels=[0 0 1 2 3 1 0 ...] <class 'numpy.ndarray'>
    data=[]
    labels=[]
    with open("../Tweets.txt") as json_data:
        for line in json_data:
            tweet=json.loads(line)
            data.append(tweet["text"])
            labels.append(tweet["cluster"])
        labels=np.array(labels)
    true_k=np.unique(labels).shape[0] #cluster的数目
    
    #文本向量化
    #Extracting features from the dataset, using a sparse vectorizer
    #Vectorizer results are normalized
    vectorizer=TfidfVectorizer(max_df=0.5, #词汇表中过滤掉df>(0.5*doc_num)的单词
                               max_features=3000, #构建词汇表仅考虑max_features(按语料词频排序)
                               min_df=2, #词汇表中过滤掉df<2的单词
                               stop_words='english', #词汇表中过滤掉英文停用词
                               use_idf=True) #启动inverse-document-frequency重新计算权重
    X=vectorizer.fit_transform(data) #shape:[n_samples,n_features] type:<class 'scipy.sparse.csr.csr_matrix'>
    print("n_samples: %d, n_features: %d" % X.shape)
    
    #调用KMeansAlgorithm聚类函数，得到聚类标签
    pred_labels,km=KMeansAlgorithm(X, true_k, True)
    
    #使用NMI(Normalized Mutual Information)作为评价指标进行评估
    NMI=metrics.normalized_mutual_info_score(labels,pred_labels)
    print("Normalized Mutual Information: %0.3f" % NMI)
    
    #get top terms per cluster
    print("Top terms per cluster:")
    order_centroids=km.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names() #向量的每个特征代表的单词(2172d,2172features,2172words)
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()



