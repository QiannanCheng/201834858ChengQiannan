# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 11:29:38 2018
定义数据处理函数，返回sklearn聚类函数及评估函数可直接调用的数据形式
X (n_samples,n_features) 样本的特征向量表示
labels (n_samples,) 样本的cluster标签
@author: Qiannan Cheng
"""
import json 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def getAvailableData(jsonFile):
    #读取json数据文件得到：
    #data=['text1','text2',...] <class 'list'>
    #labels=[0 0 1 2 3 1 0 ...] <class 'numpy.ndarray'>
    data=[]
    labels=[]
    with open(jsonFile) as json_data:
        for line in json_data:
            tweet=json.loads(line)
            data.append(tweet["text"])
            labels.append(tweet["cluster"])
        labels=np.array(labels)
    
    #文本向量化
    #Extracting features from the dataset, using a sparse vectorizer
    #Vectorizer results are normalized
    vectorizer=TfidfVectorizer(max_df=0.5, #词汇表中过滤掉df>(0.5*doc_num)的单词
                               max_features=3000, #构建词汇表仅考虑max_features(按语料词频排序)
                               min_df=2, #词汇表中过滤掉df<2的单词
                               stop_words='english', #词汇表中过滤掉英文停用词
                               use_idf=True) #启动inverse-document-frequency重新计算权重
    X=vectorizer.fit_transform(data) #稀疏矩阵 type:<class 'scipy.sparse.csr.csr_matrix'>
    X=X.toarray() #type:<class 'numpy.ndarray'>
    return X,labels

