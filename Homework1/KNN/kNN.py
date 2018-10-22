# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:21:21 2018
kNN
@author: Qiannan Cheng
"""
import numpy as np
#特征值归一化：newValue=(oldValue-min)/(max-min)
def autoNorm(dataMat):
    minVals=dataMat.min(0) #每一列的最小值
    maxVals=dataMat.max(0) #每一列的最大值
    ranges=maxVals-minVals #特征值的范围
    normDataMat=np.zeros(np.shape(dataMat)) #构建一个空矩阵作为归一化后的特征值矩阵
    raw=dataMat.shape[0] #矩阵行数
    normDataMat=dataMat-np.tile(minVals,(raw,1)) #分子
    normDataMat=normDataMat/np.tile(ranges,(raw,1)) #newValue
    return normDataMat,ranges,minVals

a=np.array([1,2,3],[3,2,1],[2,3,4])
print(a)
i,j,k=autoNorm(a)
