# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 17:21:21 2018
kNN
@author: Qiannan Cheng
"""
import numpy as np
import operator #运算符模块

#特征值归一化：newValue=(oldValue-min)/(max-min)
def autoNorm(dataMat):
    minVals=dataMat.min(0) #每一列的最小值
    maxVals=dataMat.max(0) #每一列的最大值
    ranges=maxVals-minVals #特征值的范围
    normDataMat=np.zeros(np.shape(dataMat)) #构建一个空矩阵作为归一化后的特征值矩阵
    m=dataMat.shape[0] #矩阵行数
    normDataMat=dataMat-np.tile(minVals,(m,1)) #分子
    normDataMat=normDataMat/np.tile(ranges,(m,1)) #newValue
    return normDataMat,ranges,minVals

#读取news_vertor文件为一个矩阵
def file2matrix(filename):
    f=open(filename,'r')
    lines=f.readlines() #将文件读取为一个list，其中一个line作为一个元素
    row=len(lines) #行数
    vector=lines[0].strip().split('\t')[2]
    col=len(vector.split()) #列数
    print("Matrix_dim: ["+str(row)+","+str(col)+"]") #输出矩阵维数
    returnMat=np.zeros((row,col)) #创建特征矩阵[row,col]
    classLabel=[] #创建类标签列表
    index=0
    for line in lines:
        vec=line.strip().split('\t')[2].split() #字符串列表
        vec=[float(k) for k in vec] #将list元素转为float类型
        returnMat[index,:]=vec[0:col] 
        label=line.strip().split('\t')[0]
        classLabel.append(label)
        index+=1
    f.close()
    return returnMat,classLabel

#计算一个vector与一个matrix每一行的CosSimilarity
def CosSimilarity(vec,Mat):
    fenzi=np.dot(vec,Mat.T) #分子：vec与矩阵每一行的内积
    vecNorm=np.linalg.norm(vec) #向量的模
    MatNorm=np.linalg.norm(Mat,axis=1) #矩阵行向量的模
    fenmu=vecNorm*MatNorm #分母：vec的模与矩阵行向量模的乘积
    cos=fenzi/fenmu #CosSimilarity：内积/模的乘积
    return cos

#knn分类器
def knnClassify(vecX,dataMat,labels,k):
    cos=CosSimilarity(vecX,dataMat) #计算与train_data之间的余弦相似度
    sortedIndex=np.argsort(-cos) #返回value从大到小的index
    #识别k个近邻
    classCount={}
    for i in range(k): 
        voteClass=labels[sortedIndex[i]] #排序第i个近邻的类标签
        c=cos[sortedIndex[i]] #排序第i个近邻的余弦相似度
        classCount[voteClass]=classCount.get(voteClass,0)+c #weights_factor: CosSimilarity
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) #排序vote
    return sortedClassCount[0][0] #返回得分最高的类

#Evaluation: Classifier error rate
def errorRate(testMat,testLabels,trainMat,trainLabels,k):
    trainNormMat,ranges,minVals=autoNorm(trainMat)
    m=testMat.shape[0] #test样本数目
    errorCount=0.0
    for i in range(m):
        testNormVec=(testMat[i,:]-minVals)/ranges #test样本特征值归一
        classifyResult=knnClassify(testNormVec,trainNormMat,trainLabels,k)
        print("kNN class: %s, real class : %s" % (classifyResult, testLabels[i]))
        if(classifyResult!=testLabels[i]):
            errorCount+=1.0
    ErrorRate=errorCount/float(m)
    return ErrorRate

if __name__ == '__main__':
    testMat,testLabels=file2matrix("testData.txt")
    trainMat,trainLabels=file2matrix("trainData.txt")
    result=errorRate(testMat,testLabels,trainMat,trainLabels,100)
    print("errorRate: "+str(result))