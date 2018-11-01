# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:58:46 2018
利用n-fold Cross Validation选择一个最佳的k值
利用最佳k值在测试数据上进行测试
@author: Qiannan Cheng
"""
import kNN
import six
import random
import numpy as np
import matplotlib.pyplot as plt

#打乱文件trainData.txt中行的顺序，使类分布均匀
f=open('trainData.txt','r')
f_out=open('OutOfOrderTrainData.txt','w')
content=f.readlines()
lastLine=content[-1]
content.remove(lastLine) 
random.shuffle(content) #将除了最后一行的其它行打乱顺序
for line in content:
    f_out.write(line)
f_out.write(lastLine)
f_out.close()
f.close()

#将训练数据随机分为五份
trainMat,trainLabels=kNN.file2matrix("OutOfOrderTrainData.txt") #得到训练矩阵和类标签列表
m=trainMat.shape[0] #矩阵行数
num=m//5 #每个fold所包含的news样本数
MatList=[] #3013/3013/3013/3013/3017
LabelsList=[]
for i in range(5): #i=0,1,2,3,4
    if i==4:
        MatList.append(trainMat[4*num:m])
        LabelsList.append(trainLabels[4*num:m])
    else:
        MatList.append(trainMat[i*num:i*num+num])
        LabelsList.append(trainLabels[i*num:i*num+num])
        
#5折交叉验证，得到使平均错误率最小的k值
kList=list(range(5,101,5)) #得到k的取值列表[5,10,15,20,...,100]
averageList=[] #k对应的平均错误率
minErrorRate=six.MAXSIZE
for k in kList:
    ersum=0.0
    for i in range(5): #0,1,2,3,4
        valMat=MatList[i] #选择其中一个作为校验矩阵
        valLabels=LabelsList[i] #校验矩阵对应的类标签
        idx=[0,1,2,3,4]
        idx.remove(i)
        #将其余四个矩阵合并作为训练矩阵
        traMat=np.row_stack((MatList[idx[0]],MatList[idx[1]],MatList[idx[2]],MatList[idx[3]]))
        #合并得到训练矩阵对应的类标签
        traLabels=LabelsList[idx[0]]+LabelsList[idx[1]]+LabelsList[idx[2]]+LabelsList[idx[3]]     
        er=kNN.errorRate(valMat,valLabels,traMat,traLabels,k) #得到错误率
        ersum+=er 
    average=ersum/5
    averageList.append(average) #记录k值对应的平均错误率
    print("k=%d, AverageErrorRate=%f" % (k,average))
    if average<minErrorRate:
        minErrorRate=average
        opk=k
       
#plot
plt.figure()
plt.plot(kList,averageList,'m',label='kNN')
plt.legend()
plt.xlabel('k Value')
plt.ylabel('Average Error Rate')
plt.show()
#print
print("Optimal k: %d, AverageErrorRate: %f" % (opk,minErrorRate))
#将最优k值在测试数据上进行测试
testMat,testLabels=kNN.file2matrix("testData.txt")
result=kNN.errorRate(testMat,testLabels,trainMat,trainLabels,opk)
print("errorRate on testData: "+str(result))
print("n-fold Cross Validation Finished!")


