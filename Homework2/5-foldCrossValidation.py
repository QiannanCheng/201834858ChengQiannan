# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:54:35 2018
五折交叉验证，计算朴素贝叶斯分类器的错误率
将数据分为五折，选择其中一折作为测试数据，其余作为训练数据计算错误率，取五次错误率的平均值作为最终错误率
@author: Qiannan Cheng
"""
import DividingData
import bayes
from matplotlib import pyplot as plt
import numpy as np

#生成五次实验的测试数据、训练数据、标注文件
for i in range(5):
    rightCateFile='classifyRightCate'+str(i)+'.txt'
    DividingData.dataSeg(i,rightCateFile)

#朴素贝斯分类器对五次实验的测试数据进行分类
for i in range(5):
    trainDir='trainData'+str(i)
    testDir='testData'+str(i)
    resultCateFile='classifyResultCate'+str(i)+'.txt'
    bayes.classifyNB(trainDir,testDir,resultCateFile)

#计算并记录五次实验的错误率
errorRateRecord=[]
for i in range(5):
    rightCateFile='classifyRightCate'+str(i)+'.txt'
    resultCateFile='classifyResultCate'+str(i)+'.txt'
    er=bayes.errorRate(rightCateFile,resultCateFile)
    errorRateRecord.append(er)
  
#绘制条形图：可视化每次实验的错误率
fig=plt.figure(1)
ax1=plt.subplot(111)
data=np.array([float(format(v,'.3f')) for v in errorRateRecord])
width=0.5
x_bar=np.arange(5)
rect=ax1.bar(left=x_bar,height=data,width=width,color="lightblue")
for rec in rect: #添加数据标签
    x=rec.get_x()
    height=rec.get_height()
    ax1.text(x+0.1,1.02*height,str(height))
ax1.set_xticks(x_bar)
ax1.set_xticklabels(("first","second","third","fourth","fifth"))
ax1.set_ylabel("error rate")
ax1.set_title("5-Fold Cross Validation")
ax1.grid(True)
ax1.set_ylim(0,0.2)
plt.show()
    
#计算五次实验的平均错误率
sumErrorRate=0.0
for i in range(5):
    print('errorRate%d = %f' % (i,errorRateRecord[i])) #输出每次实验的错误率
    sumErrorRate+=errorRateRecord[i]
averageErrorRate=sumErrorRate/5
print('averageErrorRate = %f' % (averageErrorRate)) #输出平均错误率
print('5-Fold Cross Validation Finished!')
    

