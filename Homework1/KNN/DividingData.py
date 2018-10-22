# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 22:20:00 2018
分数据(18828)：test_data(20%_3759) train_data(80%_15069)
方法：每个类都取20%,合并作为测试数据集,其余作为训练数据集
@author: Qiannan Cheng
"""
import os

#构建classDict:{'class1':count,'class2':count,...}
classDict={}
datadir='..\Data\preprocessed_news'
classList=os.listdir(datadir)
for nclass in classList: 
    newsList=os.listdir(datadir+'\\'+nclass)
    classDict[nclass]=len(newsList)
print(classDict)

#classDcit每个元素的value*20%：每个类划分为testdata的数目
for cl,co in classDict.items():
    classDict[cl]=int(co*0.20) #hold out 20%
print(classDict)

#分别对每个新闻类随机取出20%的数据，合并
#得到文件：testData.txt
testNewsID=[] #记录作为测试数据的news_id:(class,id)
testf=open('testData.txt','w')
for cl,co in classDict.items():
    f=open('..\\VSM\\news_vector.txt','r')
    for line in f:
        r=line.strip().split('\t')
        if r[0]==cl:
            testf.write(line) #写入测试数据
            testNewsID.append((r[0],r[1])) 
            co=co-1
        if co==0:
            break
    f.close()
testf.close()
print(len(testNewsID))

#将除了测试数据的其他数据作为训练数据
#得到文件：trainData.txt
num=0
trainf=open('trainData.txt','w')
f=open('..\\VSM\\news_vector.txt','r')
for line in f:
    r=line.strip().split('\t')
    if (r[0],r[1]) not in testNewsID:
        trainf.write(line) #写入训练数据
        num+=1
f.close()
trainf.close()
print(num)
print("Dividing Data Finished!")
        
        

