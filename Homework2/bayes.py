# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:48:37 2018
朴素贝叶斯：多项式模型+平滑技术+取对数(防止下溢出)
@author: Qiannan Cheng
"""
import os
import numpy as np

#训练函数
# @param trainDir 训练数据目录
# @return cateWordCount <类_单词,出现次数>
# @return cateWordNum <类,单词总数>
# @return vocabNum 训练数据中不重复的单词总数
def trainNB(trainDir):
    cateWordCount={}
    cateWordNum={}
    vocab=set()
    classList=os.listdir(trainDir)
    for i in range(len(classList)):
        count=0 #记录每个新闻类中单词总数
        classDir=trainDir+'/'+classList[i] #类目录
        newsList=os.listdir(classDir)
        for j in range(len(newsList)):
            newsDir=classDir+'/'+newsList[j]
            lines=open(newsDir,'rb').readlines()
            for line in lines:
                line=line.decode('utf-8','ignore')
                count+=1
                word=line.strip('\n')
                vocab.add(word) #记录训练数据中不重复词汇
                key=classList[i]+'_'+word
                cateWordCount[key]=cateWordCount.get(key,0)+1 #记录每个类中每个单词出现的次数
        cateWordNum[classList[i]]=count
    vocabNum=len(vocab)
    return cateWordCount,cateWordNum,vocabNum

#计算测试文档属于某个类的概率p(cate|doc)=p(w1,w2,...|cate)*p(cate)
#多项式模型+平滑技术：
#p(word|cate)=(类cate下单词word出现的次数+1)/(类cate下单词总数+训练数据中不重复的单词总数)
#p(cate)=类cate下单词总数/训练数据中单词总数
def calCateProb(k,testNewsWords,cateWordCount,cateWordNum,totalNum,vocabNum):
    prob=0
    wordNumInCate=cateWordNum[k] #新闻类k中单词总数
    for i in range(len(testNewsWords)):
        key=k+'_'+testNewsWords[i]
        if key in cateWordCount: 
            wordCountInCate=cateWordCount[key] 
        else:
            wordCountInCate=0.0
        xcProb=np.log((wordCountInCate+1)/(wordNumInCate+vocabNum))
        prob=prob+xcProb
    res=prob+np.log(wordNumInCate)-np.log(totalNum)
    return res

#朴素贝叶斯对测试文档进行分类
def classifyNB(trainDir,testDir,resultCateFile):
    fw=open(resultCateFile,'w')
    #训练分类器
    cateWordCount,cateWordNum,vocabNum=trainNB(trainDir)
    #得到训练数据单词总数
    totalNum=sum(cateWordNum.values())
    #对测试文档做分类
    testClassList=os.listdir(testDir)
    for i in range(len(testClassList)):
        testClassDir=testDir+'/'+testClassList[i]
        testNewsList=os.listdir(testClassDir)
        for j in range(len(testNewsList)):
            testNewsWords=[] #测试文档的单词列表
            testNewsDir=testClassDir+'/'+testNewsList[j]
            lines=open(testNewsDir,'rb').readlines()
            for line in lines:
                line=line.decode('utf-8','ignore')
                word=line.strip('\n')
                testNewsWords.append(word)
            maxP=0.0
            trainClassList=os.listdir(trainDir)
            for k in range(len(trainClassList)):
                p=calCateProb(trainClassList[k],testNewsWords,cateWordCount,cateWordNum,totalNum,vocabNum)
                if k==0:
                    maxP=p
                    bestCate=trainClassList[k]
                    continue
                if p>maxP:
                    maxP=p
                    bestCate=trainClassList[k]
            fw.write('%s %s\n' % (testNewsList[j]+'_'+testClassList[i],bestCate))
    fw.close()
    
#计算朴素贝叶斯分类的错误率
def errorRate(rightCateFile,resultCateFile):
    rightCateDict={} 
    resultCateDict={}
    errorCount=0.0
    for line in open(rightCateFile,'rb').readlines():
        line=line.decode('utf-8','ignore')
        (newsID,cate)=line.strip('\n').split()
        rightCateDict[newsID]=cate
    for line in open(resultCateFile,'rb').readlines():
        line=line.decode('utf-8','ignore')
        (newsID,cate)=line.strip('\n').split()
        resultCateDict[newsID]=cate
    for key in rightCateDict.keys():
        #输出分类结果
        print('新闻ID：'+key)
        print('朴素贝叶斯分类：'+resultCateDict[key])
        print('新闻真实所属类：'+rightCateDict[key])
        if rightCateDict[key]!=resultCateDict[key]:
            errorCount+=1.0
    errorRate=errorCount/len(rightCateDict)
    print('error rate: %f' % (errorRate))
    return errorRate

if __name__ == '__main__':
    classifyNB('trainData0','testData0','classifyResultCate0.txt')
    errorRate('classifyRightCate0.txt','classifyResultCate0.txt')
    print('Naive Bayes Finished!')
        
        
        
        
        
        

            
    
        
        
        
        
        
        
        
        
