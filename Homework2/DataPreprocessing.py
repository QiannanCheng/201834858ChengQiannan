# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:00:47 2018
对新闻数据进行预处理：
ps:之前是借助Lucene工具用java程序实现预处理，这里借助nltk模块用python程序实现预处理
1.词条化(利用非字母字符进行分割)
2.去除停用词
3.小写
4.Porter Stemmer
5.去除cf(Collection Frequency)<5的单词
@author: Qiannan Cheng
"""
import os
import re
from nltk.corpus import stopwords
import nltk

#新闻数据文件中的一行进行处理
#包括：用非字母字符进行tokenization，去停用词，小写化，词干化
def linePreprocess(line):
    stopword=stopwords.words('english') #得到英文停用词
    stemmer=nltk.PorterStemmer() #词干分析器
    splitter=re.compile('[^a-zA-Z]') #正则表达式：匹配非字母的其他字符
    words=[stemmer.stem(word.lower()) for word in splitter.split(line) if len(word)>0 and word.lower() not in stopword]
    return words

#对单个新闻文档进行处理
def newsPreprocess(news_class,news_id):
    srcFile='20news-18828/'+news_class+'/'+news_id #原始新闻文档路径
    targetFile='preprocessed_news/'+news_class+'/'+news_id #预处理后的新闻文档路径
    lineList=open(srcFile,'rb').readlines() #读取原始新闻文档，每一行作为列表的一个元素
    fw=open(targetFile,'w')
    for line in lineList:
        line=line.decode('utf-8','ignore')
        words=linePreprocess(line) #调用linePreprocess()处理每行文本，返回单词列表
        for word in words:
            fw.write('%s\n' % word)
    fw.close()

#遍历指定目录下的所有新闻文档，预处理后保存
def dirPreprocess():
    srcClassList=os.listdir('20news-18828')
    for i in range(len(srcClassList)):
        srcNewsList=os.listdir('20news-18828/'+srcClassList[i])
        targetClass='preprocessed_news/'+srcClassList[i]
        if os.path.exists(targetClass)==False:
            os.mkdir(targetClass) #创建类文件夹
        for j in range(len(srcNewsList)):
            newsPreprocess(srcClassList[i],srcNewsList[j]) 

#构建词典，统计每个单词的Collection frequency(cf)
#返回格式：[('word1',cf1),('word2',cf2),...]
#去除cf<5的单词，按单词首字母顺序对词典进行排序
def wordStat():
    wordMap={} 
    newWordMap={}
    #得到wordMap {'word1':cf1,'word2':cf2,...}
    classList=os.listdir('preprocessed_news')
    for i in range(len(classList)):
        newsList=os.listdir('preprocessed_news/'+classList[i])
        for j in range(len(newsList)):
            newsDir='preprocessed_news/'+classList[i]+'/'+newsList[j] #新闻文档的路径
            for line in open(newsDir,'rb').readlines():
                line=line.decode('utf-8','ignore')
                word=line.strip('\n') #新闻文档中的每一个单词
                wordMap[word]=wordMap.get(word,0.0)+1.0 #统计词频：单词在所有新闻文件中出现的次数
    #在worMap中去除cf<5的单词，得到newWordMap
    for key,value in wordMap.items():
        if value>4:
            newWordMap[key]=value
    #将newWordMap转化为元组列表的形式，按照单词的首字母进行排序 
    sortedNewWordMap=sorted(newWordMap.items())
    return sortedNewWordMap

#在所有新闻文档中去除词典中不包含的单词(cf<5)
def wordFilter():
    srcDir='preprocessed_news' #预处理后的新闻文档目录
    wordSet=set() #词典中所有单词的集合
    sortedNewWordMap=wordStat() #[('word1',cf1),('word2','cf2'),...]
    for i in range(len(sortedNewWordMap)):
        wordSet.add(sortedNewWordMap[i][0])
    srcClassList=os.listdir(srcDir)
    for i in range(len(srcClassList)):
        targetClassDir='used_news/'+srcClassList[i]
        srcClassDir='preprocessed_news/'+srcClassList[i]
        if os.path.exists(targetClassDir)==False:
            os.mkdir(targetClassDir) #建立新文件夹
        srcNewsList=os.listdir(srcClassDir)
        for j in range(len(srcNewsList)): #对于类文件夹中的每一个新闻文件
            targetNews=targetClassDir+'/'+srcNewsList[j]
            srcNews=srcClassDir+'/'+srcNewsList[j]
            fw=open(targetNews,'w')
            for line in open(srcNews,'rb').readlines():
                line=line.decode('utf-8','ignore')
                word=line.strip('\n')
                if word in wordSet:
                    fw.write('%s\n' % word)
            fw.close()
     
if __name__ == '__main__':
    dirPreprocess() 
    wordFilter()



