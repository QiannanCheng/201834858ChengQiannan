# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:46:53 2018
分数据：20%测试数据 80%训练数据
在每个新闻类中划分出20%的新闻文档，合并作为测试数据，其余作为训练数据
@author: Qiannan Cheng
"""
import os

#划分出一定比例的训练数据和测试数据，将标注的测试数据记录在文件中
# @param fold 划分第几折作为测试数据
# @param rightCate 记录测试数据标注<newsName_newsClass newsClass>
# @param trainDataProp 划分为训练数据的比例，默认为0.8
#ps: 由于20news-18828中含有重名文档，这里利用newsName_newsClass作为文档标识符
def dataSeg(fold,rightCateFile,trainDataProp=0.8):
    fw=open(rightCateFile,'w')
    srcDir='used_news'
    srcClassList=os.listdir(srcDir)
    for i in range(len(srcClassList)):
        srcClassDir=srcDir+'/'+srcClassList[i] 
        srcNewsList=os.listdir(srcClassDir) 
        m=len(srcNewsList) #新闻类中所包含的新闻文档的数目
        testBeginIndex=fold*(m*(1-trainDataProp)) #测试数据的起始索引
        testEndIndex=(fold+1)*(m*(1-trainDataProp)) #测试数据的结束索引
        for j in range(m): #遍历新闻类下的所有新闻文档
            if (j>=testBeginIndex) and (j<testEndIndex):
                #记录标注：文档id(文档名_所属类) 所属类
                fw.write('%s %s\n' % (srcNewsList[j]+'_'+srcClassList[i],srcClassList[i]))
                targetClassDir='TestData'+str(fold)+'/'+srcClassList[i]
            else:
                targetClassDir='TrainData'+str(fold)+'/'+srcClassList[i]
            if os.path.exists(targetClassDir)==False:
                os.makedirs(targetClassDir)
            nf=open(targetClassDir+'/'+srcNewsList[j],'w')
            lineList=open(srcClassDir+'/'+srcNewsList[j],'rb').readlines()
            for line in lineList:
                line=line.decode('utf-8','ignore')
                nf.write('%s\n' % line.strip('\n'))
            nf.close()
    fw.close()
    
if __name__ == '__main__':
    dataSeg(0,'classifyRightCate0.txt')
    
    
    
