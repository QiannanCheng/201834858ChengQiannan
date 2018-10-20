# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:40:40 2018
svm: tf-idf weights
get news_vector.txt
txt_from: news_class news_id news_vector
@author: Qiannan Cheng
"""
import os
import six
import math
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

#分词: news=[['word1','word2',...],[...],[...],...]
def word_seg(news):
    print("word_seg...")
    news=[word_tokenize(sent) for sent in tqdm(news)] #show the progress of word segmentation with tqdm
    return news

#获取单词统计: word_stats={'word1':{'df':int,'idf':float},...}
def cal_words_stat(news):
    print("cal_words_stat...")
    words_stats={}
    news_num=len(news) #news的数目
    for ws in news:
        for w in set(ws): #遍历一个news中的单词集合
            if w not in words_stats:
                words_stats[w]={}
                words_stats[w]['df']=0
                words_stats[w]['idf']=0
            words_stats[w]['df']+=1 #统计每个单词的df
    for w,winfo in words_stats.items(): #将dict转为list的形式[(word,{'df':int,'idf':float}),...]进行遍历
        words_stats[w]['idf']=np.log((1.+news_num)/(1.+winfo['df'])) #计算每个单词的idf
    return words_stats

#过滤无用单词 construct controlled vocabulary
def word_filter(news,words_stats):
    print("word_filter...")
    words_useless=set()
    min_freq=2
    max_freq=six.MAXSIZE
    for w,winfo in words_stats.items():
        #filter too frequent words and rare words
        if winfo['df']<min_freq or winfo['df']>max_freq:
            words_useless.add(w)
    #filter with useless words
    news=[[w for w in ws if w not in words_useless] for ws in tqdm(news)]
    for wu in words_useless:
        words_stats.pop(wu) #在words_stats字典中删除元素
    return news,words_stats,words_useless

#构建词典(包含全部vocabulary的一个有序list)
def build_word_dict(news):
    print("build_word_dict...")
    word_dict=[]
    for ws in news:
        for w in ws:
            if w not in word_dict:
                word_dict.append(w)
    return word_dict

#save word_dict.txt
#form: word df idf
def save_word_dict(word_dict,words_stats):
    print("save_word_dict...")
    savef=open('word_dict.txt','w')
    for w in word_dict:
        line=w+'\t'+str(words_stats[w]['df'])+'\t'+str(words_stats[w]['idf'])
        savef.write(line+'\n')
    savef.close()

#计算tf + tf normalization(sub-linear tf scaling)
def cal_tf_norm(news):
    print("cal_tf_norm...")
    news_tf=[] #格式: [{'word1':tf1,'word2':tf2,...},{...},{...},...]
    for ws in news:
        d=dict() #格式: {'word1':tf1,'word2':tf2,...}
        for w in ws:
           if w not in d:
               d[w]=0
           d[w]+=1
        #tf normalization
        for w,tf in d.items():
            tf_norm=1+np.log(tf) 
            d[w]=tf_norm
        news_tf.append(d)
    return news_tf

#向量单位化: v/|v|
def vector_unitization(vec):
    vsum=0.0
    for e in vec:
        vsum+=e*e
    vsum=math.sqrt(vsum) #|v|
    for i,e in enumerate(vec):
        vec[i]/=vsum
    return vec

if __name__ == '__main__':
    #得到news_corpus.txt
    #格式：news_class news_id text
    print("get_news_corpus...")
    news_info=[] #记录(news_class,news_id)
    news=[] #记录新闻文本['text1','text2','text3',...]
    corpus_file=open('news_corpus.txt','w')
    for root,dirs,files in os.walk("..\Data\preprocessed_news"):
        for file in files:
            news_path=os.path.join(root,file) #news文件路径
            news_class=news_path.strip().split('\\')[-2] #news所属的类
            news_id=file #news编号
            news_file=open(news_path,'r')
            content=news_file.read().strip() #news文本内容
            news_file.close()
            news_info.append((news_class,news_id))
            news.append(content)
            corpus_line=news_class+'\t'+news_id+'\t'+content+'\n'
            corpus_file.write(corpus_line)
    corpus_file.close()
    #分词
    news=word_seg(news)
    #获取单词统计(df,idf)
    words_stats=cal_words_stat(news)
    #过滤无用单词
    news,words_stats,words_useless=word_filter(news,words_stats)
    #构建顺序词典
    word_dict=build_word_dict(news)
    #保存词典
    save_word_dict(word_dict,words_stats)
    #计算词频(tf)
    news_tf=cal_tf_norm(news)
    #计算tf-idf权重/得到news_vertor/向量单位化/保存到news_vector.txt
    vdim=len(word_dict) #向量维数
    print("cal_tf-idf_weights...")
    print("get_norm_vector("+str(vdim)+"d)...")
    vec_file=open('news_vector.txt','w')
    for idx,n in enumerate(news_info): #idx:序号 n:值(news_class,news_id)
        vec=[] 
        for w in word_dict:
            if w in news_tf[idx]:
                vec.append(news_tf[idx][w]*words_stats[w]['idf']) #计算tf-idf
            else:
                vec.append(0)
        norm_vec=vector_unitization(vec) #得到单位向量
        line=n[0]+'\t'+n[1]+'\t'+' '.join(['%f' % k for k in norm_vec])
        vec_file.write(line+'\n')
    vec_file.close()
    print("svm finished!")
        
    
    
    
    
     
            
            

        
        
        
    