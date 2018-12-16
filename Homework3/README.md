# Homework3: Clustering with sklearn
## Data Processing
Define a data processing function, read data from the json file, return the data form which the clustering funtion and evaluation funtion 
in sklearn module can call directly.<br>
1. Read json file, get text list (form: [text1,text2,...]) and label array (form: [label1 label2 ...])<br>
2. Extracting features from the text list using TfidfVectorizer in sklearn, get vector representation matrix (dim: [samples_num,features_num])
###### Parameter Settings：
    vectorizer=TfidfVectorizer(max_df=0.5,           # Remove words with df>(0.5*doc_num)
                               max_features=3000,    # Building a vocabulary only considers max_features
                               min_df=2,             # Remove words with df<2
                               stop_words='english', # Remove english stopwords 
                               use_idf=True)         # Recalculate the weights using inverse-document-frequency
    X=vectorizer.fit_transform(data) # Return sparse matrix
## Clustering Algorithm
![](https://github.com/QiannanCheng/201834858ChengQiannan/blob/master/Homework3/Pictures/ClusteringAlgorithm.png)
