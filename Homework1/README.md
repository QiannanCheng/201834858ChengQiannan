# Homework1: VSM and KNN
## Data
* Original Dataset (20news-18828)
  * The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.
  * [20news-18828.tar.gz](http://qwone.com/~jason/20Newsgroups/ "Data download address") - 20 Newsgroups; duplicates removed, only "From" and "Subject" headers (18828 documents).
  ![](https://github.com/QiannanCheng/201834858ChengQiannan/blob/master/Homework1/NewsClass.png)
* Preprocessed Dataset (preprocessed_news)
## NewsPeprocess
* Java program implemented with `Lucene`
* The methods of preprocessing:
  * Tokenization
  * Normalization: lowercase
  * Remove stopwords
  * Remove punctuations
  * Stemming: Poter's Stemmer
  * Remove digital
* Output file:
  * preproccessed_news
## VSM
* Some key points:
  * Word segmentation (tokenization)
  * Statistics and calculations: `DF` and `IDF`
  * Word filter (construct controlled vocabulary): Remove `non-information words` and `rare words`
  * Build and save a `dictionary` of all vocabulary
  * Calculate `TF` and  `TF normalization` (sub-linear TF scaling)
  * Calculate `TF-IDF` weights
  * Construct a VSM representation vector
* Output file:
  * news_corpus.txt  (format: news_class  news_id  news_content)
  * word_dict.txt  (format: word  DF  IDF)
  * news_vector.txt  (format: news_class  news_id  vector)
## KNN
* Dividing Data
  * Total data(18828): Test data(20%_3759), Train data(80%_15069)
  * Method: Each class takes 20%, merged as a test data set, and the rest as a training data set
  * Output file:
    * testData.txt  (format: news_class  news_id  vector)
    * trainData.txt  (format: news_class  news_id  vector)
* kNN classifier
  * Some key functions:
    * CosSimilarity(vec,Mat): Calculate the cosine similarity between a vector and each row of a matrix
    * file2matrix(filename): Read the news_vertor file as a matrix
    * autoNorm(dataMat): Feature scaling, newValue=(oldValue-min)/(max-min) 
    * knnClassify(vecX,dataMat,labels,k): Return the classification result of a test sample
    * errorRate(testMat,testLabels,trainMat,trainLabels,k): Proportion of errors made over the whole set of instances
* n-fold Cross Validation
  * Use a five-fold cross validation to select an optimal k value and test it on the test data set
## Performance
![](https://github.com/QiannanCheng/201834858ChengQiannan/blob/master/Homework1/ResultFigure.png)
* Optimal k: 30 <br>
* Error rate on test data: 0.1372
