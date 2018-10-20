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
## SVM
* Some key points:
  * Word segmentation (tokenization)
  * Statistics and calculations: `DF` and `IDF`
  * Word filter (construct controlled vocabulary): remove `non-information words` and `rare words`
  * Build and save a `dictionary` of all vocabulary
  * Calculate `TF` and  `TF normalization` (sub-linear TF scaling)
  * Calculate `TF-IDF` weights
  * Construct a VSM representation vector and `unitization`
* Output file:
  * news_corpus.txt 
    * format: news_class  news_id  news_content
  * word_dict.txt 
    * format: word  DF  IDF
  * news_vector.txt 
    * format: news_class  news_id  vector
