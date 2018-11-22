# Homework2: Naive Bayes Classifier (NBC)
## Data Preprocessing
  Previously, the `Lucene` tool was used to implement preprocessing with a java program. Here, the `nltk` module was used to implement   preprocessing with a python program.
The methods of preprocessing:
  1. Tokenization(Split with non-alphabetic characters)
  2. Remove stopwords
  3. Lower case
  4. Porter Stemmer
  5. Remove words with cf(Cellection Frequency)<5
## Dividing Data
  20% test data and 80% train data
  Divide 20% of news in each class, merge as test data, and the rest as training data.
