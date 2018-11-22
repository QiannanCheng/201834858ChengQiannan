# Homework2: Naive Bayes Classifier (NBC)
## Data Preprocessing
Previously, the Lucene tool was used to implement preprocessing with a java program. Here, the nltk module was used to implement preprocessing with a python program.<br>
The methods of preprocessing:<br>
1. Tokenization(Split with non-alphabetic characters)<br>
2. Remove stopwords<br>
3. Lower case<br>
4. Porter Stemmer<br>
5. Remove words with cf(Cellection Frequency)<5<br>
## Dividing Data
20% test data and 80% train data.<br>
Divide 20% of news in each class, merge as test data, and the rest as training data.<br>
