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
## Naive Bayes
1. 训练<br>
对训练样本进行统计，得到类cate下单词word出现的次数、每个新闻类中单词总数、训练数据中不重复的单词总数<br>
2. 计算p(cate_k|doc)：多项式模型+平滑技术+取对数（防止下溢<br>
p(word|cate_k)=(类cate_k下单词word出现的次数+1)/(类cate_k下单词总数+训练数据中不重复的单词总数)<br>
p(cate_k)=类cate_k下单词总数/训练数据中单词总数<br>
p(cate_k|doc)=p(w1,w2,...,wn|cate_k)*p(cate_k)=Log(p(w1|cate_k))+...+Log(p(wn|cate_k))+Log(p(cate_k))<br>
3. 比较p(cate_k)<br>
选择p(cate_k)最大的类为每个测试样本的分类结果，记录所有测试样本分类结果<br>
4. 计算错误率<br>
ErrorRate=分类错误的样本数/测试样本数
