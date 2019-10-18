# -*- coding: utf-8 -*-


import pandas as pd
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer


'''
tf-idf(term frequency–inverse document frequency)是一种统计方法，
用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。

词频（tf）是一词语出现的次数除以该文件的总词语数。
假如一篇文件的总词语数是100个，而词语“母牛”出现了3次，
那么“母牛”一词在该文件中的词频就是3/100=0.03。
而计算文件频率（IDF）的方法是以文件集的文件总数，除以出现“母牛”一词的文件数。
所以，如果“母牛”一词在1,000份文件出现过，而文件总数是10,000,000份的话，
其逆向文件频率就是lg（10,000,000 / 1,000）=4。
最后的tf-idf的分数为0.03 * 4=0.12。
'''


# 按列分割读取
df = pd.read_csv('SMSSpamCollection', delimiter='\t', header=None)
y, x_train = df[0], df[1]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x_train)

lr = linear_model.LogisticRegression()
lr.fit(x, y)

testX = vectorizer.transform(['URGENT! Your mobile No. 1234 was awarded a Prize',
                              'Hey, honey, whats up'])
predictions = lr.predict(testX)
print(predictions)
