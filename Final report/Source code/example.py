from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import os
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

######Đọc dữ liệu vào python
reviews_train = []
with open('./movie_data/full_train.txt', 'r', encoding="utf8") as f:
    for line in f.readlines():
        reviews_train.append(line.strip())
reviews_test = []
for line in open('./movie_data/full_test.txt', 'r', encoding="utf8"):
    reviews_test.append(line.strip()) #thêm đối tượng line.strip() vào list
print(reviews_train[0])
print(reviews_test[0])
####
######Làm sạch dữ liệu
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "
def preprocess_reviews(reviews):   #định nghĩa hàm
    reviews = [REPLACE_NO_SPACE.sub(NO_SPACE, line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(SPACE, line) for line in reviews]
    return reviews
reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)
print(reviews_train_clean[0])
print(reviews_test_clean[0])

train_tokenized_text=[[] for _ in range(len(reviews_train_clean))]#mảng 2 chiều
test_tokenized_text=[[] for _ in range(len(reviews_test_clean))]#mảng 2 chiều

######stopwords
stop_words=set(stopwords.words("english"))
#Remove stopword cho từng câu
#remove stopword cho tập train
train_filtered_sent = [[] for _ in range(len(reviews_train_clean))]#mảng 2 chiều
i = 0
for sentence in reviews_train_clean:
    for w in word_tokenize(sentence):
        if w not in stop_words:
            train_filtered_sent[i].append(w) #phương thức append cập nhật lại list
    i+=1
#remove stopword cho tập test
test_filtered_sent = [[] for _ in range(len(reviews_test_clean))]#mảng 2 chiều
i = 0
for sentence in reviews_test_clean:
    for w in word_tokenize(sentence):
        if w not in stop_words:
            test_filtered_sent[i].append(w) #phương thức append cập nhật lại list
    i+=1
print(train_filtered_sent[0])
print(test_filtered_sent[0])
#Stemming
#################################
ps = PorterStemmer()
#tập train
train_stemmed_words = [[] for _ in range(len(train_filtered_sent))]#mảng 2 chiều
j = 0
for index in range(len(train_filtered_sent)):
    for w in train_filtered_sent[index]:
        train_stemmed_words[j].append(ps.stem(w))
    j+=1
#tập test
test_stemmed_words = [[] for _ in range(len(test_filtered_sent))]#mảng 2 chiều
j = 0
for index in range(len(test_filtered_sent)):
    for w in test_filtered_sent[index]:
        test_stemmed_words[j].append(ps.stem(w))
    j+=1
print(train_stemmed_words[0])
print(test_stemmed_words[0])

# Covert các từ thành câu
def cov_list_of_words_to_sent(words):
    sentence=' '
    for w in words :   #thêm từng từ vào câu cách nhau bởi khoảng trắng
        sentence += " " + w
    return sentence
#khai báo list
sentences_train_vectorizer = ['' for i in range(len(train_stemmed_words))]
sentences_test_vectorizer = ['' for i in range(len(test_stemmed_words))]
#print(len(train_stemmed_words))

for i in range(len(train_stemmed_words)):  #covert từng dòng
    sentences_train_vectorizer[i] = cov_list_of_words_to_sent(train_stemmed_words[i])
for i in range(len(test_stemmed_words)):  #covert từng dòng
    sentences_test_vectorizer[i] = cov_list_of_words_to_sent(test_stemmed_words[i])
print(sentences_train_vectorizer[0])
print(sentences_test_vectorizer[0])

#Vectorization(vector hóa).
cv = CountVectorizer(binary=True)
cv.fit(sentences_train_vectorizer) #Học từ vựng idf từ tập huấn luyện
print()
#Write sentences_train_vectorizer file Output.txtưt0
#
X = cv.transform(sentences_train_vectorizer)  #chuyển đổi tài liệu thành ma trận tài liệu
X_test = cv.transform(sentences_test_vectorizer)
print("cac cot")
feature_sentence = cov_list_of_words_to_sent(cv.get_feature_names())
print(feature_sentence)
text_file = open("Output.txt", "w")
text_file.write(feature_sentence)
text_file.close()

print(X[0].toarray())
print(len(cv.get_feature_names()))
print(X.shape)

#Tạo bộ phân loại
target = [1 if i < 12500 else 0 for i in range(25000)]
X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size=0.75
)
for c in [0.01, 0.05, 0.25, 0.5, 1]:    #Tìm giá trị tối ưu của c
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    print("Accuracy for C=%s: %s"
          % (c, accuracy_score(y_val, lr.predict(X_val))))

#Train Final Model
final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
#save model after training
import pickle
save_classifier = open("finalmodel.pickle","wb")
pickle.dump(final_model, save_classifier)
save_classifier.close()

print ("Final Accuracy: %s"
       % accuracy_score(target, final_model.predict(X_test)))



