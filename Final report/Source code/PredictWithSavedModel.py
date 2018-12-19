from nltk.stem import PorterStemmer
import  pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

mysentence = "Hello Mr. Smith, how are you doing today? The weather is great, and city is awesome. The sky is pinkish-blue. You shouldn't eat cardboard"
#Sentence Tokenization
tokenized_text=sent_tokenize(mysentence)
#print(tokenized_text)
#Word Tokenization
tokenized_word=word_tokenize(mysentence)
#print(tokenized_word)
######stopwords
stop_words=set(stopwords.words("english"))

filtered_sent=[]


for sentence in tokenized_text:
    for w in word_tokenize(sentence):
        if w not in stop_words:
            filtered_sent.append(w) #phương thức append cập nhật lại list
#print("Tokenized Sentence:",tokenized_word)
#print("Filterd Sentence:",filtered_sent)

#Stemming
ps = PorterStemmer()
stemmed_words=[]
for w in filtered_sent:
    stemmed_words.append(ps.stem(w))
print(len(stemmed_words))
#print("Filtered Sentence:",filtered_sent)
#print("Stemmed Sentence:",stemmed_words)

def cov_list_of_words_to_sent(words):
    sentence=' '
    for w in words :   #thêm từng từ vào câu cách nhau bởi khoảng trắng
        sentence += " " + w
    return sentence
sentences_test = cov_list_of_words_to_sent(stemmed_words)
#Vectorization(vector hóa).
with open('Output2.txt') as file:
    data = file.read()
data=[data]
cv = CountVectorizer(binary=True)
cv.fit_transform(data)
print(len(cv.get_feature_names()))
sentences_test=[sentences_test]
X_test = cv.transform(sentences_test)
loaded_model = pickle.load(open("finalmodel.pickle", 'rb'))
print(loaded_model.predict(X_test))



