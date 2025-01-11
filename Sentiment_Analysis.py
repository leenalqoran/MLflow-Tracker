import numpy as np
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download("popular")

# Commented out IPython magic to ensure Python compatibility.
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import plotly.express as px

from sklearn.model_selection import train_test_split

import re
import string
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
#from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from tqdm.auto import tqdm


from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

data = pd.read_csv('Train.csv')


data2 = pd.read_csv('Valid.csv')

data.head(5)

data2.head(5)

data1 = data.copy(deep=True)

test= data2.copy(deep=True)

data1["word_count"] = data1["text"].apply(lambda x: len(x.split(" ")))

test["word_count"] = test["text"].apply(lambda x: len(x.split(" ")))

data1["word_count"].describe()

test["word_count"].describe()

data1["char_count"] = data1["text"].apply(len)
data1["char_count"] .describe()

test["char_count"] = test["text"].apply(len)
test["char_count"] .describe()

data1.head()

test.head()


sample=data1.loc[0,"text"]
sample

sample1=test.loc[0,"text"]
sample1

def test_clean(fn , sample):
    print(sample, fn(sample), sep="\n")



stop_words = stopwords.words("english")

def remove_stop_words(text) :
    return " ".join([word for word in text.split(" ") if word not in stop_words])

test_clean(remove_stop_words,sample)

test_clean(remove_stop_words,sample1)

#Remove Hashtags
hash = re.compile(pattern="#[\w\d]+")

def remove_hashtag(text: str) -> str:
    return hash.sub(repl="", string=text)

test_clean(remove_hashtag,data1.loc[4,"text"])

test_clean(remove_hashtag,test.loc[4,"text"])

#Remove Punctuation Marks
punc_re = re.compile(r"""[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~،؟…«“\":\"…”]""")
def remove_punctation(text: str) -> str:
    return punc_re.sub(repl="", string=text)

#Remove Mention
mention_re = re.compile("@\w+")
def remove_mention(text):
    return mention_re.sub(repl="", string=text)
#Remove HTTP URLs
def remove_urls(data):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',data)

#Remove Numbers
numbers_re = re.compile("\d+")
def remove_numbers(text):
    return numbers_re.sub(repl="", string=text)
#Remove Multiple Whitespace
multiple_space_re = re.compile("\s{2,}")
def remove_multiple_whitespace(text):
    return multiple_space_re.sub(repl=" ", string=text)

def clean(text):
    text = remove_urls(text)
    text = remove_hashtag(text)
    text = remove_mention(text)
    text = remove_punctation(text)
    text = remove_numbers(text)
    text = remove_stop_words(text)
    text = remove_multiple_whitespace(text)
    text = text.lower().strip()


    return text

data1["clean_text"] = data1["text"].apply(clean)

test["clean_text"] = test["text"].apply(clean)

from nltk.stem.lancaster import LancasterStemmer
stemmer = nltk.LancasterStemmer()
stemmed_data = []
for sample in data1["clean_text"]:
    words = sample.split(" ")
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_data.append(" ".join(stemmed_words))

from nltk.stem.lancaster import LancasterStemmer
stemmer = nltk.LancasterStemmer()
Stemmed_data = []
for sample1 in test["clean_text"]:
    words = sample1.split(" ")
    Stemmed_words = [stemmer.stem(word) for word in words]
    Stemmed_data.append(" ".join(Stemmed_words))

texts_class_positive = data1[data1['label'] == 1]['clean_text']
texts_class_negative = data1[data1['label'] == 0]['clean_text']



print(data1['clean_text'][0],"\n\n",stemmed_data[0])

data1['clean_stemmed']=stemmed_data

test['clean_stemmed']=Stemmed_data

data1['clean_stemmed']

test['clean_stemmed']

lm=nltk.WordNetLemmatizer()
Lemmatizer_data=[]
for sample in data1['clean_text']:
    words=sample.split(' ')
    Lemmatizer_words=[lm.lemmatize(word) for word in words]
    Lemmatizer_data.append(" ".join(Lemmatizer_words))

lm=nltk.WordNetLemmatizer()
lemmatizer_data=[]
for sample1 in test['clean_text']:
    words1=sample1.split(' ')
    lemmatizer_words=[lm.lemmatize(word1) for word1 in words1]
    lemmatizer_data.append(" ".join(lemmatizer_words))



data1['Lemmatizer_data'] = Lemmatizer_data

test['lemmatizer_data'] = lemmatizer_data

X_train = data1['Lemmatizer_data']
X_test = test['lemmatizer_data']
y_train = data1['label']
y_test = test['label']

X_train.shape, X_test.shape, y_train.shape, y_test.shape


def test_pipeline(model):
    pipe=Pipeline([('bow',CountVectorizer()),('tfidf',TfidfTransformer()),('model',model)])
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    print('Accuracy Score: ',accuracy_score(y_test,y_pred))
    print(classification_report(y_test,y_pred))


from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier


test_pipeline(RidgeClassifier())
