import numpy as np
import networkx as nx
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import string
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import json
import collections
from sklearn.metrics import log_loss
from utils_deepwalk import deepwalk
from preprocess import remove_duplicates, import_texts, generate_data, clean_host_texts


data = '../data/'
train_file = data + 'train.csv'
train_hosts, y_train = remove_duplicates(train_file)
texts_path = '../text/text'
texts = import_texts(texts_path)

with open(data + 'test.csv', 'r') as f:
    test_hosts = f.read().splitlines()
    
train_data = generate_data(train_hosts, texts)
test_data = generate_data(test_hosts, texts)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«'
stpwords = stopwords.words('french')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)

# LGR
clf_lgr = Pipeline([('clf', LogisticRegression(solver='lbfgs',
                                               multi_class='auto', max_iter=1000))])
w = Word2Vec(size=128, window=8, min_count=0, sg=1, workers=8)
cleaned_train_data = [k.split(' ') for k in cleaned_train_data]
cleaned_test_data = [k.split(' ') for k in cleaned_test_data]
cleaned_data = cleaned_train_data+cleaned_test_data
w.build_vocab(cleaned_data)
w.train(cleaned_data, total_examples=w.corpus_count, epochs=5)

embeddings_text_train = np.zeros((len(cleaned_train_data), 128))
for i in range(len(cleaned_train_data)):
    for k in cleaned_train_data[i]:
        embeddings_text_train[i] += w.wv[k]

embeddings_text_test = np.zeros((len(cleaned_test_data), 128))
for i in range(len(cleaned_test_data)):
    for k in cleaned_test_data[i]:
        embeddings_text_test[i] += w.wv[k]

G = nx.read_weighted_edgelist(data + 'edgelist.txt', create_using=nx.DiGraph())
H = G.subgraph(train_hosts + test_hosts)
n = H.number_of_nodes()
n_hosts = len(train_hosts)


n_dim = 3
n_walks = 100
walk_length = 200
model = deepwalk(H, n_walks, walk_length, n_dim)

embeddings = np.zeros((H.number_of_nodes(), n_dim))
for i, node in enumerate(H.nodes()):
    embeddings[i, :] = model.wv[str(node)]

x_train = np.zeros((len(cleaned_train_data), embeddings_text_train.shape[1]+embeddings.shape[1]))
for i in range(len(cleaned_train_data)):
    x_train[i] = np.concatenate((embeddings_text_train[i], embeddings[int(train_hosts[i])]))

# Evaluate the model
X_train, X_eval, Y_train, Y_eval = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

clf_lgr.fit(X_train, Y_train)
print(clf_lgr.score(X_eval, Y_eval))
print(log_loss(Y_eval, clf_lgr.predict_proba(X_eval)))
