import numpy as np
import networkx as nx
import sys
sys.path.append('../')
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

embeddings = np.zeros((G.number_of_nodes(), n_dim))
for i, node in enumerate(H.nodes()):
    embeddings[int(node), :] = model.wv[str(node)]

x_train = np.zeros((len(cleaned_train_data), embeddings_text_train.shape[1]+embeddings.shape[1]))
for i in range(len(cleaned_train_data)):
    x_train[i] = np.concatenate((embeddings_text_train[i], embeddings[int(train_hosts[i])]))

x_test = np.zeros((len(cleaned_test_data), embeddings_text_test.shape[1]+embeddings.shape[1]))
for i in range(len(cleaned_test_data)):
    x_test[i] = np.concatenate((embeddings_text_test[i], embeddings[int(test_hosts[i])]))
    
# LGR
clf_lgr = Pipeline([('clf', LogisticRegression(solver='lbfgs',
                                               multi_class='auto', max_iter=5000))])
clf_lgr.fit(x_train, y_train)

y_pred = clf_lgr.predict(x_test)

# Write predictions to a file
with open('../word2vec_deepwalk.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = clf_lgr.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)