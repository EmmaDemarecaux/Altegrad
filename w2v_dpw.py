# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:56:19 2020

@author: Khaoula Belahsen
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression

import os
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import codecs
import string
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import pickle
import collections
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier


def random_walk(G, node, walk_length):

    walk = [node]
    for i in range(walk_length):
      neighbors = list(G.neighbors(walk[-1]))
      if len(neighbors)>0 :
        walk.append(neighbors[randint(0,len(neighbors)-1)])
      else :
        break
    walk = [str(node) for node in walk]
    return walk



def generate_walks(G, num_walks, walk_length):
    walks = []
    list_nodes = list(G.nodes())
    for i in range(num_walks):
      for node in G.nodes() :
        walks.append(random_walk(G,node,walk_length))
    
    return walks


def deepwalk(G, num_walks, walk_length, n_dim):
    print("Generating walks")
    walks = generate_walks(G, num_walks, walk_length)

    print("Training word2vec")
    model = Word2Vec(size=n_dim, window=8, min_count=0, sg=1, workers=8)
    model.build_vocab(walks)
    model.train(walks, total_examples=model.corpus_count, epochs=5)

    return model


def clean_host_texts(data, tok, stpwds, punct, verbosity=5):
    cleaned_data = []
    for counter, host_text in enumerate(data):
        # converting article to lowercase already done
        temp = BeautifulSoup(host_text, 'lxml')
        text = temp.get_text()  # removing HTML formatting
        text = text.translate(str.maketrans(punct, ' '*len(punct)))
        text = ''.join([l for l in text if not l.isdigit()])
        text = re.sub(' +', ' ', text)  # striping extra white space
        text = text.strip()  # striping leading and trailing white space
        tokens = tok.tokenize(text)  # tokenizing (splitting based on whitespace)
        tokens = [token for token in tokens if (token not in stpwds) and (len(token) > 1)]
        #stemmer = FrenchStemmer()   
        #tokens_stemmed = []
        #for token in tokens:
        #    tokens_stemmed.append(stemmer.stem(token))
        #tokens = tokens_stemmed
        cleaned_data.append(tokens)
        if counter % round(len(data)/verbosity) == 0:
            print(counter, '/', len(data), 'text cleaned')
    return [' '.join(l for l in sub_cleaned_data) for 
            sub_cleaned_data in cleaned_data]


# Read training data: hosts + labels
with open("C:/Users/Asus/Desktop/Altegrad/projet/train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_data = list(set(train_data))
duplicates_to_drop = [item for item, count in 
                      collections.Counter([item.split(",")[0] for item 
                                           in train_data]).items() if count > 1]

# Remove duplicates in training data: hosts + labels
train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    if host not in duplicates_to_drop:
        train_hosts.append(host)
        y_train.append(label.lower())

# Test data
# Read test data
with open("C:/Users/Asus/Desktop/Altegrad/projet/test.csv", 'r') as f:
    test_hosts = f.read().splitlines()



# Load the textual content of a set of web-pages for each host into the dictionary "text".
# The encoding parameter is required since the majority of our text is french.
texts = dict()
file_names = os.listdir('C:/Users/Asus/Desktop/Altegrad/projet/text/text')
for filename in file_names:
    with codecs.open(os.path.join('text/text/', filename), encoding="utf8", errors='ignore') as f: 
        texts[filename] = f.read().replace("\n", "").lower()


#import json
#with open('/content/drive/My Drive/AlteGrad/dict', 'r') as file:
#    texts = json.load(file)

# Get textual content of web hosts of the training set
train_data = list()
for host in train_hosts:
    if host in texts:
        train_data.append(texts[host])
    else:
        train_data.append('')


# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«'
stpwords = stopwords.words('french')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)

# Saving cleaned_train_data
with open('cleaned_train_data_kb.pkl', 'wb') as f:
    pickle.dump(cleaned_train_data, f)
    
# Loading cleaned_train_data
#with open('/content/drive/My Drive/AlteGrad/cleaned_train_data.pkl', 'rb') as f:
#    cleaned_train_data = pickle.load(f)

test_data = list()
for host in test_hosts:
    if host in texts:
        test_data.append(texts[host])
    else:
        test_data.append('')

# Preprocessing texts
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)


# Saving cleaned_test_data
with open('cleaned_test_data_kb.pkl', 'wb') as f:
    pickle.dump(cleaned_test_data, f)
    
# Loading cleaned_test_data
# with open('/content/drive/My Drive/AlteGrad/cleaned_test_data.pkl', 'rb') as f:
#    cleaned_test_data = pickle.load(f)
    
# GB
    
clf_gb = Pipeline([('vect', TfidfVectorizer(decode_error='ignore', ngram_range=(1, 2))),
                    ('clf', GradientBoostingClassifier())])

#  ------------------------- Grid Search
 
clf_gb = clf_gb.fit(cleaned_train_data, y_train)

clf_gb.fit(X_train, Y_train)
print(clf_gb.score(X_eval, Y_eval))  


# LGR
# clf_lgr = Pipeline([('clf', LogisticRegression(solver='lbfgs',
#                                               multi_class='auto', max_iter=1000))])
w = Word2Vec(size=128, window=8, min_count=0, sg=1, workers=8)
#cleaned_train_data = cleaned_train_data.toarray()
cleaned_train_data = [k.split(' ') for k in cleaned_train_data]
cleaned_test_data = [k.split(' ') for k in cleaned_test_data]
cleaned_data = cleaned_train_data+cleaned_test_data
embeddings_words = w.build_vocab(cleaned_data)

embeddings_text_train = np.zeros((len(cleaned_train_data),128))
for i in range(len(cleaned_train_data)) :
  for k in cleaned_train_data[i] :
    embeddings_text_train[i] += w.wv[k]

embeddings_text_test = np.zeros((len(cleaned_test_data),128))
for i in range(len(cleaned_test_data)) :
  for k in cleaned_test_data[i] :
    embeddings_text_test[i] += w.wv[k]

G = nx.read_weighted_edgelist('C:/Users/Asus/Desktop/Altegrad/projet/edgelist.txt', create_using=nx.DiGraph())
n = len(train_hosts)
n_dim = 3
n_walks = 100
walk_length = 200
model = deepwalk(G,n_walks,walk_length,n_dim)


embeddings = np.zeros((G.number_of_nodes(), n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

x_train = np.zeros((len(cleaned_train_data),embeddings_text_train.shape[1]+embeddings.shape[1]))
for i in range(len(cleaned_train_data)):
    x_train[i] = np.concatenate((embeddings_text_train[i],embeddings[int(train_hosts[i])]))
    

# Evaluate the model
X_train, X_eval, Y_train, Y_eval = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42
        )


    
# clf_lgr.fit(X_train, Y_train)
# print(clf_lgr.score(X_eval, Y_eval))
# print(log_loss(Y_eval, clf_lgr.predict_proba(X_eval)))

clf_gb.fit(X_train, Y_train)
print(clf_gb.score(X_eval, Y_eval))  
print(log_loss(Y_eval, clf_gb.predict_proba(X_eval)))