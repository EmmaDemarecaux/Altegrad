# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:59:03 2020

@author: Aicha BOUJANDAR
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score



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


with open("train.csv", 'r') as f:
    train_data = f.read().splitlines()

train_hosts = list()
y_train = list()
for row in train_data:
    host, label = row.split(",")
    train_hosts.append(host)
    y_train.append(label.lower())
y_train = np.array(y_train)

with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    

G = nx.read_weighted_edgelist('edgelist.txt', create_using=nx.DiGraph())


n = len(train_hosts)
n_dim = 128
n_walks = 100
walk_length = 20
model = deepwalk(G,n_walks,walk_length,n_dim)


embeddings = np.zeros((G.number_of_nodes(), n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

Y_train = y_train[idx_train]
Y_test = y_train[idx_test]





LG = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=2000)
LG.fit(X_train,Y_train)
Y_pred = LG.predict(X_test)
LG.score(X_test,Y_test)




