# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:59:03 2020

@author: Aicha BOUJANDAR
"""

import numpy as np
import networkx as nx
from random import randint
from gensim.models import Word2Vec




def random_walk(G, node, walk_length):


    walk = [node]
    for i in range(walk_length):
      neighbors = list(G.neighbors(walk[-1]))
      if len(neighbors)>0 :
        weights = list()
        for neigh in neighbors :
            weights.append(G[walk[-1]][neigh]['weight'])
        weights = np.array(weights)
        weights = weights/np.sum(weights)
        indice = np.random.choice(np.arange(0,len(neighbors)),size=1,p=weights)[0]
        walk.append(neighbors[indice])
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


if __name__== '__main__' :

    G = nx.read_weighted_edgelist('edgelist.txt', create_using=nx.DiGraph())



    n_dim = 10
    n_walks = 10
    walk_length = 20
    model = deepwalk(G,n_walks,walk_length,n_dim)


    embeddings = np.zeros((G.number_of_nodes(), n_dim))
    for i, node in enumerate(G.nodes()):
        embeddings[i,:] = model.wv[str(node)]


