import numpy as np
import networkx as nx
from gensim.models import Word2Vec


#This function is a weighted version of the function random_walk defined in lab5
def random_walk(g, node_, n):
    '''
    Inputs :
        g : the graph, nodes_ : the node from which the walk will be generated, n : length of the walk
        
    Output :
        walk : the generated walk
    '''
    walk = [node_]
    for _ in range(n):
        neighbors = list(g.neighbors(walk[-1]))
        if len(neighbors) > 0:
            weights = list()
            for neigh in neighbors:
                weights.append(g[walk[-1]][neigh]['weight'])
            weights = np.array(weights)
            weights = weights/np.sum(weights)
            indices = np.random.choice(np.arange(0, len(neighbors)), size=1, p=weights)[0]
            walk.append(neighbors[indices])
        else:
            break
    walk = [str(node_) for node_ in walk]
    return walk


#This function was defined in the lab5 of the course
def generate_walks(g, num_walks, n):
    '''
    Inputs :
        g : the graph, num_walks : number of walks to be generated, n : length of the walks
    Output :
        the generated walks
    '''
    walks = []
    for _ in range(num_walks):
        for node_ in g.nodes():
            walks.append(random_walk(g, node_, n))
    return walks

#This function was defined in the lab5 of the course
def deepwalk(g, num_walks, n, size):
    '''
    Inputs : 
        g : the graph, num_walks : number of walks to be generated
        n : length of the walks, size : the embedding dimension
    Output :
        The nodes embedding model
    '''
    print("Generating walks")
    walks = generate_walks(g, num_walks, n)
    print("Training word2vec")
    m = Word2Vec(size=size, window=8, min_count=0, sg=1, workers=8)
    m.build_vocab(walks)
    m.train(walks, total_examples=m.corpus_count, epochs=5)
    return m

