import numpy as np
import networkx as nx
from gensim.models import Word2Vec


def random_walk(g, node_, n):
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


def generate_walks(g, num_walks, n):
    walks = []
    for _ in range(num_walks):
        for node_ in g.nodes():
            walks.append(random_walk(g, node_, n))
    return walks


def deepwalk(g, num_walks, n, size):
    print("Generating walks")
    walks = generate_walks(g, num_walks, n)
    print("Training word2vec")
    m = Word2Vec(size=size, window=8, min_count=0, sg=1, workers=8)
    m.build_vocab(walks)
    m.train(walks, total_examples=m.corpus_count, epochs=5)
    return m


if __name__ == '__main__':
    G = nx.read_weighted_edgelist('../data/edgelist.txt', create_using=nx.DiGraph())
    n_dim = 10
    n_walks = 10
    walk_length = 20
    model = deepwalk(G, n_walks, walk_length, n_dim)
    embeddings = np.zeros((G.number_of_nodes(), n_dim))
    for i, node in enumerate(G.nodes()):
        embeddings[i, :] = model.wv[str(node)]
