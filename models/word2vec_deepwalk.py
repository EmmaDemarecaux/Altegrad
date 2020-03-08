import numpy as np
import networkx as nx
import string
import sys
import csv
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
sys.path.append('../utils')
from utils_deepwalk import deepwalk
sys.path.append('../')
from preprocess import get_train_data, import_texts, generate_data, clean_host_texts

# Generating train and test data
data = '../data/'
train_file = data + 'train_noduplicates.csv'
train_hosts, y_train = get_train_data(train_file)
texts_path = '../text/text'
texts = import_texts(texts_path)

with open(data + 'test.csv', 'r') as f:
    test_hosts = f.read().splitlines()
    
train_data = generate_data(train_hosts, texts)
test_data = generate_data(test_hosts, texts)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…°'
stpwords_fr = stopwords.words('french')
stpwords_en = stopwords.words('english')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer,
                                      stpwds=stpwords_fr + stpwords_en, punct=punctuation)
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer,
                                     stpwds=stpwords_fr + stpwords_en, punct=punctuation)

# Applying the word2vec model on the train and the test data
w = Word2Vec(size=128, window=8, min_count=0, sg=1, workers=8)
cleaned_train_data = [k.split(' ') for k in cleaned_train_data]
cleaned_test_data = [k.split(' ') for k in cleaned_test_data]
cleaned_data = cleaned_train_data+cleaned_test_data
w.build_vocab(cleaned_data)
w.train(cleaned_data, total_examples=w.corpus_count, epochs=5)

# Defining the embedding of each document of the train data as the sum of its words embeddings
embeddings_text_train = np.zeros((len(cleaned_train_data), 128))
for i in range(len(cleaned_train_data)):
    for k in cleaned_train_data[i]:
        embeddings_text_train[i] += w.wv[k]

# Defining the embedding of each document of the test data as the sum of its words embeddings
embeddings_text_test = np.zeros((len(cleaned_test_data), 128))
for i in range(len(cleaned_test_data)):
    for k in cleaned_test_data[i]:
        embeddings_text_test[i] += w.wv[k]

# Reading the web domain graph and extract the subgraph of annotated nodes
G = nx.read_weighted_edgelist(data + 'edgelist.txt', create_using=nx.DiGraph())
H = G.subgraph(train_hosts + test_hosts)
n = H.number_of_nodes()
n_hosts = len(train_hosts)

# Applying weighted version of DeepWalk algorithm on the subgraph H
n_dim = 3
n_walks = 100
walk_length = 200
model = deepwalk(H, n_walks, walk_length, n_dim)

# Nodes embeddings (we set the embedding of nodes not belonging to H to zeros since they will not be used)
embeddings = np.zeros((G.number_of_nodes(), n_dim))
for i, node in enumerate(H.nodes()):
    embeddings[int(node), :] = model.wv[str(node)]

# Train data features : mix of documents embedding and nodes embeddings
x_train = np.zeros((len(cleaned_train_data), embeddings_text_train.shape[1]+embeddings.shape[1]))
for i in range(len(cleaned_train_data)):
    x_train[i] = np.concatenate((embeddings_text_train[i], embeddings[int(train_hosts[i])]))

# Test data features : mix of documents embedding and nodes embeddings
x_test = np.zeros((len(cleaned_test_data), embeddings_text_test.shape[1]+embeddings.shape[1]))
for i in range(len(cleaned_test_data)):
    x_test[i] = np.concatenate((embeddings_text_test[i], embeddings[int(test_hosts[i])]))
    
# Logistic Regression Model
clf_lgr = Pipeline([('clf', LogisticRegression(tol=1e-05, C=4.59))])
clf_lgr.fit(x_train, y_train)
y_pred = clf_lgr.predict_proba(x_test)

# Writing predictions to a file
with open('../word2vec_deepwalk.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = clf_lgr.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
