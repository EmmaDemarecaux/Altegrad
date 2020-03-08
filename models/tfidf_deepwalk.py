import csv
import string
import networkx as nx
import numpy as np
import sys
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
sys.path.append('../utils')
from utils_deepwalk import deepwalk
sys.path.append('../')
from preprocess import get_train_data, import_texts, generate_data, clean_host_texts

# Generating train data without duplicates and test data
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

# TF-IDF / fitting and transforming train data
tf = TfidfVectorizer(decode_error='ignore', sublinear_tf=True, ngram_range=(1, 1),
                     min_df=0.0149, max_df=0.9, binary=False, smooth_idf=True)

x_train_or = tf.fit_transform(cleaned_train_data)
x_train_or = x_train_or.toarray()

# Reading the web domain graph and extracting the subgraph of annotated nodes
G = nx.read_weighted_edgelist(data + 'edgelist.txt', create_using=nx.DiGraph())
H = G.subgraph(train_hosts + test_hosts)
n = H.number_of_nodes()
n_hosts = len(train_hosts)

# Applying weighted version of DeepWalk algorithm on the subgraph H
n_dim = 128
n_walks = 500
walk_length = 100
model = deepwalk(H, n_walks, walk_length, n_dim)

# Nodes embeddings (we set the embedding of nodes not belonging to H to zeros since they will not be used)
embeddings = np.zeros((G.number_of_nodes(), n_dim))
for i, node in enumerate(H.nodes()):
    embeddings[int(node), :] = model.wv[str(node)]

# Training data features : mix of TF-IDF and nodes embeddings
x_train = np.zeros((x_train_or.shape[0], x_train_or.shape[1]+n_dim))
for i in range(x_train_or.shape[0]):
    x_train[i] = np.concatenate((x_train_or[i], embeddings[int(train_hosts[i])]))

# Logistic Regression Model
clf_lgr = Pipeline([('clf', LogisticRegression(tol=1e-05, C=4.59))])
clf_lgr.fit(x_train, y_train)

# Test data : TF-IDF transformation + creating the test features (mix of TF-IDF features and embeddings)
x_test_or = tf.transform(cleaned_test_data) 
x_test_or = x_test_or.toarray() 
x_test = np.zeros((x_test_or.shape[0], x_test_or.shape[1]+n_dim))
for i in range(x_test_or.shape[0]):
    x_test[i] = np.concatenate((x_test_or[i], embeddings[int(test_hosts[i])]))
y_pred = clf_lgr.predict_proba(x_test)

# Writing predictions to a file
with open('../tfidf_deepwalk.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = clf_lgr.classes_.tolist()
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = y_pred[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
