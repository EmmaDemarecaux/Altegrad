"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import numpy as np
import networkx as nx
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from scipy.sparse import csr_matrix
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import codecs
import string
from nltk.tokenize import TweetTokenizer
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer
import pickle
import collections
import torch.nn as nn
import csv


def normalize_adjacency(a):
    a_tilde = a + np.eye(a.shape[0])
    d = np.sum(a_tilde, axis=1)
    d = d.squeeze().tolist()[0]
    d = np.power(d, -1/2)
    d_tilde = np.diag(d)
    a_normalized = np.dot(np.dot(d_tilde, a_tilde), d_tilde)
    return a_normalized


def accuracy(output, labels):
    """Computes classification accuracy"""
    predictions = output.max(1)[1].type_as(labels)
    correct = predictions.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.logprob = nn.LogSoftmax(dim=1)

    def forward(self, x_in, adj):
        # message passing layer with h1 hidden units
        x1 = self.fc1(x_in)
        x1 = torch.mm(adj, x1)
        a1 = self.relu(x1)
        # dropout layer
        a1 = self.dropout(a1)
        # message passing layer with h2 hidden units
        x2 = self.fc2(a1)
        x2 = torch.mm(adj, x2)
        a2 = self.relu(x2)
        # fully-connected layer with nclass units
        x = self.fc3(a2)
        return self.logprob(x)
    

def train(nb_epochs, idx_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    predictions = model(features, adj)
    loss_train = F.nll_loss(predictions[idx_train], y[idx_train])
    acc_train = accuracy(predictions[idx_train], y[idx_train])
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:03d}'.format(nb_epochs+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    
def test(idx_test):
    model.eval()
    predictions = model(features, adj)
    loss_test = F.nll_loss(predictions[idx_test], y[idx_test])
    acc_test = accuracy(predictions[idx_test], y[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


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
        stemmer = FrenchStemmer()
        tokens_stemmed = []
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
        cleaned_data.append(tokens)
        if counter % round(len(data)/verbosity) == 0:
            print(counter, '/', len(data), 'text cleaned')
    return [' '.join(l for l in sub_cleaned_data) for
            sub_cleaned_data in cleaned_data]


# Read training data: hosts + labels
with open("train.csv", 'r') as f:
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

# Load the textual content of a set of web-pages for each host into the dictionary "text".
# The encoding parameter is required since the majority of our text is french.
texts = dict()
file_names = os.listdir('text/text')
for filename in file_names:
    with codecs.open(os.path.join('text/text/', filename), encoding="utf8", errors='ignore') as f:
        texts[filename] = f.read().replace("\n", "").lower()

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
# cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)

# # Saving cleaned_train_data
# with open('cleaned_train_data.pkl', 'wb') as f:
#     pickle.dump(cleaned_train_data, f)

# Loading cleaned_train_data
with open('cleaned_train_data.pkl', 'rb') as f:
    cleaned_train_data = pickle.load(f)

# Processing labels
n_class = len(set(y_train))
class_dict = dict([(j, i) for (i, j) in enumerate(set(y_train))])
y = np.array([class_dict[i] for i in y_train])

# Test data
# Read test data
with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    
# Get textual content of web hosts of the test set
test_data = list()
for host in test_hosts:
    if host in texts:
        test_data.append(texts[host])
    else:
        test_data.append('')

# Preprocessing texts
# cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)
#
# Saving cleaned_test_data
# with open('cleaned_test_data.pkl', 'wb') as f:
#    pickle.dump(cleaned_test_data, f)
    
# Loading cleaned_test_data
with open('cleaned_test_data.pkl', 'rb') as f:
    cleaned_test_data = pickle.load(f)

# Create a directed, weighted graph
G = nx.read_weighted_edgelist('edgelist.txt', create_using=nx.DiGraph())
H = G.subgraph(train_hosts + test_hosts)
n = H.number_of_nodes()
n_hosts = len(train_hosts)
print('number of nodes = ', n)
print('nnmber of training hosts = ', n_hosts)
print('number of edges = ', H.number_of_edges())

# Adjacency matrix
adj = nx.to_numpy_matrix(H, nodelist=train_hosts + test_hosts)  # Obtains the adjacency matrix
adj = normalize_adjacency(adj)  # Normalizes the adjacency matrix

# GNN hyperparameters
epochs = 200
n_hidden_1 = 100
n_hidden_2 = 200
learning_rate = 0.05
weight_decay = 1e-2
dropout_rate = 0.4

# Node embeddings
vect = TfidfVectorizer(decode_error='ignore', min_df=0.1, max_df=0.8)
X_embed = vect.fit_transform(cleaned_train_data + cleaned_test_data)

# Set the feature of all nodes
# features_matrix = np.eye(n) # Generates node features
features_matrix = csr_matrix.toarray(X_embed)

# Yields indices to split data into training and test sets
idx = np.random.RandomState(seed=42).permutation(n_hosts)
index_train = idx[:int(0.8*n_hosts)]
index_test = idx[int(0.8*n_hosts):]

# Transforms the numpy matrices/vectors to torch tensors
features = torch.FloatTensor(features_matrix)
y = torch.LongTensor(y)
adj = torch.FloatTensor(adj)
index_train = torch.LongTensor(index_train)
index_test = torch.LongTensor(index_test)

# Creates the model and specifies the optimizer
model = GNN(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch, index_train)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print()

# Testing
test(index_test)

# Train model on all the training data
index_train_all = np.array(range(n_hosts))
index_train_all = torch.LongTensor(index_train_all)

model = GNN(features.shape[1], n_hidden_1, n_hidden_2, n_class, dropout_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Train model
t_total = time.time()
for epoch in range(epochs):
    train(epoch, index_train_all)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
print()
model.eval()
preds = model(features, adj)
preds = preds.exp().detach().numpy()
result = preds[n_hosts:,:]

# Write predictions to a file
with open('graph_gnn.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = list(class_dict.keys())
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = preds[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
