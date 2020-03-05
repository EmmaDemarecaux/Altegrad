import networkx as nx
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import csv
import string
import numpy as np
from scipy.sparse import csr_matrix
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import remove_duplicates, import_texts, generate_data, clean_host_texts


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
    def __init__(self, n_feat, nh_1, nh_2, nc, dropout):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(n_feat, nh_1)
        self.fc2 = nn.Linear(nh_1, nh_2)
        self.fc3 = nn.Linear(nh_2, nc)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.logprob = nn.LogSoftmax(dim=1)

    def forward(self, x_in, adja):
        # message passing layer with h1 hidden units
        x1 = self.fc1(x_in)
        x1 = torch.mm(adja, x1)
        a1 = self.relu(x1)
        # dropout layer
        a1 = self.dropout(a1)
        # message passing layer with h2 hidden units
        x2 = self.fc2(a1)
        x2 = torch.mm(adja, x2)
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
    

train_file = 'train.csv'
train_hosts, y_train = remove_duplicates(train_file)
texts_path = 'text/text'
texts = import_texts(texts_path)

with open("test.csv", 'r') as f:
    test_hosts = f.read().splitlines()
    
train_data = generate_data(train_hosts, texts)
test_data = generate_data(test_hosts, texts)

# Preprocessing texts
tokenizer = TweetTokenizer()
punctuation = string.punctuation + '’“”.»«…'
stpwords = stopwords.words('french')
cleaned_train_data = clean_host_texts(data=train_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)
cleaned_test_data = clean_host_texts(data=test_data, tok=tokenizer, stpwds=stpwords, punct=punctuation)

# Processing labels
n_class = len(set(y_train))
class_dict = dict([(j, i) for (i, j) in enumerate(set(y_train))])
y = np.array([class_dict[i] for i in y_train])

# Create a directed, weighted graph
G = nx.read_weighted_edgelist('edgelist.txt', create_using=nx.DiGraph())
H = G.subgraph(train_hosts + test_hosts)
n = H.number_of_nodes()
n_hosts = len(train_hosts)
print('Number of nodes = ', n)
print('Number of training hosts = ', n_hosts)
print('Number of edges = ', H.number_of_edges())

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
vect = TfidfVectorizer(decode_error='ignore', sublinear_tf=True,
                       min_df=0.06, max_df=0.7, smooth_idf=True)
# vect = TfidfVectorizer(decode_error='ignore', min_df=0.1, max_df=0.8)
X_embed = vect.fit_transform(cleaned_train_data + cleaned_test_data)

# Set the feature of all nodes
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
result = preds[n_hosts:, :]

# Write predictions to a file
with open('tfidf_gnn.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    lst = list(class_dict.keys())
    lst.insert(0, "Host")
    writer.writerow(lst)
    for i, test_host in enumerate(test_hosts):
        lst = preds[i, :].tolist()
        lst.insert(0, test_host)
        writer.writerow(lst)
