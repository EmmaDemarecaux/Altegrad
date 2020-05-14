import numpy as np
import torch
import torch.nn as nn


def normalize_adjacency(a):
    """
    Given the adjacency matrix A of a graph, we normalize it as follows: A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2).
    A_tilde = A + I; D_tilde is a diagonal matrix such taht D_tilde(ii) = sum_j A_tilde(ij)
    The formula adds self-loops to the graph, and normalizes each row of the emerging matrix
    such that the sum of its element is equal to 1.
    Inputs :
        a : the adjacency matrix of the graph
    Output :
        walk : The generated walk
    """
    a_tilde = a + np.eye(a.shape[0])
    d = np.sum(a_tilde, axis=1)
    d = d.squeeze().tolist()[0]
    d = np.power(d, -1 / 2)
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
