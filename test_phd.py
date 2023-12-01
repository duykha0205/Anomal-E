import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing
from dgl.data import DGLDataset
import dgl
import time
import networkx as nx
import category_encoders as ce
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch
import tqdm
import math

from typing import *
from sklearn.preprocessing import StandardScaler, Normalizer
import socket
import struct
import random
from sklearn.model_selection import train_test_split

file_name = "/kaggle/input/nf-unsw-nb15-v2csv/NF-UNSW-NB15-v2.csv"
data = pd.read_csv(file_name)

data.rename(columns=lambda x: x.strip(), inplace=True)
data['IPV4_SRC_ADDR'] = data["IPV4_SRC_ADDR"].apply(str)
data['L4_SRC_PORT'] = data["L4_SRC_PORT"].apply(str)
data['IPV4_DST_ADDR'] = data["IPV4_DST_ADDR"].apply(str)
data['L4_DST_PORT'] = data["L4_DST_PORT"].apply(str)
data.drop(columns=["L4_SRC_PORT", "L4_DST_PORT"], inplace=True)
data = data.groupby(by='Attack').sample(frac=0.1, random_state=13)

X = data.drop(columns=["Attack", "Label"])
y = data[["Attack", "Label"]]

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=13, stratify=y)

encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL',
                                  'CLIENT_TCP_FLAGS','SERVER_TCP_FLAGS','ICMP_TYPE',
                                  'ICMP_IPV4_TYPE','DNS_QUERY_ID','DNS_QUERY_TYPE',
                                  'FTP_COMMAND_RET_CODE'])
encoder.fit(X_train, y_train.Label)

# Transform on training set
X_train = encoder.transform(X_train)

# Transform on testing set
X_test = encoder.transform(X_test)

X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

scaler = Normalizer()
cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns))) # Ignore first two as the represents IP addresses
scaler.fit(X_train[cols_to_norm])

# Transform on training set
X_train[cols_to_norm] = scaler.transform(X_train[cols_to_norm])
X_train['h'] = X_train.iloc[:, 2:].values.tolist()

# Transform on testing set
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test.iloc[:, 2:].values.tolist()

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

lab_enc = preprocessing.LabelEncoder()
lab_enc.fit(data["Attack"])

# Transform on training set
train["Attack"] = lab_enc.transform(train["Attack"])

# Transform on testing set
test["Attack"] = lab_enc.transform(test["Attack"])

# Training graph

train_g = nx.from_pandas_edgelist(train, "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
            ["h", "Label", "Attack"], create_using=nx.MultiGraph())

train_g = train_g.to_directed()
train_g = dgl.from_networkx(train_g, edge_attrs=['h', 'Attack', 'Label'])
nfeat_weight = torch.ones([train_g.number_of_nodes(),
train_g.edata['h'].shape[1]])
train_g.ndata['h'] = nfeat_weight

# Testing graph
test_g = nx.from_pandas_edgelist(test, "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
            ["h", "Label", "Attack"], create_using=nx.MultiGraph())

test_g = test_g.to_directed()
test_g = dgl.from_networkx(test_g, edge_attrs=['h', 'Attack', 'Label'])
nfeat_weight = torch.ones([test_g.number_of_nodes(),
test_g.edata['h'].shape[1]])
test_g.ndata['h'] = nfeat_weight

import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import tqdm
import gc

class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
      super(SAGELayer, self).__init__()
      self.W_apply = nn.Linear(ndim_in + edims , ndim_out)
      self.activation = F.relu
      self.W_edge = nn.Linear(128 * 2, 256)
      self.reset_parameters()

    def reset_parameters(self):
      gain = nn.init.calculate_gain('relu')
      nn.init.xavier_uniform_(self.W_apply.weight, gain=gain)

    def message_func(self, edges):
      return {'m':  edges.data['h']}

    def forward(self, g_dgl, nfeats, efeats):
      with g_dgl.local_scope():
        g = g_dgl
        g.ndata['h'] = nfeats
        g.edata['h'] = efeats
        g.update_all(self.message_func, fn.mean('m', 'h_neigh'))
        g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))

        # Compute edge embeddings
        u, v = g.edges()
        edge = self.W_edge(torch.cat((g.srcdata['h'][u], g.dstdata['h'][v]), 2))
        return g.ndata['h'], edge
    
class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim,  activation):
      super(SAGE, self).__init__()
      self.layers = nn.ModuleList()
      self.layers.append(SAGELayer(ndim_in, edim, 128, F.relu))

    def forward(self, g, nfeats, efeats, corrupt=False):
      if corrupt:
        e_perm = torch.randperm(g.number_of_edges())
        #n_perm = torch.randperm(g.number_of_nodes())
        efeats = efeats[e_perm]
        #nfeats = nfeats[n_perm]
      for i, layer in enumerate(self.layers):
        #nfeats = layer(g, nfeats, efeats)
        nfeats, e_feats = layer(g, nfeats, efeats)
      #return nfeats.sum(1)
      return nfeats.sum(1), e_feats.sum(1)
    
class Discriminator(nn.Module):
    def __init__(self, n_hidden):
      super(Discriminator, self).__init__()
      self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
      self.reset_parameters()

    def uniform(self, size, tensor):
      bound = 1.0 / math.sqrt(size)
      if tensor is not None:
        tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
      size = self.weight.size(0)
      self.uniform(size, self.weight)

    def forward(self, features, summary):
      features = torch.matmul(features, torch.matmul(self.weight, summary))
      return features

class DGI(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation):
      super(DGI, self).__init__()
      self.encoder = SAGE(ndim_in, ndim_out, edim,  F.relu)
      #self.discriminator = Discriminator(128)
      self.discriminator = Discriminator(256)
      self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False)
      negative = self.encoder(g, n_features, e_features, corrupt=True)
      self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, n_features, e_features):
      positive = self.encoder(g, n_features, e_features, corrupt=False)
      negative = self.encoder(g, n_features, e_features, corrupt=True)

      positive = positive[1]
      negative = negative[1]

      summary = torch.sigmoid(positive.mean(dim=0))

      positive = self.discriminator(positive, summary)
      negative = self.discriminator(negative, summary)

      l1 = self.loss(positive, torch.ones_like(positive))
      l2 = self.loss(negative, torch.zeros_like(negative))

      return l1 + l2

ndim_in = train_g.ndata['h'].shape[1]
hidden_features = 128
ndim_out = 128
num_layers = 1
edim = train_g.edata['h'].shape[1]
learning_rate = 1e-3
epochs = 1000

dgi = DGI(ndim_in,
    ndim_out,
    edim,
    F.relu)

dgi_optimizer = torch.optim.Adam(dgi.parameters(),
                lr=1e-3,
                weight_decay=0.)

# Format node and edge features for E-GraphSAGE
train_g.ndata['h'] = torch.reshape(train_g.ndata['h'],
                                   (train_g.ndata['h'].shape[0], 1,
                                    train_g.ndata['h'].shape[1]))

train_g.edata['h'] = torch.reshape(train_g.edata['h'],
                                   (train_g.edata['h'].shape[0], 1,
                                    train_g.edata['h'].shape[1]))



# Convert to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_g = train_g.to(device=device)
dgi = dgi.to(device=device)

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP


cnt_wait = 0
best = 1e9
best_t = 0
dur = []
node_features = train_g.ndata['h']
edge_features = train_g.edata['h']

def demo_basic():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29444'
    # os.environ['RANK'] = '0'
    # os.environ['WORLD_SIZE'] = '2' 
    dist.init_process_group("nccl")
    rank = dist.get_rank()
#     rank = 0
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    ddp_dgi = DDP(dgi, device_ids=[device_id])
    dgi_optimizer = torch.optim.Adam(ddp_dgi.parameters(),
                lr=1e-3,
                weight_decay=0.)

    for epoch in range(epochs):
        ddp_dgi.train()
        if epoch >= 3:
            t0 = time.time()

        dgi_optimizer.zero_grad()
        loss = ddp_dgi(train_g, node_features, edge_features)
        loss.backward()
        dgi_optimizer.step()

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(dgi.state_dict(), 'best_dgi.pkl')
        else:
            cnt_wait += 1

        if epoch >= 3:
            dur.append(time.time() - t0)

        if epoch % 50 == 0:

            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur),
                  loss.item(),
                  train_g.num_edges() / np.mean(dur) / 1000))

    dist.destroy_process_group()
    
demo_basic()