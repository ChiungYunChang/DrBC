from torch_geometric.nn import GCNConv
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
from networkx.algorithms.centrality.degree_alg import out_degree_centrality
import time
from torch.optim import Adam
import torch.nn.functional as F
import wandb
from networkx.algorithms.centrality import betweenness
# from google.colab import drive
import pandas as pd
from pprint import pprint
import numpy as np
import networkx as nx
import os
import scipy.stats as stats  # Kendall tau
import random
import math



batch_size = 5



class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        #         # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


#### Step 2 : Implement DrBC"""


class Encoder(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=128):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.relu = nn.LeakyReLU()

        self.gcn1 = GCNConv(128, 128)  # message passing
        self.gru1 = nn.GRU(128, 128)

        self.gcn2 = GCNConv(128, 128)
        self.gru2 = nn.GRU(128, 128)

        self.gcn3 = GCNConv(128, 128)
        self.gru3 = nn.GRU(128, 128)

    def forward(self, x, edge_index):
        outs = []
        x = self.fc(x)
        x = self.relu(x)
        gcnx = self.gcn1(x, edge_index)
        x1, _ = self.gru1(gcnx.view(1, *gcnx.shape), x.view(1, *x.shape))
        gcnx = self.gcn2(x1[0], edge_index)
        x2, _ = self.gru2(gcnx.view(1, *gcnx.shape), x1)
        gcnx = self.gcn3(x2[0], edge_index)
        x3, _ = self.gru3(gcnx.view(1, *gcnx.shape), x2)

        # Layer Aggregator : Max Pooling
        outs = torch.stack([x1[0], x2[0], x3[0]])
        max_outs = torch.max(outs, dim=0).values
        return max_outs


class Decoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32):
        super(Decoder, self).__init__()
        # Decoder
        self.fc1 = nn.Linear(128, hidden_dim)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class DrBC_model(nn.Module):
    def __init__(self, input_dim=3, embedding_dim=128):
        super(DrBC_model, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim)
        self.decoder = Decoder(input_dim, embedding_dim)

    def forward(self, x, edge_index):
        en_x = self.encoder(x, edge_index)
        outs = self.decoder(en_x)
        return outs

def loss_function(out, bc_value, source_ids, target_ids):
    pred = out[source_ids] - out[target_ids]
    gt = torch.sigmoid((bc_value[source_ids] - bc_value[target_ids]))
    gt = gt.view(-1, 1)
    loss = F.binary_cross_entropy_with_logits(pred, gt, reduction="sum")
    return loss


class test_data():
    def __init__(self):
        super(test_data, self).__init__()
        self.TestGraph_feature = []
        self.data = []
        self.edge_list = []
        self.BC = []
        self.TestGraph_edge = [[], []]  # TestGraph_edge[0]=source,[1]= target

    def readfile(self, path):
        data = []
        file = open(path, "r")
        for line in file.readlines():
            data.append(line.split())
        file.close()
        return data

    def get_num_node(self, score_path):
        num_node = 0
        # calculate node number
        for i in score_path:
            num_node += 1
        print(num_node)
        node_degree = [0] * num_node
        return node_degree

    def get_edge_node_index(self, edge_index, node_degree):
        # initial node degree
        start_edge = [];
        end_edge = []
        print(np.shape(edge_index))
        for [s, t] in edge_index:
            # get node degree
            node_degree[int(s)] += 1
            node_degree[int(t)] += 1
            # bidirectional edge
            start_edge += [int(s)]
            end_edge += [int(t)]
        # node extension : bidirectional edge
        for nb in node_degree:
            self.TestGraph_feature.append([nb, 1, 1])
        self.TestGraph_edge[0].extend(start_edge)
        self.TestGraph_edge[1].extend(end_edge)
        self.TestGraph_edge[0].extend(end_edge)
        self.TestGraph_edge[1].extend(start_edge)
        return self.TestGraph_edge, self.TestGraph_feature

    # betweenness centrality value
    def get_BC(self, file_path):

        gt = self.readfile(file_path)
        print("BC",np.shape(gt))
        for (node_id, bc) in gt:
            bc = -math.log(float(bc) + 1e-8)
            self.BC.append([bc])
        return self.BC

    def topN_accuracy(self, outs, label):
        topk = [1, 5, 10]
        k_accuracy = []
        node_nums = len(outs)

        label = label.reshape(-1)
        outs = outs.reshape(-1)
        tau, _ = stats.kendalltau(outs.cpu().detach().numpy(), label.cpu().detach().numpy())
        print("Kendall tau: {}".format(tau))
        label = torch.argsort(label)
        outs = torch.argsort(outs)

        for k in topk:
            k_num = int(node_nums * k / 100)
            k_label = label[:k_num].tolist()
            k_outs = outs[:k_num].tolist()

            correct = list(set(k_label) & set(k_outs))
            accuracy = len(correct) / (k_num)
            k_accuracy.append(accuracy)
            print("Top-{} accuracy: {}".format(k, accuracy * 100))

        return k_accuracy,tau



def testing(PATH, Ground_truthPath):
    testing = test_data()
    model = DrBC_model()
    model.load_state_dict(torch.load('model/500800.pth'))
    # predictbc_value = torch.tensor(testing.get_BC(PATH))
    ground_trueBC = torch.tensor(testing.get_BC(Ground_truthPath))
    # print(ground_trueBC)
    model.cuda()
    model.eval()
    with torch.no_grad():
        # ground truth
        dataset = testing.readfile(PATH)
        dataset_score = testing.readfile(Ground_truthPath)
        node_degree_data = testing.get_num_node(dataset_score)
        TestGraph_edge_index, TestGraph_feature = testing.get_edge_node_index(dataset, node_degree_data)
        print("DATA DONE")
        # model = model.cpu()
        outs = model(torch.FloatTensor(TestGraph_feature).cuda(), torch.tensor(TestGraph_edge_index).cuda())
        k_accuracy ,tau= testing.topN_accuracy(outs, ground_trueBC )
    return k_accuracy,tau

# file path
yt_node_pair_file = './youtube/com-youtube.txt'
yt_bc_score_file = './youtube/com-youtube_score.txt'
Synthetic_graph_root = "Synthetic/5000/"
Synthetic_score = "Synthetic/5000/"
# youtube graph
# print(testing(yt_node_pair_file, yt_bc_score_file))
# Synthetic graph
ACC = []
k =[]
for id in range(30):
    Synthetic_graph = Synthetic_graph_root + f'{id}.txt'
    Synthetic_score = Synthetic_graph_root + f'{id}_score.txt'
    k_accuracy,tau= testing(Synthetic_graph, Synthetic_score)
    ACC.append(k_accuracy)
    k.append(tau)
ACC = np.array(ACC)
average = np.mean(ACC,axis=0)
k= np.mean(k)
print(average,"TTTT", "k",k)



