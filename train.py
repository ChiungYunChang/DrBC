print('yyy')
from torch_geometric.nn import GCNConv
import networkx as nx

batch_size = 5
import wandb
import random
import math
import numpy as np


class Graph():
    def __init__(self, batch_size):
        self.graph_data = []
        # generate the graph with power-law distribution
        for i in range(batch_size):
            G = nx.random_graphs.powerlaw_cluster_graph(random.randint(800, 1000), 5, 0.05)
            self.graph_data.append(G)

    # calculate node degree
    def node_degree(self):
        degree_list = []
        for g in self.graph_data:
            for n in range(g.number_of_nodes()):
                degree_list.append([g.degree[n], 1, 1])
        return torch.Tensor(degree_list)

    # calculate edge index
    def get_edge_index(self):
        start_node, end_node, node_number = [], [], 0
        for graph in self.graph_data:
            for edge in graph.edges():
                start, end = edge
                start_node.append(start + node_number)
                end_node.append(end + node_number)
            node_number += graph.number_of_nodes()
        # bidirection edge
        edge_index = [start_node + end_node, end_node + start_node]
        return torch.LongTensor(edge_index)

        # calculate betweeness centrality

    def calculate_bc(self):
        bc_list = [list(nx.betweenness_centrality(graph).values()) for graph in self.graph_data]
        labels = []
        for bc in bc_list:
            labels.extend(bc)
        log_labels = [-math.log(v + 1e-8) for v in labels]
        return torch.Tensor(log_labels)

    # Using random node pairs to compute the loss
    def get_pairs(self, repeat=5):
        id_nums = 0;
        source_ids = [];
        target_ids = []
        for graph in self.graph_data:
            node_nums = len(graph.nodes)
            source_id = [i for i in range(id_nums, node_nums + id_nums)]
            target_id = [i for i in range(id_nums, node_nums + id_nums)]
            # sample 5|V| source nodes and 5|V| target nodes 5|V|
            source_id *= 5
            target_id *= 5
            random.shuffle(source_id)
            random.shuffle(target_id)
            source_ids.extend(source_id)
            target_ids.extend(target_id)
        return source_ids, target_ids



from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

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


"""#### Step 2 : Implement DrBC"""

from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch


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


from networkx.algorithms.centrality.degree_alg import out_degree_centrality
import time
from torch.optim import Adam
import torch.nn.functional as F
import wandb


def train():
    model = DrBC_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Define optimizer.
    eval_graph = Graph(batch_size=2)
    evsl_bc_value = eval_graph.calculate_bc()
    min_valid_loss = np.inf
    iternum = 10000
    wandb.init(project="DrBC-yun-2")
    wandb.run.config.name = 'p=0.5'
    wandb.config.lr = 0.0001
    wandb.config.emdeddindDimen = 128
    wandb.config.iterarion = 10000
    wandb.config.node = "1000-800"
    wandb.config.edge= 8
    wandb.config.p= 0.05
    for iter in range(iternum):
        optimizer.zero_grad()
        model.train()
        if iter % 200 == 0:
            graph = Graph(10)
            bc_value = graph.calculate_bc()
        out = model(graph.node_degree().cuda(), graph.get_edge_index().cuda())
        source_ids, target_ids = graph.get_pairs()
        loss = loss_function(out, bc_value.cuda(), source_ids, target_ids)
        wandb.log({'pair_loss': loss, 'iter': iter})  # record trsining loss
        loss.backward()
        optimizer.step()

        # validation
        if iter % 100 == 0:
            model.eval()
            with torch.no_grad():
                out = model(eval_graph.node_degree().cuda(), eval_graph.get_edge_index().cuda())
            source_ids, target_ids = eval_graph.get_pairs()
            vali_loss = loss_function(out, evsl_bc_value.cuda(), source_ids, target_ids)
            print(f' Validation Loss: {vali_loss}')
            wandb.log({'validation_loss': vali_loss, 'iter': iter})
            if min_valid_loss > vali_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{vali_loss:.6f}) \t Saving The Model')
                min_valid_loss = vali_loss
                # Saving State Dict
                torch.save(model.state_dict(), 'saved_model.pth')
            print("[{}/{}] Loss:{:.4f}".format(iter, iternum, loss.item()))
    return model


train()

import numpy as np
import networkx as nx
import os
import scipy.stats as stats  # Kendall tau
import random
import math


# connect to google drive
# drive.mount('/content/drive')

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
        for (node_id, bc) in gt:
            bc = -math.log(float(bc) + 1e-8)
            self.BC.append([bc])
        return self.BC

    # Top-N % accuracy
    def takeSecond(self, elem):
        return elem[1]

    def topN_accuracy(self, file, outs, n, bc_value):
        predict_value, bc_value = [], []
        for i, j in enumerate(outs.tolist()):
            predict_value.append([i, *j])
        bc_value.sort(key=self.takeSecond, reverse=True)
        predict_value.sort(key=self.takeSecond, reverse=True)
        p, t = [], []
        for x in range(int(len(predict_value) * n / 100)):
            p.append(predict_value[x][0])
            t.append(bc_value[x][0])
        return (len(set(t) & set(p)) / len(p))

    def kendall_tau(ground_true, outs, bc_value):
        predict_value, bc_value = [], []
        for i, j in enumerate(outs.tolist()):
            predict_value.append(*j)
        for i in ground_true:
            bc_value.append(i[1])
        # print(predict_value)
        # print(bc_value)
        tau, _ = stats.kendalltau(predict_value, bc_value)
        return (tau)
        print(kendall_tau(f, outs))


def testing(PATH, Ground_truthPath):
    testing = test_data()
    model = DrBC_model()
    # model = torch.load('/content/drive/MyDrive/dataset/model/saved_model.pth')
    model.load_state_dict(torch.load('saved_model.pth'))
    predictbc_value = testing.get_BC(PATH)
    ground_trueBC = testing.get_BC(Ground_truthPath)
    # model.cuda()
    model.eval()
    with torch.no_grad():
        # ground truth
        dataset = testing.readfile(PATH)
        dataset_score = testing.readfile(Ground_truthPath)
        node_degree_data = testing.get_num_node(dataset_score)
        TestGraph_edge_index, TestGraph_feature = testing.get_edge_node_index(dataset, node_degree_data)
        # model = model.cpu()
        outs = model(torch.FloatTensor(TestGraph_feature), torch.tensor(TestGraph_edge_index))
        print(testing.topN_accuracy(PATH, outs, 1, predictbc_value))
        print(testing.topN_accuracy(PATH, outs, 5, predictbc_value))
        print(testing.topN_accuracy(PATH, outs, 10, predictbc_value))
        testing.kendall_tau(ground_trueBC, outs, predictbc_value)


# file path
# yt_node_pair_file =  './youtube/com-youtube.txt'
# yt_bc_score_file = './youtube/com-youtube_score.txt'
# Synthetic_graph = "./Synthetic/5000/Synthetic_graph.txt"
# Synthetic_score = "./Synthetic/5000/Synthetic_score.txt"
# # youtube graph
# testing(yt_node_pair_file,yt_bc_score_file)
# # Synthetic graph
# testing(Synthetic_graph,Synthetic_score)

"""## Data preprocessing"""

# import os

# root_path = "D:/qiongyun/desktop/Master/GNN/hw1_data/Synthetic/5000/"
# f = "D:/qiongyun/desktop/Master/GNN/hw1_data/Synthetic/Synthetic_graph.txt"
# f2 = "D:/qiongyun/desktop/Master/GNN/hw1_data/Synthetic/Synthetic_score.txt"

# # Synthetic graph
# file = open(f,"a") #append mode
# file_2 = open(f2,"a") #append mode
# for id in range(29):
#     file1 = open(os.path.join(root_path, str(id) + '.txt'))
#     file2 = open(os.path.join(root_path, str(id) + '_score.txt'))
#     file.write(file1.read())
#     file_2.write(file2.read())
# file.close()
# file2.close()