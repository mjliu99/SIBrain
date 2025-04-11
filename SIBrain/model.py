from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn import GATConv
import sys
import scipy.io as sio
class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        ret = self.fc(x)
        return ret

class GNN(nn.Module):
    def __init__(self, num_of_features, device):
        """
        GNN 类是一个图神经网络模型，包含多个图卷积层和后续的池化及多层感知机处理。
        参数：
        num_of_features：输入特征的数量。
        device：计算设备，如 'cuda' 或 'cpu'。
        """
        super(GNN, self).__init__()
        self.num_of_features = num_of_features
        self.first_gcn_dimensions = 128
        self.second_gcn_dimensions = 128
        self.SOPOOL_dim_1 = 512
        self.SOPOOL_dim_2 = 512
        self.linear_hidden_dimensions = 512
        self.output_dimensions = 2
        self.device = device

        # 第一个图卷积层，将输入特征维度映射到 first_gcn_dimensions
        self.graph_conv_1 = GATConv(num_of_features, self.first_gcn_dimensions,edge_dim=1)
        # 第二个图卷积层，将 first_gcn_dimensions 映射到 second_gcn_dimensions
        self.graph_conv_2 = GATConv(self.first_gcn_dimensions, self.second_gcn_dimensions,edge_dim=1)

        # 自定义的池化层，使用多个线性层和 ReLU 激活函数
        self.SOPOOL = nn.Sequential(OrderedDict([
            ("Linear_1", nn.Linear(self.second_gcn_dimensions, self.SOPOOL_dim_1)),
            ("ReLU", nn.ReLU()),
            ("Linear_2", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_1)),
            ("ReLU", nn.ReLU()),
            ("Linear_3", nn.Linear(self.SOPOOL_dim_1, self.SOPOOL_dim_2)),
            ("ReLU", nn.ReLU())
        ]))

        # 多层感知机层，将池化结果映射到最终输出维度
        self.MLP_1 = nn.Sequential(OrderedDict([
            ("Linear_1", nn.Linear(self.SOPOOL_dim_2 ** 2, self.linear_hidden_dimensions)),
            ("ReLU", nn.ReLU()),
            ("Linear_2", nn.Linear(self.linear_hidden_dimensions, self.linear_hidden_dimensions)),
            ("ReLU", nn.ReLU()),
            ("Linear_3", nn.Linear(self.linear_hidden_dimensions, self.output_dimensions)),
            ("ReLU", nn.ReLU()),
        ]))

    def forward(self, graph):
        """
        前向传播函数，适用于单图。
        参数：
        graph：单图数据，包含节点特征、边索引等信息。
        """
        # 将图数据移到指定设备上
        graph = graph.to(self.device)
        # 第一个图卷积层操作，使用 ReLU 激活
        node_features_1 = F.relu(
            self.graph_conv_1(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr))
        # 这样吧 node_features_1判断一下结果 如果是nan 打印它的graph.x成mat矩阵 graph.attr成mat矩阵 我要看看区别
        # 检查 node_features_1 是否包含 NaN
        if torch.isnan(node_features_1).any():
            print("NaN detected in node_features_1!")

            # 将 graph.x 和 graph.attr 转换为 NumPy 并保存为 .mat 文件
            graph_x_np = graph.x.cpu().numpy() if isinstance(graph.x, torch.Tensor) else graph.x
            graph_attr_np = graph.attr.cpu().numpy() if isinstance(graph.attr, torch.Tensor) else graph.attr

            sio.savemat("graph_x.mat", {"graph_x": graph_x_np})
            sio.savemat("graph_attr.mat", {"graph_attr": graph_attr_np})

            print("graph.x and graph.attr have been saved as .mat files for debugging.")
            sys.exit(1)
        # 第二个图卷积层操作，使用 ReLU 激活
        node_features_2 = F.relu(
            self.graph_conv_2(x=node_features_1, edge_index=graph.edge_index, edge_attr=graph.edge_attr))
        # 对节点特征进行 dropout 操作
        node_features_ = F.dropout(node_features_2, p=0.5, training=self.training)
        # 对节点特征进行归一化操作
        normalized_node_features = F.normalize(node_features_, dim=1)
        # 这里使用了全局平均池化 (global_mean_pool)，也可以替换为 global_max_pool
        graph_level_features = global_mean_pool(normalized_node_features, graph.batch)
        # 对单图应用 SOPOOL 操作
        graph_put = self.SOPOOL(normalized_node_features)
        # 计算图的特征矩阵的转置积并展平
        # HH_tensor = torch.mm(graph.t(), graph).view(1, -1)
        # 通过 MLP 层得到最终输出
        # output = F.dropout(self.MLP_1(HH_tensor), p=0.5, training=self.training)
        torch.cuda.empty_cache()
        return graph_put, graph_level_features


class CCA_SSG(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, device):
        super().__init__()
        self.backbone = GNN(in_dim, device)
        # else:
        #     self.backbone = MLP(in_dim, hid_dim, out_dim)

    def get_embedding(self, batch):
        embedding = []
        for graph in batch.to_data_list():
            x, edge_index = graph.x, graph.edge_index
            h,output = self.backbone(graph)  # 节点级嵌入
            # z = self.pool(h, torch.zeros(x.size(0), dtype=torch.long, device=x.device))  # 图级嵌入

            embedding.append(output)
        return embedding


    def forward(self, batch1, batch2):
        """
        针对每个图独立进行前向计算和标准化。
        """
        z1_list, z2_list = [], []

        # 分别处理 batch1 中的每个图
        for graph in batch1.to_data_list():
            x, edge_index = graph.x, graph.edge_index
            h,output = self.backbone(graph)  # 节点级嵌入
            z1 = (h - h.mean(0)) / h.std(0)
            # z = self.pool(h, torch.zeros(x.size(0), dtype=torch.long, device=x.device))  # 图级嵌入

            z1_list.append(z1)

        # 分别处理 batch2 中的每个图
        for graph in batch2.to_data_list():
            x, edge_index = graph.x, graph.edge_index
            h,output = self.backbone(graph)  # 节点级嵌入
            z2 = (h - h.mean(0)) / h.std(0)
            # z = self.pool(h, torch.zeros(x.size(0), dtype=torch.long, device=x.device))  # 图级嵌入
            z2_list.append(z2)
        # 将每个图的结果组合成一个 tensor
        z1 = torch.stack(z1_list, dim=0)  # shape: [num_graphs_in_batch1, embedding_dim]
        z2 = torch.stack(z2_list, dim=0)  # shape: [num_graphs_in_batch2, embedding_dim]

        return z1, z2

