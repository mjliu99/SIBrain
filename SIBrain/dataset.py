import numpy as np
import torch as th
import os
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset
import scipy.io as scio
from torch_geometric.data import Data
from torch.utils.data import random_split
def load(name):
    if name == 'cora':
        dataset = CoraGraphDataset()
    elif name == 'citeseer':
        dataset = CiteseerGraphDataset()
    elif name == 'pubmed':
        dataset = PubmedGraphDataset()
    elif name == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
    elif name == 'comp':
        dataset = AmazonCoBuyComputerDataset()
    elif name == 'cs':
        dataset = CoauthorCSDataset()
    elif name == 'physics':
        dataset = CoauthorPhysicsDataset()

    graph = dataset[0]
    citegraph = ['cora', 'citeseer', 'pubmed']
    cograph = ['photo', 'comp', 'cs', 'physics']

    if name in citegraph:
        train_mask = graph.ndata.pop('train_mask')
        val_mask = graph.ndata.pop('val_mask')
        test_mask = graph.ndata.pop('test_mask')

        train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
        val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
        test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if name in cograph:
        train_ratio = 0.1
        val_ratio = 0.1
        test_ratio = 0.8

        N = graph.number_of_nodes()
        train_num = int(N * train_ratio)
        val_num = int(N * (train_ratio + val_ratio))

        idx = np.arange(N)
        np.random.shuffle(idx)

        train_idx = idx[:train_num]
        val_idx = idx[train_num:val_num]
        test_idx = idx[val_num:]

        train_idx = th.tensor(train_idx)
        val_idx = th.tensor(val_idx)
        test_idx = th.tensor(test_idx)

    num_class = dataset.num_classes
    feat = graph.ndata.pop('feat')
    labels = graph.ndata.pop('label')

    return graph, feat, labels, num_class, train_idx, val_idx, test_idx


def load_multigraph_dataset(name):
    """
    此函数的目的是从文件 'md_AAL_0.4.mat' 中读取数据，并将其重构为 'torch_geometric.data.Data' 类型的数据。
    同时，该函数将数据集划分为训练集、验证集和测试集。
    """
    # 从文件中加载数据，假设文件是一个 MATLAB 文件，使用 scio.loadmat 函数
    # 2-8
    # data = scio.loadmat('data/abide_processed.mat')  # Data is available at google drive
    # 6-4
    data = scio.loadmat('data/ABIDE28001.mat')  # Data is available at google drive
    # 创建一个空列表用于存储图数据
    dataset = []
    # data2 = scio.loadmat('data/abide.mat')
    # sub_graph_struct = data2['graph_struct']
    # 遍历数据中的每个图的索引
    # for graph_index in range(len(data['label'])):
    #     # 获取标签数据
    #     label = data['label']
    #     # 获取图结构数据
    #     graph_struct = data['graph_struct'][0]
    #     # 提取边数据并转换为 Tensor 类型
    #     edge = th.Tensor(graph_struct[graph_index]['edge'])
    #     # 提取 ROI 数据并转换为 Tensor 类型
    #     ROI = th.Tensor(graph_struct[graph_index]['ROI'])
    #
    #     SubROI = th.Tensor(sub_graph_struct[graph_index][1])
    #     subROI = (SubROI + SubROI.t()) / 2  # 对称化处理，使 ROI 矩阵变为对称矩阵
    #     # 提取当前图的标签并转换为 Tensor 类型
    #     y = th.Tensor(label[graph_index])
    #     # 创建稀疏矩阵 A，使用稀疏 COO 格式存储
    #     A = th.sparse_coo_tensor(
    #         indices=edge[:, :2].t().long(),
    #         values=edge[:, -1].reshape(-1, ).float(),
    #         size=(116, 116)
    #     )
    #     # 对稀疏矩阵 A 进行对称化处理
    #     G = (A.t() + A).coalesce()
    #     # 处理 subROI：提取边索引和边属性
    #     subROI_indices = th.nonzero(subROI)  # 提取非零元素的索引
    #     sub_edge_index = subROI_indices.t()  # 转置为 2 x N 的形式
    #     # 创建图数据对象，包含节点特征、边索引、边属性和标签
    #     graph = Data(
    #         x=ROI.reshape(-1, 116).float(),
    #         edge_index=G.indices().reshape(2, -1).long(),
    #         edge_attr=G.values().reshape(-1, 1).float(),
    #         sub_edge_index=sub_edge_index.long(),  # 附加子图边索引
    #         y=y.long()
    #     )
    #     edge_set = set(map(tuple, graph.edge_index.t().tolist()))
    #     # 将图数据添加到数据集列表中
    #     dataset.append(graph)
    # 遍历数据中的每个图的索引
    # 遍历数据中的每个图的索引
    for graph_index in range(len(data['label'])):
        # 获取标签数据
        label = data['label']
        # 获取图结构数据
        graph_struct = data['graphs']
        feature_nodes = th.Tensor(graph_struct[graph_index][0])  # 假设 '1' 是 ROI 的键
        # 提取 ROI 数据并转换为 Tensor 类型
        ROI = th.Tensor(graph_struct[graph_index][1])  # 假设 '1' 是 ROI 的键
        y = th.Tensor(label[graph_index])
        # 对邻接矩阵 (ROI) 进行对称化处理
        # ROI = (ROI + ROI.t()) / 2  # 对称化处理，使 ROI 矩阵变为对称矩阵

       # ROI当graph.x 然后根据ROI抽取出edge_index edge_attr  graph.edge_index  graph.edge_attr 又当邻接矩阵
       # 我的ROI是116X116的无向图
        # 获取边索引和边属性
        edge_index = th.nonzero(ROI > 0, as_tuple=False).t()  # 获取非零元素的索引
        edge_attr = ROI[edge_index[0], edge_index[1]]  # 提取对应的边权重

        # 构建 PyG 数据对象
        graph = Data(
            x=feature_nodes,
            attr=ROI,# 节点特征矩阵
            edge_index=edge_index,  # 边索引
            edge_attr=edge_attr,  # 边属性（权重）
            y=y.long()  # 标签
        )

        # 将图数据添加到数据集列表中
        dataset.append(graph)

    # for graph_index in range(len(data['label'])):
    #     # 获取标签数据
    #     label = data['label']
    #     # 获取图结构数据
    #     graph_struct = data['graph_struct'][0]
    #
    #     # 提取边数据并转换为 Tensor 类型
    #     edge = th.Tensor(graph_struct[graph_index]['edge'])
    #     # 提取 ROI 数据并转换为 Tensor 类型
    #     ROI = th.Tensor(graph_struct[graph_index]['ROI'])
    #     # 提取当前图的标签并转换为 Tensor 类型
    #     y = th.Tensor(label[graph_index])
    #
    #     # 创建稀疏矩阵 A，使用稀疏 COO 格式存储
    #     A = th.sparse_coo_tensor(
    #         indices=edge[:, :2].t().long(),
    #         values=edge[:, -1].reshape(-1, ).float(),
    #         size=(116, 116)
    #     )
    #     # 对稀疏矩阵 A 进行对称化处理
    #     G = (A.t() + A).coalesce()
    #
    #     # 创建图数据对象，包含节点特征、边索引、边属性和标签
    #     graph = Data(
    #         x=ROI.reshape(-1, 116).float(),
    #         edge_index=G.indices().reshape(2, -1).long(),
    #         edge_attr=G.values().reshape(-1, 1).float(),
    #         y=y.long()
    #     )
    #     # 将图数据添加到数据集列表中
    #     dataset.append(graph)


    # 计算训练集的大小，为数据集的 80%
    train_size = int(0.8 * len(dataset))
    # 计算验证集的大小，为数据集的 10%
    val_size = int(0.1 * len(dataset))
    # 计算测试集的大小，为数据集的剩余部分
    test_size = len(dataset) - train_size - val_size

    # 使用 random_split 函数将数据集划分为训练集、验证集和测试集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 返回划分好的训练集、验证集和测试集
    return train_dataset, val_dataset, test_dataset
    # for graph_index in range(len(data['label'])):
    #     # 获取标签数据
    #     label = data['label']
    #     # 获取图结构数据
    #     graph_struct = data['graph_struct'][0]
    #
    #     # 提取边数据并转换为 Tensor 类型
    #     edge = th.Tensor(graph_struct[graph_index]['edge'])
    #     # 提取 ROI 数据并转换为 Tensor 类型
    #     ROI = th.Tensor(graph_struct[graph_index]['ROI'])
    #     # 提取当前图的标签并转换为 Tensor 类型
    #     y = th.Tensor(label[graph_index])
    #
    #     # 创建稀疏矩阵 A，使用稀疏 COO 格式存储
    #     A = th.sparse_coo_tensor(
    #         indices=edge[:, :2].t().long(),
    #         values=edge[:, -1].reshape(-1, ).float(),
    #         size=(116, 116)
    #     )
    #     # 对稀疏矩阵 A 进行对称化处理
    #     G = (A.t() + A).coalesce()
    #
    #     # 创建图数据对象，包含节点特征、边索引、边属性和标签
    #     graph = Data(
    #         x=ROI.reshape(-1, 116).float(),
    #         edge_index=G.indices().reshape(2, -1).long(),
    #         edge_attr=G.values().reshape(-1, 1).float(),
    #         y=y.long()
    #     )
    #     # 将图数据添加到数据集列表中
    #     dataset.append(graph)