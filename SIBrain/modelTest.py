import torch
import scipy.io as sio
import torch.nn.functional as F
from torch_geometric.nn import GATConv  # 使用 GATConv
from torch import nn
from torch_geometric.data import Data

# 加载 .mat 文件
graph_x_data = sio.loadmat('data/graph_x.mat')
graph_attr_data = sio.loadmat('data/graph_attr.mat')

# 假设保存的矩阵是 graph_x 和 graph_attr
graph_x = torch.tensor(graph_x_data['graph_x'], dtype=torch.float32)
graph_attr = torch.tensor(graph_attr_data['graph_attr'], dtype=torch.float32)

# 检查 graph_x 和 graph_attr 是否包含 NaN 或 Inf
if torch.isnan(graph_x).any() or torch.isinf(graph_x).any():
    print("NaN or Inf detected in graph_x!")
    print(graph_x)

if torch.isnan(graph_attr).any() or torch.isinf(graph_attr).any():
    print("NaN or Inf detected in graph_attr!")
    print(graph_attr)

# 处理 NaN 和 Inf（如果有的话）
graph_x = torch.nan_to_num(graph_x)  # 替换 NaN 为 0，Inf 为有限数值
graph_attr = torch.nan_to_num(graph_attr)  # 替换 NaN 为 0，Inf 为有限数值

# 创建一个 PyG 的 Data 对象
edge_index = graph_attr.to_sparse().coalesce().indices()  # 获取邻接矩阵的稀疏表示
edge_attr = graph_attr.to_sparse().coalesce().values()  # 获取邻接矩阵的边权重

# 将数据传入 Data 对象
data = Data(x=graph_x, edge_index=edge_index, edge_attr=edge_attr)

# 定义 GAT 网络（Graph Attention）
class GATModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, 64, edge_dim=1)  # GATConv 层，edge_dim 为边特征的维度
        self.conv2 = GATConv(64, 512, edge_dim=1)  # 第二层，输出维度为 512

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        return x  # 直接返回节点特征

# 初始化模型
model = GATModel(in_channels=graph_x.size(1), out_channels=512)  # 输出维度为 512

# 测试模型的前向传播
out = model(data)
print(out.shape)  # 打印输出的形状，应该是 (116, 512)
