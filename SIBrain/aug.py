import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import dense_to_sparse


def randomly_zero_symmetric(matrix, drop_rate=0.99):
    """
    在保持对称性的前提下，随机将对称矩阵中部分上三角元素置零。

    :param matrix: NxN 的对称矩阵
    :param drop_rate: 上三角（不包含对角线）中要置零的元素比例
    :return: 置零后的对称矩阵
    """
    assert matrix.shape[0] == matrix.shape[1], "Matrix must be square."

    N = matrix.shape[0]
    # 获取上三角（不包含对角线）的所有索引
    upper_tri_indices = np.triu_indices(N, k=1)
    # 计算上三角中需要置零的数量
    num_to_zero = int(len(upper_tri_indices[0]) * drop_rate)

    # 随机选取上三角中部分索引进行置零
    zero_indices = np.random.choice(len(upper_tri_indices[0]), num_to_zero, replace=False)

    # 对称地将选中的位置置零
    for idx in zero_indices:
        i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        matrix[i, j] = 0
        matrix[j, i] = 0

    return matrix
def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
    """
    单图增强：随机丢弃边（但保留子图）

    实现思路：
    1. 从 graph.x 中获取大图节点特征（矩阵），从 graph.attr 中获取子图结构信息。
    2. 生成子图掩码：将 sub_attr 中非零的位置置为0，零的位置置为1，从而得到一个掩码矩阵。
    3. 将这个掩码矩阵与大图特征矩阵（feature_nodes）逐元素相乘，得到矩阵 A，
       表示大图中比子图多出来的边信息。
    4. 对 A 应用随机对称置零操作（利用 edge_mask_rate 控制置零比例），得到稀疏化后的矩阵 B。
    5. 最后用 B 加上 sub_attr（子图结构）得到最终结果，既保留了子图的边信息，又对其他边进行了稀疏化。
    """

    feature_nodes = graph.x  # 可能是 PyTorch Tensor
    sub_attr = graph.attr  # 可能是 PyTorch Tensor

    # 确保 sub_attr 在 CPU 并转换为 NumPy 数组
    sub_attr_np = sub_attr.cpu().numpy() if isinstance(sub_attr, torch.Tensor) else sub_attr
    feature_nodes_np = feature_nodes.cpu().numpy() if isinstance(feature_nodes, torch.Tensor) else feature_nodes

    # 生成子图掩码
    mask = np.where(sub_attr_np != 0, 0, 1)

    # 提取大图中不属于子图的边信息
    A = feature_nodes_np * mask

    # 进行随机对称置零
    A_sparse = randomly_zero_symmetric(A.copy(), drop_rate=0.99)

    # 计算最终矩阵
    final_matrix = A_sparse + sub_attr_np

    # 转换回 PyTorch Tensor 并转移到与 graph 相同的设备
    final_matrix_tensor = torch.tensor(final_matrix, dtype=torch.float32, device=graph.x.device)

    # 获取 attr 矩阵中非零元素的行列索引
    non_zero_indices = torch.nonzero(final_matrix_tensor != 0)

    # 生成 edge_index，形状为 [2, num_edges]
    edge_index = non_zero_indices.t()

    # 生成 edge_attr，取 attr 中对应位置的值作为边的属性
    edge_attr = final_matrix_tensor[non_zero_indices[:, 0], non_zero_indices[:, 1]]
    # 创建增强后的新图
    augmented_graph = Data(
        x=x,
        attr=final_matrix,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=graph.y
    )

    return augmented_graph, x


def mask_edge(edge_index, mask_prob):
    """
    随机掩码边
    """
    num_edges = edge_index.size(1)
    mask_rates = torch.full((num_edges,), mask_prob, device=edge_index.device)
    masks = torch.bernoulli(1 - mask_rates).bool()
    mask_idx = masks.nonzero(as_tuple=True)[0]  # 仅保留被选中的索引
    return mask_idx


def is_in_subgraph(edge_index, sub_edge_index):
    """
    检查 edge_index 中的边是否在 sub_edge_index 里
    返回一个布尔掩码，True 表示该边在子图内
    """
    edge_set = set(map(tuple, edge_index.t().tolist()))  # 把 edge_index 转换成集合
    sub_edge_set = set(map(tuple, sub_edge_index.t().tolist()))  # 把 sub_edge_index 也转换成集合
    mask = torch.tensor([tuple(e) in sub_edge_set for e in edge_index.t().tolist()], device=edge_index.device)
    return mask

def batch_random_aug(batch, feat_drop_rate, edge_mask_rate, add_self_loops_flag=True):
    """
    批量增强方法：对批次中的每个图单独进行增强处理。
    增强后可以选择是否添加自环。
    """
    augmented_graphs = []

    # 遍历批次中的每个图
    for graph in batch.to_data_list():
        x = graph.x  # 节点特征
        augmented_graph, _ = random_aug(graph, x, feat_drop_rate, edge_mask_rate)
        #
        # # 添加自环（如果需要）
        # if add_self_loops_flag:
        #     augmented_graph.edge_index, augmented_graph.edge_attr = add_self_loops(
        #         augmented_graph.edge_index,
        #         augmented_graph.edge_attr,
        #         fill_value=1.0,  # 你可以根据需要设置自环的权重
        #         num_nodes=augmented_graph.num_nodes
        #     )

        augmented_graphs.append(augmented_graph)

    # 重新组合为一个批次
    augmented_batch = Batch.from_data_list(augmented_graphs)

    return augmented_batch
# def random_aug(graph, x, feat_drop_rate, edge_mask_rate):
#     """
#     单图增强：只处理边，随机删除子图以外的边。
#     """
#     # 获取边信息
#     edge_index = graph.edge_index  # 原始边索引
#     sub_edge_index = graph.sub_edge_index  # 子图边索引
#     edge_attr = graph.edge_attr  # 原始边权重
#
#
#     # 获取图边的集合（转换为集合操作方便）
#     edge_set = set(map(tuple, edge_index.t().tolist()))
#
#
#     # 获取子图边的集合（转换为集合操作方便）
#     sub_edge_set = set(map(tuple, sub_edge_index.t().tolist()))  # 转为 {(u, v), ...} 格式
#
#     # 筛选出非子图边的索引
#     all_edges = edge_index.t().tolist()  # 转为 [[u, v], ...] 格式
#     non_sub_edge_mask = [i for i, edge in enumerate(all_edges) if tuple(edge) not in sub_edge_set]  # 非子图边的索引
#     non_sub_edges = edge_index[:, non_sub_edge_mask]  # 非子图边索引矩阵
#
#     # 对非子图边进行随机掩码
#     non_sub_edge_mask_random = mask_edge(non_sub_edges, edge_mask_rate)  # 生成非子图边的掩码
#     kept_non_sub_edges = non_sub_edges[:, non_sub_edge_mask_random]  # 掩码后保留的非子图边
#
#     # 子图边保持不变
#     kept_sub_edges = sub_edge_index
#
#     # 合并子图边和掩码后的非子图边
#     new_edge_index = torch.cat([kept_sub_edges, kept_non_sub_edges], dim=1)
#
#     # 创建增强后的新图
#     augmented_graph = Data(
#         x=x,  # 不对特征进行修改
#         edge_index=new_edge_index,
#         edge_attr=None,  # 如果需要，可以修改以支持边属性
#         y=graph.y
#     )
#
#     return augmented_graph, x
