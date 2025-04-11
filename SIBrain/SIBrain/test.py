import pickle
import numpy as np
import torch
import scipy.io as sio  # 导入 scipy.io 用于保存 mat 文件

# 处理图中的边矩阵和节点特征
def apply_edge_mask_to_node_features(edge_mat, node_features):
    num_nodes = node_features.shape[0]  # 假设是 116x116 的特征矩阵

    # 创建一个全为0的掩码矩阵，初始化为0
    mask = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 遍历每个边，标记掩码矩阵的相应位置
    for i in range(edge_mat.shape[1]):  # edge_mat 是 2xN
        node1, node2 = edge_mat[:, i]
        mask[node1, node2] = 1  # 设置有边的地方为1，表示有连接

    # 将 mask 转换为 torch.Tensor
    mask = torch.tensor(mask, dtype=torch.float32)

    # 使用掩码矩阵对 node_features 进行操作
    masked_node_features = mask * node_features  # 现在都为 Tensor，可以进行操作了

    return masked_node_features

# 处理所有图
def process_graphs(graphs):
    processed_graphs = []
    for graph in graphs:
        edge_mat = graph.edge_mat  # 2xN 边矩阵
        node_features = graph.node_features  # 116x116 节点特征矩阵

        # 生成新的 node features
        new_node_features = apply_edge_mask_to_node_features(edge_mat, node_features)

        # 给 graph 新属性 new_node_features
        graph.new_node_features = new_node_features
        processed_graphs.append(graph)
        print(f"Processed graph with new_node_features shape: {new_node_features.shape}")

    return processed_graphs

# 加载图文件并处理
def load_graphs(epoch, save_dir="./"):
    filename = f"{save_dir}/all_graphs_epoch_{epoch}.pkl"
    with open(filename, 'rb') as f:
        all_graphs = pickle.load(f)

    # 处理图
    all_graphs = process_graphs(all_graphs)

    # 保存处理后的图
    save_filename = f"{save_dir}/all_graphs_epoch_{epoch}_processed.pkl"
    with open(save_filename, 'wb') as f:
        pickle.dump(all_graphs, f)
    print(f"Processed graphs saved to {save_filename}")

    # 返回处理后的图
    return all_graphs

# 将处理后的图保存为 .mat 文件
def save_graphs_to_mat(graphs, save_dir="./", filename="processed_graphs.mat"):
    # 将每个图的 node_features, new_node_features 和 label 存入一个列表
    data = []
    for graph in graphs:
        node_features = graph.node_features.numpy()  # 将 Tensor 转换为 numpy 数组
        new_node_features = graph.new_node_features.numpy()  # 同上
        label = graph.label  # 假设图有 label 属性

        # 合并为一行，格式是 [node_features, new_node_features, label]
        data.append([node_features, new_node_features, label])

    # 转换成 numpy 数组，准备保存为 .mat 文件
    data = np.array(data, dtype=object)

    # 保存为 .mat 文件
    mat_filename = f"{save_dir}/{filename}"
    sio.savemat(mat_filename, {"graphs": data})
    print(f"Graphs saved to {mat_filename}")

# 主函数
if __name__ == '__main__':
    epoch = 13
    graphs = load_graphs(epoch)

    # 保存处理后的图为 .mat 文件
    save_graphs_to_mat(graphs, save_dir="./", filename=f"processed_abideFinally_{epoch}.mat")
