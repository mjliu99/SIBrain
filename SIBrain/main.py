import argparse
from model import CCA_SSG, LogReg
from aug import batch_random_aug
from dataset import load_multigraph_dataset
from torch_geometric.data import DataLoader
import numpy as np
import torch as th
import torch.nn as nn

import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='CCA-SSG')

parser.add_argument('--dataname', type=str, default='MyData', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=30, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of CCA-SSG.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of CCA-SSG.')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')

parser.add_argument('--lambd', type=float, default=0.0001, help='trade-off ratio.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')

parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')

parser.add_argument('--der', type=float, default=0.95, help='Drop edge ratio.')
parser.add_argument('--dfr', type=float, default=0.95, help='Drop feature ratio.')

parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')

args = parser.parse_args()

# check cuda
if args.gpu!= -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


if __name__ == '__main__':
    print(args)
    train_dataset, val_dataset, test_dataset = load_multigraph_dataset(args.dataname)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    print(f"Dataset loaded: {args.dataname}")
    print(f"Number of graphs: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    # in_dim为90 是因为我的数据集的节点特征矩阵是116*116
    in_dim = 116
    model = CCA_SSG(in_dim, args.hid_dim, args.out_dim, args.n_layers, args.device)
    model = model.to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)


    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0  # 累计每个 epoch 的损失

        # 训练循环
        for batch_idx, batch in enumerate(train_loader):  # 从 DataLoader 获取批量图数据
            batch = batch.to(args.device)
            optimizer.zero_grad()  # 清空之前计算的梯度

            # 批量增强，针对每个图单独增强
            augmented_batch1 = batch_random_aug(batch, args.dfr, args.der, add_self_loops_flag=True)
            augmented_batch2 = batch_random_aug(batch, args.dfr, args.der, add_self_loops_flag=True)

            # 使用增强后的图和特征进行训练
            z1, z2 = model(augmented_batch1, augmented_batch2)

            # 初始化批次损失
            batch_loss = 0

            # 遍历批次中的每个图，逐图计算损失
            for i in range(z1.size(0)):  # z1.size(0) 是 batch_size
                z1_i = z1[i]  # 当前图的 z1: [num_nodes, embedding_dim]
                z2_i = z2[i]  # 当前图的 z2: [num_nodes, embedding_dim]

                N = z1_i.size(0)  # 当前图的节点数

                # 计算协方差矩阵
                c = th.mm(z1_i.T, z2_i) / N  # [embedding_dim, embedding_dim]
                c1 = th.mm(z1_i.T, z1_i) / N  # [embedding_dim, embedding_dim]
                c2 = th.mm(z2_i.T, z2_i) / N  # [embedding_dim, embedding_dim]

                # 损失计算
                loss_inv = -th.diagonal(c).sum()  # 对齐损失
                iden = th.eye(c.size(0), device=c.device)
                loss_dec1 = ((iden - c1).pow(2).sum())  # 去冗余损失 1
                loss_dec2 = ((iden - c2).pow(2).sum())  # 去冗余损失 2

                # 当前图的总损失
                graph_loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)

                # 累计到批次损失
                batch_loss += graph_loss

            # 对批次损失取平均
            batch_loss = batch_loss / z1.size(0)

            # 累计 epoch 损失
            epoch_loss += batch_loss.item()

            # 反向传播并更新梯度
            batch_loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数

        # 打印每个 epoch 的平均损失
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f'Epoch={epoch:03d}, Avg Loss={avg_epoch_loss:.4f}')
        print("=== Evaluation ===")
        model.eval()

        # 用于存储所有图的嵌入和标签
        train_embs, train_labels = [], []
        val_embs, val_labels = [], []
        test_embs, test_labels = [], []
        for batch in train_loader:
            batch = batch.to(args.device)
            embeds = model.get_embedding(batch)  # 假设 get_embedding 方法支持图级嵌入
            # print(type(embeds))  # 检查类型
            if isinstance(embeds, list):  # 如果是列表类型
                # 将列表中的每个图向量合并成一个大张量
                batch_embeds = th.cat(embeds, dim=0)  # 按批次维度拼接
            else:
                batch_embeds = embeds  # 如果已经是张量，就直接使用

            train_embs.append(batch_embeds)  # 添加到训练集嵌入列表
            train_labels.append(batch.y)  # 添加到训练集标签列表

        # 处理验证集
        for batch in val_loader:
            batch = batch.to(args.device)
            embeds = model.get_embedding(batch)
            if isinstance(embeds, list):
                batch_embeds = th.cat(embeds, dim=0)
            else:

                batch_embeds = embeds
            val_embs.append(batch_embeds)
            val_labels.append(batch.y)

        # 处理测试集
        for batch in test_loader:
            batch = batch.to(args.device)
            embeds = model.get_embedding(batch)
            if isinstance(embeds, list):
                batch_embeds = th.cat(embeds, dim=0)
            else:
                batch_embeds = embeds
            test_embs.append(batch_embeds)
            test_labels.append(batch.y)

        # 合并批次嵌入和标签
        train_embs = th.cat(train_embs, dim=0)  # 形状：[num_train_graphs, embedding_dim]
        train_labels = th.cat(train_labels, dim=0)  # 形状：[num_train_graphs]
        val_embs = th.cat(val_embs, dim=0)
        val_labels = th.cat(val_labels, dim=0)
        test_embs = th.cat(test_embs, dim=0)
        test_labels = th.cat(test_labels, dim=0)
        file_name = f'embeddings_and_labels_epoch_{epoch:03d}.pt'

        # 保存嵌入和标签
        th.save({
            'train_embs': train_embs,
            'train_labels': train_labels,
            'val_embs': val_embs,
            'val_labels': val_labels,
            'test_embs': test_embs,
            'test_labels': test_labels
        }, file_name)


