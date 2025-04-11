import argparse
from model import CCA_SSG, LogReg
from aug import batch_random_aug
from dataset import load_multigraph_dataset
from torch_geometric.data import DataLoader
import numpy as np
import torch as th
import torch.nn as nn
from sklearn.metrics import roc_auc_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='CCA-SSG')

parser.add_argument('--dataname', type=str, default='MyData', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=3, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of CCA-SSG.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of CCA-SSG.')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')
parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')
parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')
parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')
parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'


def run_classifier():
    # 加载保存的嵌入和标签
    data = th.load('ABIDE28001/embeddings_and_labels_epoch_010.pt')

    train_embs = data['train_embs'].to(args.device)
    train_labels = data['train_labels'].to(args.device)
    val_embs = data['val_embs'].to(args.device)
    val_labels = data['val_labels'].to(args.device)
    test_embs = data['test_embs'].to(args.device)
    test_labels = data['test_labels'].to(args.device)

    num_class = 2

    logreg = LogReg(train_embs.shape[1], num_class).to(args.device)
    opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    for epoch in range(2000):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = (preds == train_labels).float().mean()
        loss = loss_fn(logits, train_labels)
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():
            # 验证集和测试集预测
            val_logits = logreg(val_embs)
            test_logits = logreg(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            val_acc = (val_preds == val_labels).float().mean()

            # 计算 softmax 之后的概率
            test_probs = th.softmax(test_logits, dim=1)[:, 1].cpu().numpy()  # 取类别 1 的概率

            # 调整阈值（比如 0.40），低于 0.40 预测为 0，高于 0.40 预测为 1
            threshold = 0.5
            test_preds = (test_probs > threshold).astype(int)

            # 计算各项指标
            test_labels_np = test_labels.cpu().numpy()

            test_acc = np.mean(test_preds == test_labels_np)
            try:
                test_auc = roc_auc_score(test_labels_np, test_probs)
            except Exception as e:
                test_auc = 0.0
            test_recall = recall_score(test_labels_np, test_preds, zero_division=0)
            test_f1 = f1_score(test_labels_np, test_preds, zero_division=0)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                eval_acc = test_acc

            print(f'Epoch:{epoch}, train_acc:{train_acc:.4f}, val_acc:{val_acc:.4f}, '
                  f'test_acc:{test_acc:.4f}, test_auc:{test_auc:.4f}, '
                  f'test_recall:{test_recall:.4f}, test_f1:{test_f1:.4f}')

    print(f'Linear evaluation accuracy: {eval_acc:.4f}')


if __name__ == '__main__':
    run_classifier()
