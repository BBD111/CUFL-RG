import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.1, residual_epsilon=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.residual_epsilon = residual_epsilon

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features), device=torch.device('cuda')))
        nn.init.xavier_uniform_(self.W.data)

        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1), device=torch.device('cuda')))
        nn.init.xavier_uniform_(self.a.data)

        self.W_1 = nn.Parameter(torch.randn(in_features, out_features, device=torch.device('cuda')))
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, refined_adj=None):
        # h: 中心节点特征 [1, D] 或 [D]
        # adj: 邻居节点特征矩阵 [N, D]
        # refined_adj: IES 输出的边权重向量 [N, 1] 或 [N]

        if h.dim() == 1:
            h = h.unsqueeze(0)

        W_h = torch.matmul(h, self.W)          # [1, out_features]
        W_adj = torch.mm(adj, self.W)          # [N, out_features]

        # 拼接中心节点与邻居节点特征
        a_input = torch.cat((W_h.repeat(W_adj.shape[0], 1), W_adj), dim=1)  # [N, 2*out_features]

        # 原始注意力分数
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(-1)       # [N]

        # 先做标准 GAT softmax
        attention = F.softmax(e, dim=-1)                                    # [N]

        # 再融合 IES 边权，重新归一化：
        # m_tilde_j = eps + (1 - eps) * m_j
        # alpha'_j = alpha_j * m_tilde_j / sum_k alpha_k * m_tilde_k
        if refined_adj is not None:
            if refined_adj.dim() > 1:
                refined_adj = refined_adj.squeeze(-1)                        # [N]

            refined_adj = self.residual_epsilon + (1.0 - self.residual_epsilon) * refined_adj
            attention = attention * refined_adj
            attention = attention / (attention.sum() + 1e-9)

        W_adj_transform = torch.mm(adj, self.W_1)                           # [N, out_features]

        # 加权聚合邻居信息
        h_prime = torch.matmul(attention.unsqueeze(0), W_adj_transform)     # [1, out_features]

        return h_prime