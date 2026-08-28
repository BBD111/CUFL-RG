import torch
import torch.nn as nn
import torch.nn.functional as F


class IESModule(nn.Module):
    def __init__(self, num_nodes, total_rounds, gamma=0.1, device='cuda:0',
                 warmup_rounds=20, init_value=2.0, pacing_max=0.8, loss_scale=1.0,
                 budget_rho=0.6, budget_mu=0.1):
        super(IESModule, self).__init__()
        self.num_nodes = num_nodes
        self.total_rounds = total_rounds
        self.gamma = gamma
        self.device = device
        self.warmup_rounds = warmup_rounds
        self.pacing_max = pacing_max
        self.loss_scale = loss_scale

        # 预算/均值约束参数
        self.budget_rho = budget_rho
        self.budget_mu = budget_mu

        # 初始化
        self.S_k = nn.Parameter(torch.ones(num_nodes, num_nodes, device=self.device) * init_value)

    def get_pacing_lambda(self, current_round):
        # warm-up 期间不推进课程
        if current_round < self.warmup_rounds:
            return 0.0

        # warm-up 之后再线性增长，而且上限不要太激进
        effective_round = current_round - self.warmup_rounds
        effective_total = max(self.total_rounds - self.warmup_rounds, 1)

        if effective_total > 0:
            return min((effective_round / effective_total) * self.pacing_max, self.pacing_max)
        return self.pacing_max

    def forward(self, adj, node_embeddings, current_round):
        """
        adj: 原始二进制邻接矩阵 [N, N]
        node_embeddings: 节点特征 [N, D]
        current_round: 当前训练轮数
        """
        # warm-up：不做筛边，直接保留原图，不引入 IES 干扰
        if current_round < self.warmup_rounds:
            refined_adj = adj
            ies_loss = torch.tensor(0.0, device=adj.device)
            return refined_adj, ies_loss

        # 1. 余弦相似度重建图
        norm_emb = F.normalize(node_embeddings, p=2, dim=1)
        A_hat = torch.mm(norm_emb, norm_emb.t())

        # 2. 当前进度参数
        pacing_lambda = self.get_pacing_lambda(current_round)

        # 3. 掩码权重
        mask_weights = torch.sigmoid(self.S_k)

        # 4. 精炼邻接矩阵
        refined_adj = mask_weights * adj

        # 5. 稳定版 IES 损失：只惩罚超过容忍度的边，保证损失非负
        rec_error = torch.abs(adj - A_hat)
        margin = F.relu(rec_error - pacing_lambda)

        # 只在真实存在的边上统计课程损失
        edge_mask = (adj > 0).float()
        edge_num = torch.clamp(edge_mask.sum(), min=1.0)

        ies_core_loss = self.loss_scale * torch.sum(
            mask_weights * edge_mask * margin * margin
        ) / edge_num

        # 6. 预算/均值约束项，防止所有边权一起缩小
        avg_edge_weight = torch.sum(mask_weights * edge_mask) / edge_num
        budget_loss = (avg_edge_weight - self.budget_rho) ** 2

        # 7. 最终课程损失
        ies_loss = ies_core_loss + self.budget_mu * budget_loss

        return refined_adj, ies_loss