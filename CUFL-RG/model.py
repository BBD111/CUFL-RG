import torch
import torch.nn as nn
from GAT import GraphAttentionLayer


class model(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.GAT_neighbor = GraphAttentionLayer(embed_size, embed_size)
        self.GAT_item = GraphAttentionLayer(embed_size, embed_size)

        self.u_relation_neighbor = nn.Parameter(torch.randn(embed_size, device=torch.device('cuda:0')))
        self.u_relation_item = nn.Parameter(torch.randn(embed_size, device=torch.device('cuda:0')))
        self.u_relation_self = nn.Parameter(torch.randn(embed_size, device=torch.device('cuda:0')))
        self.c = nn.Parameter(torch.randn(2 * embed_size, device=torch.device('cuda:0')))

        self.i_relation_neighbor = nn.Parameter(torch.randn(embed_size, device=torch.device('cuda:0')))
        self.i_relation_self = nn.Parameter(torch.randn(embed_size, device=torch.device('cuda:0')))
        self.d = nn.Parameter(torch.randn(2 * embed_size, device=torch.device('cuda:0')))

    def predict(self, user_embedding, item_embedding):
        return torch.matmul(user_embedding, item_embedding.t())

    def forward(self, feature_self, feature_neighbor, feature_item, feature_item_neighbor, item_neighbor_len_list,
                refined_adjs=None):
        # 解包 refined_adjs (来自 IES 模块的掩码)
        adj_social = None
        adj_interact = None
        adj_item_corr_list = None

        if refined_adjs is not None:
            adj_social, adj_interact, adj_item_corr_list = refined_adjs

        if type(feature_item) == torch.Tensor:
            item_embedding = torch.randn(len(feature_item), self.embed_size, device=torch.device('cuda:0'))
            count = 0
            for i in range(len(feature_item)):
                current_item_index = torch.tensor(i, dtype=torch.long)
                feature_item_i = feature_item[current_item_index]

                length = item_neighbor_len_list[current_item_index][1]
                feature_item_neighbor_i = feature_item_neighbor[0 + count: length + count]

                # 获取当前物品的 IES 邻接权重
                current_adj_corr = None
                if adj_item_corr_list is not None:
                    current_adj_corr = adj_item_corr_list[0 + count: length + count]

                count = length + count

                # 传入 refined_adj
                g_n = self.GAT_neighbor(feature_item_i, feature_item_neighbor_i, refined_adj=current_adj_corr)

                # 维度匹配 squeeze(0)
                i_n = torch.matmul(self.d, torch.cat((g_n.squeeze(0), self.i_relation_neighbor)))
                i_s = torch.matmul(self.d, torch.cat((feature_item_i, self.i_relation_self)))

                n = nn.Softmax(dim=-1)
                i_tensor = torch.stack([i_n, i_s])
                i_tensor = n(i_tensor)
                p_n, p_s = i_tensor
                item_embedding_i = p_s * feature_item_i + p_n * g_n
                item_embedding[current_item_index] = item_embedding_i

            # 传入 refined_adj (Social 和 Interact)
            f_n = self.GAT_neighbor(feature_self, feature_neighbor, refined_adj=adj_social)
            f_i = self.GAT_item(feature_self, item_embedding, refined_adj=adj_interact)

            # 维度匹配 squeeze(0)
            e_n = torch.matmul(self.c, torch.cat((f_n.squeeze(0), self.u_relation_neighbor)))
            e_i = torch.matmul(self.c, torch.cat((f_i.squeeze(0), self.u_relation_item)))
            e_s = torch.matmul(self.c, torch.cat((feature_self.squeeze(0), self.u_relation_self)))

            m = nn.Softmax(dim=-1)
            e_tensor = torch.stack([e_n, e_i, e_s])
            e_tensor = m(e_tensor)
            r_n, r_i, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n + r_i * f_i

            item_neighbor_embedding = feature_item_neighbor
            return user_embedding, item_embedding, item_neighbor_embedding

        else:
            # 冷启动
            f_n = self.GAT_neighbor(feature_self, feature_neighbor, refined_adj=adj_social)

            e_n = torch.matmul(self.c, torch.cat((f_n.squeeze(0), self.u_relation_neighbor)))
            e_s = torch.matmul(self.c, torch.cat((feature_self.squeeze(0), self.u_relation_self)))

            m = nn.Softmax(dim=-1)
            e_tensor = torch.stack([e_n, e_s])
            e_tensor = m(e_tensor)
            r_n, r_s = e_tensor
            user_embedding = r_s * feature_self + r_n * f_n
            return user_embedding