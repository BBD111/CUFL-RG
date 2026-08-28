import torch
import copy
from random import sample
import numpy as np
import dgl
import pdb
from model import model
import networkx as nx
import matplotlib.pyplot as plt
import collections
from curriculum import IESModule
import torch.nn.functional as F


torch.multiprocessing.set_sharing_strategy('file_system')


class user():
    def __init__(self, id_self, items, ratings, neighbors, embed_size, clip, ldp_epsilon, negative_sample,
                 item_relations, total_rounds=100,
                 social_reg=0.03, contrastive_temp=0.2, contrastive_neg_num=16,
                 item_contrastive_reg=0.01, item_contrastive_temp=0.2, item_contrastive_neg_num=8):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.negative_sample = negative_sample
        self.clip = clip
        self.ldp_epsilon = ldp_epsilon
        self.id_self = id_self
        self.items = items
        self.embed_size = embed_size
        self.ratings = ratings
        self.neighbors = neighbors

        # 不在初始化阶段把所有用户模型都放到 GPU，避免所有用户同时持有 GPU 模型副本
        self.model = model(embed_size)

        self.item_relations = item_relations
        self.total_rounds = total_rounds
        self.ies_learning_rate = 0.01

        # 用户侧：社交 InfoNCE
        self.social_reg = social_reg
        self.contrastive_temp = contrastive_temp
        self.contrastive_neg_num = contrastive_neg_num

        # 物品侧：关系 InfoNCE
        self.item_contrastive_reg = item_contrastive_reg
        self.item_contrastive_temp = item_contrastive_temp
        self.item_contrastive_neg_num = item_contrastive_neg_num

        # IES 模块先放在 CPU，当前用户真正训练或评估时再搬到 GPU
        self.ies_social = IESModule(len(neighbors) + 1, total_rounds).to(torch.device('cpu'))
        self.ies_item = IESModule(len(items) + 1, total_rounds).to(torch.device('cpu'))

        unique_rel_items = set(items)
        for i in items:
            if i in item_relations:
                unique_rel_items.update(item_relations[i][0])

        self.item_rel_nodes_list = list(unique_rel_items)
        self.item_rel_map = {item: idx for idx, item in enumerate(self.item_rel_nodes_list)}
        self.ies_relation = IESModule(len(unique_rel_items), total_rounds).to(torch.device('cpu'))

        self.item_embedding_all_new = None
        self.local_item_feature_cache = {}

        self.user_feature = torch.randn(self.embed_size, device=torch.device('cpu'))
        self.item_feature = None
        self.item_neighbor_feature = None

        self.ies_loss_val = 0.0

        self.rating_max = None
        self.rating_min = None


    def _move_modules_to(self, device):
        if self.model is not None:
            self.model = self.model.to(device)

        self.ies_social = self.ies_social.to(device)
        self.ies_item = self.ies_item.to(device)
        self.ies_relation = self.ies_relation.to(device)

    def release_gpu_cache(self, keep_user_feature=False):
        if self.model is not None:
            try:
                self.model = self.model.to(torch.device('cpu'))
            except Exception:
                pass
            self.model = None

        try:
            self.ies_social = self.ies_social.to(torch.device('cpu'))
            self.ies_item = self.ies_item.to(torch.device('cpu'))
            self.ies_relation = self.ies_relation.to(torch.device('cpu'))
        except Exception:
            pass

        if torch.is_tensor(self.user_feature):
            self.user_feature = self.user_feature.detach()
            if not keep_user_feature:
                self.user_feature = self.user_feature.cpu()

        if torch.is_tensor(self.item_feature):
            self.item_feature = self.item_feature.detach()
            if not keep_user_feature:
                self.item_feature = self.item_feature.cpu()

        if torch.is_tensor(self.item_neighbor_feature):
            self.item_neighbor_feature = self.item_neighbor_feature.detach()
            if not keep_user_feature:
                self.item_neighbor_feature = self.item_neighbor_feature.cpu()

        # 训练结束后可以清空局部预测缓存；验证阶段需要暂时保留
        if not keep_user_feature:
            self.local_item_feature_cache = {}

        self.item_embedding_all_new = None
        self.ies_loss_val = 0.0

    def _cache_local_item_features(self, item_feature):
        self.local_item_feature_cache = {}

        if item_feature is None:
            return

        for local_idx, item_id in enumerate(self.items):
            self.local_item_feature_cache[int(item_id)] = item_feature[local_idx].detach().cpu()

    def _get_item_embedding_from_local_cache(self, item_id, embedding_item):
        item_id = int(item_id)
        device = embedding_item.device

        if item_id in self.local_item_feature_cache:
            return self.local_item_feature_cache[item_id].to(device)

        return embedding_item[item_id]

    # =========================================================
    # 图构造
    # =========================================================
    def _get_star_adj(self, size, device=None):
        if device is None:
            device = self.device

        adj = torch.zeros(size, size, device=device)
        adj[0, 1:] = 1
        adj[1:, 0] = 1
        adj.fill_diagonal_(1)
        return adj

    def _get_item_rel_adj(self, device=None):
        if device is None:
            device = self.device

        size = len(self.item_rel_nodes_list)
        adj = torch.zeros(size, size, device=device)

        for item in self.items:
            if item in self.item_relations:
                idx1 = self.item_rel_map[item]
                neighbors = self.item_relations[item][0]

                for n in neighbors:
                    if n in self.item_rel_map:
                        idx2 = self.item_rel_map[n]
                        adj[idx1, idx2] = 1
                        adj[idx2, idx1] = 1

        adj.fill_diagonal_(1)
        return adj

    def user_embedding(self, embedding):
        device = embedding.device

        neighbor_idx = torch.tensor(self.neighbors, dtype=torch.long, device=device)
        self_idx = torch.tensor([self.id_self], dtype=torch.long, device=device)

        return embedding[neighbor_idx], embedding[self_idx]

    def item_embedding(self, embedding):
        device = embedding.device
        item_idx = torch.tensor(self.items, dtype=torch.long, device=device)
        return embedding[item_idx]

    def item_relations_embedding(self, embedding):
        device = embedding.device

        item_n = []
        for i in self.item_relations:
            item_i = self.item_relations[i][0]
            for j in range(len(item_i)):
                item_n.append(item_i[j])

        item_n_tensor = torch.tensor(item_n, dtype=torch.long, device=device)
        return embedding[item_n_tensor]

    def item_neighbor_len(self, item_relations):
        item_neighbor_len_list = []

        for i in self.item_relations:
            item_i = self.item_relations[i][0]
            item_neighbor_len_list.append([i, len(item_i)])

        return item_neighbor_len_list

    # =========================================================
    # 用户侧：IES 加权朋友原型
    # =========================================================
    def get_weighted_friend_prototype(self, embedding_user, current_round=0):
        if len(self.neighbors) == 0:
            return None

        device = embedding_user.device
        self._move_modules_to(device)

        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        h_social = torch.cat((self_embedding, neighbor_embedding), dim=0)
        adj_social = self._get_star_adj(len(self.neighbors) + 1, device=device)

        refined_adj_s, _ = self.ies_social(adj_social, h_social, current_round)

        weights = refined_adj_s[0, 1:]
        weights = weights / (weights.sum() + 1e-9)

        friend_proto = torch.matmul(weights.unsqueeze(0), neighbor_embedding).squeeze(0)
        return friend_proto

    # =========================================================
    # 用户侧：随机非邻居负样本集合
    # =========================================================
    def get_negative_samples(self, embedding_user):
        all_user_num = embedding_user.shape[0]
        excluded = set(self.neighbors + [self.id_self])
        candidate_ids = [i for i in range(all_user_num) if i not in excluded]

        if len(candidate_ids) == 0:
            return None

        sample_num = min(self.contrastive_neg_num, len(candidate_ids))
        neg_ids = sample(candidate_ids, sample_num)

        neg_ids_tensor = torch.tensor(neg_ids, dtype=torch.long, device=embedding_user.device)
        neg_embedding = embedding_user[neg_ids_tensor]

        return neg_embedding

    # =========================================================
    # 用户侧：社交 InfoNCE
    # =========================================================
    def social_regularization(self, embedding_user, current_round=0):
        if len(self.neighbors) == 0:
            return torch.tensor(0.0, device=embedding_user.device)

        device = embedding_user.device

        if not torch.is_tensor(self.user_feature):
            return torch.tensor(0.0, device=device)

        z_u = self.user_feature.to(device)
        z_u = z_u.squeeze(0) if z_u.dim() > 1 else z_u

        z_pos = self.get_weighted_friend_prototype(embedding_user, current_round)
        z_negs = self.get_negative_samples(embedding_user)

        if z_pos is None or z_negs is None:
            return torch.tensor(0.0, device=device)

        z_u = F.normalize(z_u, dim=0)
        z_pos = F.normalize(z_pos, dim=0)
        z_negs = F.normalize(z_negs, dim=1)

        pos_logit = torch.matmul(z_u, z_pos) / self.contrastive_temp
        neg_logits = torch.matmul(z_negs, z_u) / self.contrastive_temp

        numerator = torch.exp(pos_logit)
        denominator = numerator + torch.sum(torch.exp(neg_logits))

        info_nce_loss = -torch.log(numerator / (denominator + 1e-9))
        return info_nce_loss

    # =========================================================
    # 物品侧：IES 加权物品关系原型
    # =========================================================
    def get_weighted_item_relation_prototypes(self, embedding_item, current_round=0):
        """
        为当前用户交互过的每个 item，构造 IES 加权的物品关系原型。
        返回:
            item_proto_dict: {item_id: prototype_embedding}
        """
        if len(self.items) == 0:
            return {}

        device = embedding_item.device
        self._move_modules_to(device)

        rel_node_tensor = torch.tensor(self.item_rel_nodes_list, dtype=torch.long, device=device)
        all_item_emb = embedding_item[rel_node_tensor]

        adj_rel = self._get_item_rel_adj(device=device)
        refined_adj_rel, _ = self.ies_relation(adj_rel, all_item_emb, current_round)

        item_proto_dict = {}

        for item in self.items:
            if item not in self.item_rel_map:
                continue
            if item not in self.item_relations:
                continue

            rel_neighbors = self.item_relations[item][0]
            valid_neighbors = [n for n in rel_neighbors if n in self.item_rel_map]

            if len(valid_neighbors) == 0:
                continue

            idx_i = self.item_rel_map[item]
            idx_neighbors = torch.tensor(
                [self.item_rel_map[n] for n in valid_neighbors],
                dtype=torch.long,
                device=device
            )

            weights = refined_adj_rel[idx_i, idx_neighbors]
            weights = weights / (weights.sum() + 1e-9)

            valid_neighbor_tensor = torch.tensor(valid_neighbors, dtype=torch.long, device=device)
            neighbor_embs = embedding_item[valid_neighbor_tensor]

            proto = torch.matmul(weights.unsqueeze(0), neighbor_embs).squeeze(0)

            item_proto_dict[item] = proto

        return item_proto_dict

    # =========================================================
    # 物品侧：随机负物品集合
    # =========================================================
    def get_item_negative_samples(self, embedding_item, item_id):
        """
        为某个 item 随机采样一批负物品。
        负样本候选：不属于该 item relation 邻域，且不是自身。
        """
        all_item_num = embedding_item.shape[0]

        rel_neighbors = []
        if item_id in self.item_relations:
            rel_neighbors = self.item_relations[item_id][0]

        excluded = set(rel_neighbors + [item_id])
        candidate_ids = [i for i in range(all_item_num) if i not in excluded]

        if len(candidate_ids) == 0:
            return None

        sample_num = min(self.item_contrastive_neg_num, len(candidate_ids))
        neg_ids = sample(candidate_ids, sample_num)

        neg_ids_tensor = torch.tensor(neg_ids, dtype=torch.long, device=embedding_item.device)
        neg_embedding = embedding_item[neg_ids_tensor]

        return neg_embedding

    # =========================================================
    # 物品侧：关系 InfoNCE
    # =========================================================
    def item_relation_contrastive_loss(self, embedding_item, current_round=0):
        """
        对当前用户交互过的 item 做物品关系 InfoNCE：
        - anchor: 当前 item 表示
        - positive: IES 加权物品关系原型
        - negatives: 随机负物品
        """
        if len(self.items) == 0:
            return torch.tensor(0.0, device=embedding_item.device)

        if not hasattr(self, 'item_feature') or self.item_feature is None:
            return torch.tensor(0.0, device=embedding_item.device)

        device = embedding_item.device

        item_proto_dict = self.get_weighted_item_relation_prototypes(embedding_item, current_round)

        if len(item_proto_dict) == 0:
            return torch.tensor(0.0, device=device)

        losses = []

        item_feature = self.item_feature.to(device)

        for local_idx, item in enumerate(self.items):
            if item not in item_proto_dict:
                continue

            z_item = item_feature[local_idx]
            z_pos = item_proto_dict[item]
            z_negs = self.get_item_negative_samples(embedding_item, item)

            if z_negs is None:
                continue

            z_item = F.normalize(z_item, dim=0)
            z_pos = F.normalize(z_pos, dim=0)
            z_negs = F.normalize(z_negs, dim=1)

            pos_logit = torch.matmul(z_item, z_pos) / self.item_contrastive_temp
            neg_logits = torch.matmul(z_negs, z_item) / self.item_contrastive_temp

            numerator = torch.exp(pos_logit)
            denominator = numerator + torch.sum(torch.exp(neg_logits))

            loss_i = -torch.log(numerator / (denominator + 1e-9))
            losses.append(loss_i)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        return torch.mean(torch.stack(losses))

    def GNN(self, embedding_user, embedding_item, sampled_items, embedding_item_relations, current_round=0):
        device = embedding_user.device

        if self.model is None:
            raise RuntimeError('Local model is None. Please call update_local_GNN before user.train().')

        self._move_modules_to(device)

        # IES: 社交图
        neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
        h_social = torch.cat((self_embedding, neighbor_embedding), dim=0)
        adj_social = self._get_star_adj(len(self.neighbors) + 1, device=device)
        refined_adj_s, loss_s = self.ies_social(adj_social, h_social, current_round)
        adj_social_weights = refined_adj_s[0, 1:].unsqueeze(1)

        # IES: 交互图
        items_embedding = self.item_embedding(embedding_item)

        if len(self.items) > 0:
            h_item = torch.cat((self_embedding, items_embedding), dim=0)
            adj_item = self._get_star_adj(len(self.items) + 1, device=device)
            refined_adj_i, loss_i = self.ies_item(adj_item, h_item, current_round)
            adj_interact_weights = refined_adj_i[0, 1:].unsqueeze(1)
        else:
            loss_i = 0
            adj_interact_weights = None

        # IES: 物品关系图
        rel_node_tensor = torch.tensor(self.item_rel_nodes_list, dtype=torch.long, device=device)
        all_item_emb = embedding_item[rel_node_tensor]

        adj_rel = self._get_item_rel_adj(device=device)
        refined_adj_rel, loss_rel = self.ies_relation(adj_rel, all_item_emb, current_round)

        adj_rel_list = []

        if len(self.items) > 0:
            for i in self.item_relations:
                if i in self.item_rel_map:
                    idx1 = self.item_rel_map[i]
                    neighbors = self.item_relations[i][0]

                    for n in neighbors:
                        if n in self.item_rel_map:
                            idx2 = self.item_rel_map[n]
                            w = refined_adj_rel[idx1, idx2]
                            adj_rel_list.append(w)

            if len(adj_rel_list) > 0:
                adj_item_corr_weights = torch.stack(adj_rel_list).unsqueeze(1)
            else:
                adj_item_corr_weights = None
        else:
            adj_item_corr_weights = None
            loss_rel = 0

        self.ies_loss_val = loss_s + loss_i + loss_rel
        refined_adjs = (adj_social_weights, adj_interact_weights, adj_item_corr_weights)

        item_relations_embedding = self.item_relations_embedding(embedding_item_relations)
        item_neighbor_len_list = self.item_neighbor_len(self.item_relations)

        if len(self.items) > 0:
            user_feature, item_feature, item_neighbor_feature = self.model(
                self_embedding,
                neighbor_embedding,
                items_embedding,
                item_relations_embedding,
                item_neighbor_len_list,
                refined_adjs=refined_adjs
            )

            self.user_feature = user_feature
            self.item_feature = item_feature
            self.item_neighbor_feature = item_neighbor_feature.detach()

            self._cache_local_item_features(self.item_feature)

            sampled_items_tensor = torch.tensor(sampled_items, dtype=torch.long, device=device)
            sampled_items_embedding = embedding_item[sampled_items_tensor]

            items_embedding_with_sampled = torch.cat(
                (self.item_feature, sampled_items_embedding),
                dim=0
            )
        else:
            user_feature = self.model(
                self_embedding,
                neighbor_embedding,
                items_embedding,
                item_relations_embedding,
                item_neighbor_len_list,
                refined_adjs=refined_adjs
            )

            self.user_feature = user_feature
            self.local_item_feature_cache = {}

            sampled_items_tensor = torch.tensor(sampled_items, dtype=torch.long, device=device)
            sampled_items_embedding = embedding_item[sampled_items_tensor]

            items_embedding_with_sampled = torch.cat(
                (items_embedding, sampled_items_embedding),
                dim=0
            )

        predicted = torch.matmul(user_feature, items_embedding_with_sampled.t())
        return predicted

    def update_local_GNN(self, global_model, rating_max, rating_min, embedding_user, embedding_item,
                         embedding_item_relations, current_round=0, keep_model=True):

        device = embedding_user.device

        self.model = copy.deepcopy(global_model).to(device)
        self.rating_max = rating_max
        self.rating_min = rating_min

        self._move_modules_to(device)

        with torch.no_grad():
            neighbor_embedding, self_embedding = self.user_embedding(embedding_user)
            items_embedding = self.item_embedding(embedding_item)
            item_relations_embedding = self.item_relations_embedding(embedding_item_relations)
            item_neighbor_len_list = self.item_neighbor_len(self.item_relations)

            h_social = torch.cat((self_embedding, neighbor_embedding), dim=0)
            adj_social = self._get_star_adj(len(self.neighbors) + 1, device=device)
            refined_adj_s, _ = self.ies_social(adj_social, h_social, current_round)
            adj_social_weights = refined_adj_s[0, 1:].unsqueeze(1)

            adj_interact_weights = None
            adj_item_corr_weights = None

            if len(self.items) > 0:
                h_item = torch.cat((self_embedding, items_embedding), dim=0)
                adj_item = self._get_star_adj(len(self.items) + 1, device=device)
                refined_adj_i, _ = self.ies_item(adj_item, h_item, current_round)
                adj_interact_weights = refined_adj_i[0, 1:].unsqueeze(1)

                rel_node_tensor = torch.tensor(self.item_rel_nodes_list, dtype=torch.long, device=device)
                all_item_emb = embedding_item[rel_node_tensor]

                adj_rel = self._get_item_rel_adj(device=device)
                refined_adj_rel, _ = self.ies_relation(adj_rel, all_item_emb, current_round)

                adj_rel_list = []

                for i in self.item_relations:
                    if i in self.item_rel_map:
                        idx1 = self.item_rel_map[i]
                        neighbors = self.item_relations[i][0]

                        for n in neighbors:
                            if n in self.item_rel_map:
                                idx2 = self.item_rel_map[n]
                                adj_rel_list.append(refined_adj_rel[idx1, idx2])

                if len(adj_rel_list) > 0:
                    adj_item_corr_weights = torch.stack(adj_rel_list).unsqueeze(1)

            refined_adjs = (adj_social_weights, adj_interact_weights, adj_item_corr_weights)

            if len(self.items) > 0:
                items_embedding = self.item_embedding(embedding_item)
            else:
                items_embedding = False

            if len(self.items) > 0:
                user_feature, item_feature, item_neighbor_feature = self.model(
                    self_embedding,
                    neighbor_embedding,
                    items_embedding,
                    item_relations_embedding,
                    item_neighbor_len_list,
                    refined_adjs=refined_adjs
                )

                self.user_feature = user_feature.detach()
                self.item_feature = item_feature.detach()
                self.item_neighbor_feature = item_neighbor_feature.detach()

                self._cache_local_item_features(self.item_feature)
            else:
                user_feature = self.model(
                    self_embedding,
                    neighbor_embedding,
                    items_embedding,
                    item_relations_embedding,
                    item_neighbor_len_list,
                    refined_adjs=refined_adjs
                )

                self.user_feature = user_feature.detach()
                self.item_feature = None
                self.item_neighbor_feature = None
                self.local_item_feature_cache = {}

        if not keep_model:
            self.release_gpu_cache(keep_user_feature=True)

    def predict(self, item_id, embedding_user, embedding_item):
        with torch.no_grad():
            device = embedding_item.device

            if not torch.is_tensor(self.user_feature):
                raise RuntimeError('user_feature is not available. Please call update_local_GNN before predict().')

            user_feature = self.user_feature.to(device)
            item_embedding = self._get_item_embedding_from_local_cache(item_id, embedding_item)

            score = torch.matmul(user_feature, item_embedding.t())

        return score.detach().cpu().item()

    def loss(self, predicted, sampled_rating, embedding_user=None, embedding_item=None, current_round=0):
        if sampled_rating.dim() > 1:
            sampled_rating = sampled_rating.view(-1)

        if predicted.dim() > 1:
            predicted = predicted.view(-1)

        true_label = torch.cat(
            (
                torch.tensor(self.ratings).float().to(sampled_rating.device),
                sampled_rating
            )
        )

        pred_loss = torch.sqrt(torch.mean((predicted - true_label) ** 2))

        total_loss = pred_loss + self.ies_loss_val

        if embedding_user is not None and self.social_reg > 0:
            social_loss = self.social_regularization(embedding_user, current_round)
            total_loss = total_loss + self.social_reg * social_loss

        if embedding_item is not None and self.item_contrastive_reg > 0:
            item_cl_loss = self.item_relation_contrastive_loss(embedding_item, current_round)
            total_loss = total_loss + self.item_contrastive_reg * item_cl_loss

        return total_loss

    def negative_sample_item(self, embedding_item):
        file_path = '/home/cwmlb/Documents/dbb/CUFL-RG(epinions课+对)/data/epinions/q.npy'

        q = np.load(file_path, allow_pickle=True)
        q_list = q.tolist()
        counter_list = list(enumerate(q_list, 0))
        counter_list = np.array(counter_list)

        labels = []

        for i in self.items:
            value = q[int(i)]
            labels.append(value)

        data = collections.Counter(labels)
        data_list = dict(data)

        if len(data.values()) > 0:
            max_value = max(list(data.values()))
            mode_val = [num for num, freq in data_list.items() if freq == max_value]
        else:
            mode_val = []

        if len(mode_val) == 1:
            a = mode_val[0]
            item = []

            for j in counter_list:
                b = j[1]

                if b == a:
                    if b not in self.items:
                        item.append(j[0])
                    else:
                        continue
                else:
                    continue

            if self.negative_sample > len(item):
                self.negative_sample = len(item)
                sampled_items = sample(item, self.negative_sample)
            else:
                sampled_items = sample(item, self.negative_sample)
        else:
            item_num = embedding_item.shape[0]
            ls = [i for i in range(item_num) if i not in self.items]
            sampled_items = sample(ls, self.negative_sample)

        sampled_items_tensor = torch.tensor(sampled_items, dtype=torch.long, device=embedding_item.device)
        sampled_item_embedding = embedding_item[sampled_items_tensor]

        user_feature = self.user_feature.to(embedding_item.device)

        predicted = torch.matmul(user_feature, sampled_item_embedding.t())
        predicted = torch.round(torch.clip(predicted, min=self.rating_min, max=self.rating_max))

        return sampled_items, predicted

    def LDP(self, tensor):
        """
        Local Differential Privacy mechanism for uploaded gradients.

        The mechanism follows:
            M(g) = clip(g, C) + Laplace(0, 2C / epsilon)

        where:
            C       = self.clip
            epsilon = self.ldp_epsilon

        Each released gradient coordinate satisfies epsilon-LDP.
        For multiple released coordinates, the overall privacy budget follows
        the standard sequential composition property.
        """

        epsilon = getattr(self, "ldp_epsilon", None)

        # Move to CPU and detach from computation graph.
        # This keeps the same behavior as the original code, where uploaded
        # gradients are returned as CPU tensors for server-side aggregation.
        tensor = tensor.detach().cpu()

        # Clip each gradient coordinate into [-C, C].
        clipped_tensor = torch.clamp(tensor, min=-self.clip, max=self.clip)

        # No-LDP setting: only return clipped gradients.
        # You can use --ldp_epsilon inf or --ldp_epsilon 1e12 as an approximate no-noise setting.
        if epsilon is None or epsilon == float("inf") or epsilon >= 1e12:
            return clipped_tensor

        if epsilon <= 0:
            raise ValueError("ldp_epsilon must be positive. Use inf or 1e12 for the No-LDP setting.")

        # Sensitivity of one clipped gradient coordinate is bounded by 2C.
        noise_scale = 2.0 * self.clip / epsilon

        # Add independent Laplace noise to each coordinate.
        noise = torch.distributions.Laplace(
            loc=torch.zeros_like(clipped_tensor),
            scale=torch.full_like(clipped_tensor, noise_scale)
        ).sample()

        return clipped_tensor + noise

    def train(self, embedding_user, embedding_item, embedding_item_relations, current_round=0):
        device = embedding_user.device

        if self.model is None:
            raise RuntimeError('Local model is None. Please call update_local_GNN before user.train().')

        self._move_modules_to(device)

        embedding_user = torch.clone(embedding_user).detach()
        embedding_item = torch.clone(embedding_item).detach()
        embedding_item_relations = torch.clone(embedding_item_relations).detach()

        embedding_user.requires_grad = True
        embedding_item.requires_grad = True

        self.model.train()

        sampled_items, sampled_rating = self.negative_sample_item(embedding_item)
        returned_items = self.items + sampled_items

        predicted = self.GNN(
            embedding_user,
            embedding_item,
            sampled_items,
            embedding_item_relations,
            current_round
        )

        loss = self.loss(
            predicted,
            sampled_rating,
            embedding_user,
            embedding_item,
            current_round
        )

        self.model.zero_grad(set_to_none=True)
        self.ies_social.zero_grad(set_to_none=True)
        self.ies_item.zero_grad(set_to_none=True)
        self.ies_relation.zero_grad(set_to_none=True)

        loss.backward()

        with torch.no_grad():
            if self.ies_social.S_k.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.ies_social.S_k], max_norm=1.0)
                self.ies_social.S_k -= self.ies_learning_rate * self.ies_social.S_k.grad

            if self.ies_item.S_k.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.ies_item.S_k], max_norm=1.0)
                self.ies_item.S_k -= self.ies_learning_rate * self.ies_item.S_k.grad

            if self.ies_relation.S_k.grad is not None:
                torch.nn.utils.clip_grad_norm_([self.ies_relation.S_k], max_norm=1.0)
                self.ies_relation.S_k -= self.ies_learning_rate * self.ies_relation.S_k.grad

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        model_grad = []

        for param in list(self.model.parameters()):
            if param.grad is None:
                grad = torch.zeros_like(param, device=torch.device('cpu'))
            else:
                grad = self.LDP(param.grad)

            # 聚合前先放 CPU，避免 parameter_list 长时间占用 GPU 显存
            model_grad.append(grad.cpu())

        if embedding_item.grad is None:
            item_grad = torch.zeros(
                len(returned_items),
                self.embed_size,
                dtype=torch.float32
            )
        else:
            item_grad = self.LDP(embedding_item.grad[returned_items, :]).cpu()

        returned_users = self.neighbors + [self.id_self]

        if embedding_user.grad is None:
            user_grad = torch.zeros(
                len(returned_users),
                self.embed_size,
                dtype=torch.float32
            )
        else:
            user_grad = self.LDP(embedding_user.grad[returned_users, :]).cpu()

        loss_value = float(loss.detach().cpu())

        res = (
            model_grad,
            item_grad,
            user_grad,
            returned_items,
            returned_users,
            loss_value
        )

        self.model.zero_grad(set_to_none=True)
        self.ies_social.zero_grad(set_to_none=True)
        self.ies_item.zero_grad(set_to_none=True)
        self.ies_relation.zero_grad(set_to_none=True)

        del embedding_user
        del embedding_item
        del embedding_item_relations
        del sampled_rating
        del predicted
        del loss

        # 当前用户训练结束后释放 GPU 模型副本和大张量缓存
        # local_item_feature_cache 也清空，因为训练阶段不需要长期保留预测缓存
        self.release_gpu_cache(keep_user_feature=False)

        return res