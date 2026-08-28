import torch
import numpy as np
import math
import gc
from random import sample
from model import model


torch.multiprocessing.set_sharing_strategy('file_system')


class server():
    def __init__(self, user_list, user_batch, users, items, embed_size, lr, device,
                 rating_max, rating_min, weight_decay):

        self.device = device if torch.cuda.is_available() else torch.device('cpu')

        self.user_list_with_coldstart = user_list
        self.user_list = self.generate_user_list(self.user_list_with_coldstart)

        # 防止 user_batch 大于可训练用户数量时报错
        self.batch_size = min(user_batch, len(self.user_list))

        self.user_embedding = torch.randn(
            len(users),
            embed_size,
            device=self.device
        )

        self.item_embedding = torch.randn(
            len(items),
            embed_size,
            device=self.device
        )

        self.model = model(embed_size).to(self.device)

        self.lr = lr
        self.rating_max = rating_max
        self.rating_min = rating_min
        self.weight_decay = weight_decay

        # 不再初始化时给所有用户下发模型，避免所有用户都持有 GPU 模型副本
        # self.distribute(self.user_list_with_coldstart, current_round=0)

    def generate_user_list(self, user_list_with_coldstart):
        ls = []

        for user in user_list_with_coldstart:
            if len(user.items) > 0:
                ls.append(user)

        return ls

    def aggregator(self, parameter_list):
        flag = False
        number = 0

        gradient_item = torch.zeros_like(self.item_embedding, device=self.device)
        gradient_user = torch.zeros_like(self.user_embedding, device=self.device)

        loss = 0.0

        item_count = torch.zeros(
            self.item_embedding.shape[0],
            device=self.device
        )

        user_count = torch.zeros(
            self.user_embedding.shape[0],
            device=self.device
        )

        gradient_model = None

        for parameter in parameter_list:
            model_grad, item_grad, user_grad, returned_items, returned_users, loss_user = parameter

            num = len(returned_items)

            item_count[returned_items] += 1
            user_count[returned_users] += num

            loss += (loss_user ** 2) * num
            number += num

            item_grad = item_grad.to(self.device)
            user_grad = user_grad.to(self.device)

            if not flag:
                flag = True

                gradient_model = []

                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num

                for i in range(len(model_grad)):
                    gradient_model.append(model_grad[i].to(self.device) * num)
            else:
                gradient_item[returned_items, :] += item_grad * num
                gradient_user[returned_users, :] += user_grad * num

                for i in range(len(model_grad)):
                    gradient_model[i] += model_grad[i].to(self.device) * num

            del item_grad
            del user_grad

        loss = math.sqrt(loss / number)
        print('training average loss:', loss)

        item_count[item_count == 0] = 1
        user_count[user_count == 0] = 1

        gradient_item /= item_count.unsqueeze(1)
        gradient_user /= user_count.unsqueeze(1)

        for i in range(len(gradient_model)):
            gradient_model[i] = gradient_model[i] / number

        return gradient_model, gradient_item, gradient_user

    def distribute(self, users, current_round=0, keep_model=True):
        for user in users:
            user.update_local_GNN(
                self.model,
                self.rating_max,
                self.rating_min,
                self.user_embedding,
                self.item_embedding,
                self.item_embedding,
                current_round=current_round,
                keep_model=keep_model
            )

    def predict(self, valid_data, current_round=0):
        users = valid_data[:, 0]
        items = valid_data[:, 1]
        res = []

        unique_user_ids = sorted(list(set([int(i) for i in users])))
        unique_users = [
            self.user_list_with_coldstart[int(i)]
            for i in unique_user_ids
        ]

        # 评估只需要刷新 user_feature，不需要长期保留每个用户的本地模型副本
        self.distribute(
            unique_users,
            current_round=current_round,
            keep_model=False
        )

        with torch.no_grad():
            for i in range(len(users)):
                u_id = int(users[int(i)])
                item_id = int(items[int(i)])

                res_temp = self.user_list_with_coldstart[u_id].predict(
                    item_id,
                    self.user_embedding,
                    self.item_embedding
                )

                res.append(float(res_temp))

        # 预测结束后释放评估用户的 GPU 缓存
        for u in unique_users:
            if hasattr(u, 'release_gpu_cache'):
                u.release_gpu_cache(keep_user_feature=False)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return np.array(res)

    def save_model_weights(self):
        torch.save(self.model.state_dict(), './results/model_weights.pth')

    def train(self, current_round=0):
        parameter_list = []

        users = sample(self.user_list, self.batch_size)

        # 逐个用户下发模型并训练，避免一个 batch 内 256 个用户同时持有 GPU 模型副本
        for user in users:
            self.distribute(
                [user],
                current_round=current_round,
                keep_model=True
            )

            res = user.train(
                self.user_embedding,
                self.item_embedding,
                self.item_embedding,
                current_round=current_round
            )

            parameter_list.append(res)

        # 全局聚合
        gradient_model, gradient_item, gradient_user = self.aggregator(parameter_list)

        with torch.no_grad():
            # 更新全局模型参数
            ls_global_param = list(self.model.parameters())

            for param_idx in range(len(ls_global_param)):
                grad = gradient_model[param_idx].to(self.device)

                ls_global_param[param_idx].data = (
                    ls_global_param[param_idx].data
                    - self.lr * grad
                    - self.weight_decay * ls_global_param[param_idx].data
                )

            # 更新全局 item / user embedding
            item_index = gradient_item.sum(dim=-1) != 0
            user_index = gradient_user.sum(dim=-1) != 0

            self.item_embedding[item_index] = (
                self.item_embedding[item_index]
                - self.lr * gradient_item[item_index]
                - self.weight_decay * self.item_embedding[item_index]
            )

            self.user_embedding[user_index] = (
                self.user_embedding[user_index]
                - self.lr * gradient_user[user_index]
                - self.weight_decay * self.user_embedding[user_index]
            )

        del parameter_list
        del gradient_model
        del gradient_item
        del gradient_user

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()