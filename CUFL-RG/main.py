import pickle
import pandas as pd
import torch
import numpy as np
from user import user
from server import server
from sklearn import metrics
import math
import argparse
import warnings
import sys
import faulthandler
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import random
import os
import copy

faulthandler.enable()
warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 保证 cudnn 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description="args for FedGNN")
parser.add_argument('--embed_size', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--data', default='epinions')
parser.add_argument('--user_batch', type=int, default=256)
parser.add_argument('--clip', type=float, default=0.1)
# parser.add_argument('--laplace_lambda', type=float, default=0.1)
parser.add_argument('--ldp_epsilon', type=float, default=3)
parser.add_argument('--negative_sample', type=int, default=10)
parser.add_argument('--valid_step', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--rounds', type=int, default=1500)

# 用户侧：社交 InfoNCE
parser.add_argument('--social_reg', type=float, default=0.03)
parser.add_argument('--contrastive_temp', type=float, default=0.1)
parser.add_argument('--contrastive_neg_num', type=int, default=16)

# 物品侧：关系 InfoNCE
parser.add_argument('--item_contrastive_reg', type=float, default=0.01)
parser.add_argument('--item_contrastive_temp', type=float, default=0.175)
parser.add_argument('--item_contrastive_neg_num', type=int, default=8)

# 新增：支持一次跑多个随机种子
parser.add_argument('--seeds', type=str, default='42,52,62,72,82',
                    help='5个随机种子，用逗号分隔，例如 42,52,62,72,82')
parser.add_argument('--result_path', type=str, default='./multi_seed_results.csv',
                    help='保存多随机种子结果的 csv 路径')

args = parser.parse_args()

embed_size = args.embed_size
user_batch = args.user_batch
lr = args.lr
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


def processing_valid_data(valid_data):
    res = []
    for key in valid_data.keys():
        if len(valid_data[key]) > 0:
            for ratings in valid_data[key]:
                item, rate = ratings
                res.append((int(key), int(item), rate))
    return np.array(res)


def loss(server_obj, valid_data,current_round=0):
    label = valid_data[:, -1]
    predicted = server_obj.predict(valid_data, current_round=current_round)
    mae = sum(abs(label - predicted)) / len(label)
    rmse = math.sqrt(sum((label - predicted) ** 2) / len(label))
    return mae, rmse


def load_data(args):
    data_path = f'/home/cwmlb/Documents/dbb/CUFL-RG(epinions课+对)/data/epinions/{args.data}_FedMF.pkl'
    with open(data_path, 'rb') as data_file:
        dataset = pickle.load(data_file)

    train_data = dataset[0]
    valid_data = dataset[1]
    test_data = dataset[2]
    user_id_list = dataset[3]
    item_id_list = dataset[4]
    social = dataset[5]
    knowledge = dataset[6]

    valid_data = processing_valid_data(valid_data)
    test_data = processing_valid_data(test_data)
    train_data_verify = processing_valid_data(train_data)

    return train_data, valid_data, test_data, user_id_list, item_id_list, social, knowledge


def build_user_list(train_data, user_id_list, social, knowledge, embed_size, args):
    rating_max = -9999
    rating_min = 9999
    user_list = []

    for u in user_id_list:
        ratings = train_data[u]
        items = []
        rating = []
        item_relations = {}
        list1 = []

        for i in range(len(ratings)):
            item, rate = ratings[i]
            items.append(item)
            rating.append(rate)

        if len(rating) > 0:
            rating_max = max(rating_max, max(rating))
            rating_min = min(rating_min, min(rating))

        for i in items:
            list1.append(knowledge[i])
            item_relations[i] = list1
            list1 = []

        user_list.append(
            user(
                u,
                items,
                rating,
                list(social[u]),
                embed_size,
                args.clip,
                args.ldp_epsilon,
                args.negative_sample,
                item_relations,
                total_rounds=args.rounds,
                social_reg=args.social_reg,
                contrastive_temp=args.contrastive_temp,
                contrastive_neg_num=args.contrastive_neg_num,
                item_contrastive_reg=args.item_contrastive_reg,
                item_contrastive_temp=args.item_contrastive_temp,
                item_contrastive_neg_num=args.item_contrastive_neg_num
            )
        )

    return user_list, rating_max, rating_min


def run_once(seed, args):
    print('=' * 80)
    print(f'开始运行随机种子 seed = {seed}')
    print('=' * 80)

    # 1. 固定随机种子
    set_seed(seed)

    # 2. 每个 seed 都重新加载数据、重新初始化模型
    train_data, valid_data, test_data, user_id_list, item_id_list, social, knowledge = load_data(args)

    user_list, rating_max, rating_min = build_user_list(
        train_data, user_id_list, social, knowledge, embed_size, args
    )

    server_obj = server(
        user_list, user_batch, user_id_list, item_id_list,
        embed_size, lr, device, rating_max, rating_min, args.weight_decay
    )

    count = 0
    initial_lr = lr
    decay_rate = 0.9
    decay_interval = 100
    rmse_best = 9999
    epoch_counter = 0

    # 记录“最佳验证集”对应的测试结果
    best_valid_mae = None
    best_valid_rmse = None
    best_test_mae = None
    best_test_rmse = None
    best_epoch = None

    while True:
        for i in range(args.valid_step):
            server_obj.train(current_round=epoch_counter)
            epoch_counter += 1

            if epoch_counter % 10 == 0 and epoch_counter != 0:
                server_obj.save_model_weights()

        # 学习率衰减逻辑
        if (epoch_counter + 1) % decay_interval == 0:
            print('valid for lr decay check')
            mae_tmp, rmse_tmp = loss(server_obj, valid_data, current_round=epoch_counter)
            current_lr = initial_lr * (decay_rate ** (epoch_counter // decay_interval))
            initial_lr = current_lr

            if rmse_tmp < rmse_best:
                rmse_best = rmse_tmp
            else:
                decay_interval += 5
                print("Epoch {}, Current Learning Rate: {}".format(epoch_counter, current_lr))

        # 每个 valid_step 后评估一次
        print('valid')
        mae, rmse = loss(server_obj, valid_data, current_round=epoch_counter)
        print('valid mae: {}, valid rmse: {}'.format(mae, rmse))

        print('test')
        mae_test, rmse_test = loss(server_obj, test_data, current_round=epoch_counter)
        print('test mae: {}, test rmse: {}'.format(mae_test, rmse_test))

        # 保存最佳验证集对应的测试结果
        if rmse < rmse_best:
            rmse_best = rmse
            count = 0

            best_valid_mae = mae
            best_valid_rmse = rmse
            best_test_mae = mae_test
            best_test_rmse = rmse_test
            best_epoch = epoch_counter

            print(f'[seed {seed}] 更新最优结果: epoch={best_epoch}, '
                  f'best_valid_rmse={best_valid_rmse:.6f}, best_test_rmse={best_test_rmse:.6f}')
        else:
            count += 1

        if count > 20:
            print('not improved for 20 evals, stop training')
            break

    print(f'[seed {seed}] 最终保存结果（最佳验证集对应）:')
    print('best epoch: {}'.format(best_epoch))
    print('best valid mae: {}, best valid rmse: {}'.format(best_valid_mae, best_valid_rmse))
    print('best test mae: {}, best test rmse: {}'.format(best_test_mae, best_test_rmse))

    return {
        'seed': seed,
        'best_epoch': best_epoch,
        'best_valid_mae': best_valid_mae,
        'best_valid_rmse': best_valid_rmse,
        'best_test_mae': best_test_mae,
        'best_test_rmse': best_test_rmse
    }


if __name__ == '__main__':
    # 解析 5 个随机种子
    seed_list = [int(x.strip()) for x in args.seeds.split(',') if x.strip() != '']

    all_results = []

    for seed in seed_list:
        result = run_once(seed, args)
        all_results.append(result)

    # 保存每个 seed 的结果
    df = pd.DataFrame(all_results)
    df.to_csv(args.result_path, index=False)

    # 计算均值和标准差
    summary = {
        'best_valid_mae_mean': df['best_valid_mae'].mean(),
        'best_valid_mae_std': df['best_valid_mae'].std(),
        'best_valid_rmse_mean': df['best_valid_rmse'].mean(),
        'best_valid_rmse_std': df['best_valid_rmse'].std(),
        'best_test_mae_mean': df['best_test_mae'].mean(),
        'best_test_mae_std': df['best_test_mae'].std(),
        'best_test_rmse_mean': df['best_test_rmse'].mean(),
        'best_test_rmse_std': df['best_test_rmse'].std(),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = args.result_path.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    print('=' * 80)
    print('5个随机种子结果如下：')
    print(df)
    print('=' * 80)
    print('均值和标准差如下：')
    print(summary_df)
    print('=' * 80)
    print(f'详细结果已保存到: {args.result_path}')
    print(f'统计结果已保存到: {summary_path}')


