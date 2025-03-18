'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from time import time
from tqdm import tqdm
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import sklearn.preprocessing
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import json
import matplotlib.pyplot as plt


import os



CORES = multiprocessing.cpu_count() // 2


def Certainty_aware_train(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None,sample=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.Loss = loss_class

    S = utils.UniformNegativeSample(dataset,sample)

    users_np = S[:, 0]
    posItems_np = S[:, 1]
    negItems_np = S[:, 2]

    prob = sample[-1]
    
    users, posItems, negItems, prob = torch.Tensor(users_np).long().to(world.device), torch.Tensor(posItems_np).long().to(world.device), torch.Tensor(negItems_np).long().to(world.device), torch.FloatTensor(prob).to(world.device)
    users, posItems, negItems, prob = utils.shuffle(users, posItems, negItems, prob)

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    if world.model_name == "sgl":
        recommend_model.set_subgraph()

    for (batch_i,
            (batch_users,
            batch_pos,
            batch_neg,
            batch_prob)) in enumerate(utils.minibatch(users,
                                                    posItems,
                                                    negItems,
                                                    prob,
                                                    batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, prob = batch_prob)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()

    return f"loss{aver_loss:.3f}-{time_info}"


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None,penalty=False):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.Loss = loss_class
    
    with timer(name="Sample"):

        S = utils.UniformSample_original(dataset)
    
    users_np = S[:, 0]
    posItems_np = S[:, 1]
    negItems_np = S[:, 2]
    
    users, posItems, negItems = torch.Tensor(users_np).long().to(world.device), torch.Tensor(posItems_np).long().to(world.device), torch.Tensor(negItems_np).long().to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.


    losses = []

    if world.model_name == "sgl":
        recommend_model.set_subgraph()

    for (batch_i,
            (batch_users,
            batch_pos,
            batch_neg)) in enumerate(utils.minibatch(users,
                                                    posItems,
                                                    negItems,
                                                    batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        aver_loss += cri

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()

    return f"loss{aver_loss:.3f}-{time_info}"


# サンプル選択のための関数
def caluculate_criterion(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):

    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.Loss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    
    users_np = S[:, 0]
    posItems_np = S[:, 1]
    negItems_np = S[:, 2]
    
    co_occurrences = []
    for i in range(len(users_np)):
        co_occurrence = dataset.get_c_ui(users_np[i], posItems_np[i])
        co_occurrences.append(co_occurrence)

    users, posItems, negItems = torch.Tensor(users_np).long().to(world.device), torch.Tensor(posItems_np).long().to(world.device), torch.Tensor(negItems_np).long().to(world.device)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.


    criterions = []
    combined_embs = np.empty((0, 8, 64))

    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          )) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   
                                                   batch_size=world.config['bpr_batch_size'])):

        features = Recmodel.get_sample_features(batch_users,batch_pos,batch_neg)            
        cs_np = features.to('cpu').detach().numpy()
        cs = cs_np.copy().tolist()
        criterions.extend(cs)

    criterions_np = np.array(criterions)
    co_occurrences_np = np.array(co_occurrences)

    criterions_np = np.column_stack((criterions_np, co_occurrences_np))

    column_mapping = {"score": 0, "cui": 1}

    # 取り出したい列名のリスト
    columns_to_extract = world.config['gmm_input']

    # 対応するインデックスを辞書から取得
    indices = [column_mapping[col] for col in columns_to_extract]

    criterions_np = criterions_np[:, indices]


    # scaling
    scaler = preprocessing.MinMaxScaler()
    criterions_np = scaler.fit_transform(criterions_np)

    
    return users_np, posItems_np, negItems_np, criterions_np#, is_noisy

def calculate_certainty(criterions):
    d1 = False
    if criterions.shape[1] == 1:
        d1 = True

    input_criterions = np.array(criterions)


    gmm = GaussianMixture(n_components=2,max_iter=100,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_criterions)
    prob = gmm.predict_proba(input_criterions)

    if not d1:

        # L2ノルム（ユークリッド距離）の計算
        norms = np.linalg.norm(gmm.means_, axis=1)
        # 最大ノルムのインデックスを取得
        max_norm_index = np.argmax(norms)
        prob = prob[:,max_norm_index]
    
    #1次元の場合
    else:
        prob = prob[:,gmm.means_.argmax()]


    return utils.scale_probabilities(prob)


def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]


    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []

    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue,r,k))


    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}


def Base_Test(dataset,Recmodel,epoch,testDict=None,dualnet=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset: utils.BasicDataset
    testDict: dict


    # eval mode with no dropout


    Recmodel = Recmodel.eval()
    if dualnet:
        dualnet.eval()

    max_K = max(world.topks)


    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        category_list = []
        # auc_record = []
        # ratings = []


        total_batch = len(users) // u_batch_size + 1
        for batch_users in tqdm(utils.minibatch(users, batch_size=u_batch_size),total=total_batch):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)
        
            rating = Recmodel.getUsersRating(batch_users_gpu,get_emb=False)

            # 2つのネットワークの予測スコアの平均値を利用する場合
            if dualnet and world.config['ensemble_prediction']:
                dualnet_rating = dualnet.getUsersRating(batch_users_gpu,get_emb=False)

                rating = torch.add(rating,dualnet_rating)

            

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)


            rating[exclude_index, exclude_items] = -(1<<10)
            scores, rating_K = torch.topk(rating, k=max_K)
            

            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())


            groundTrue_list.append(groundTrue)

        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)

        pre_results = []
        for x in X:
            pre_results.append(test_one_batch(x))
        #scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        
        # print(results["entropy"])

        # print(len(users))

        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        print(results)
    return results


def Val(dataset, Recmodel, epoch, w=None, multicore=0,dualnet=None):
    return Base_Test(dataset,Recmodel,epoch,dataset.valDict,dualnet=dualnet)

def Test(dataset, Recmodel, epoch, w=None, multicore=0,dualnet=None):
    return Base_Test(dataset,Recmodel,epoch,dataset.testDict,dualnet=dualnet)
