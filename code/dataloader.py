"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Shuxian Bi (stanbi@mail.ustc.edu.cn),Jianbai Ye (gusye@mail.ustc.edu.cn)
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import numpy as np
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, lil_matrix, save_npz, load_npz
import scipy.sparse as sp
import world
from world import cprint
from time import time


from tqdm import tqdm
import matplotlib.pyplot as plt

#import dgl
import random

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/lastfm"):
        # train or test
        cprint("loading [last fm]")
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        # self.n_users = 1892
        # self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = np.array(trainData[:][1])
        # self.trainDataSize = len(self.trainUser)
        self.testUser  = np.array(testData[:][0])
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = np.array(testData[:][1])
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()

    @property
    def n_users(self):
        return 1892
    
    @property
    def m_items(self):
        return 4489
    
    @property
    def trainDataSize(self):
        return len(self.trainUser)
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)
            
            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).int()
            self.Graph = torch.sparse.IntTensor(index, data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            dense = self.Graph.to_dense()
            D = torch.sum(dense, dim=1).float()
            D[D==0.] = 1.
            D_sqrt = torch.sqrt(D).unsqueeze(dim=0)
            dense = dense/D_sqrt
            dense = dense/D_sqrt.t()
            index = dense.nonzero()
            data  = dense[dense >= 1e-9]
            assert len(index) == len(data)
            self.Graph = torch.sparse.FloatTensor(index.t(), data, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
            self.Graph = self.Graph.coalesce().to(world.device)
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    def __len__(self):
        return len(self.trainUniqueUsers)


class MyDataset(BasicDataset):
    def __init__(self,config = world.config,path = None):
        cprint(f'loading [{path}]')

        self.split = config['A_split']
        self.folds = config['A_n_fold']
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']

        self.n_user = 0
        self.m_item = 0

        self.path = path
        
        train_npz,test_npz,val_npz = self.load_data(path)

        self.traindataSize = 0
        self.testDataSize = 0


        self.trainUniqueUsers = np.unique(train_npz.row)
        self.trainUser = train_npz.row
        self.trainItem = train_npz.col

        self.traindataSize = len(self.trainItem)

        self.testUniqueUsers = np.unique(test_npz.row)
        self.testUser = test_npz.row
        self.testItem = test_npz.col
        self.testDataSize = len(self.testItem)

        self.valUniqueUsers = np.unique(val_npz.row)
        self.valUser = val_npz.row
        self.valItem = val_npz.col
        self.valDataSize = len(self.valItem)

        self.n_user = max(max(self.trainUniqueUsers),max(self.testUniqueUsers)) + 1
        self.m_item = max(max(self.trainItem),max(self.valItem),max(self.testItem)) + 1


        noisy_rate = config['noisy_rate']
        #if noisy_rate:
        self.set_noisy_labels(noisy_rate=noisy_rate)

        self.train_item_list = dict(zip(self.trainUser,self.trainItem))

        self.Graph = None
        self.scales = None

        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.valDataSize} interactions for validation")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{world.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                    shape=(self.n_user, self.m_item))
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))


        self.__testDict = self.__build_test()
        self.__valDict = self.__build_val()

        print(f"{world.dataset} is ready to go")

    # If you want random noise, specify a value other than 0 in config.
    def set_noisy_labels(self, noisy_rate):   
        print("noisy_rate : ", noisy_rate)


        self.noisy_interactions = lil_matrix((self.n_user, self.m_item))

        # (users,items), bipartite graph
        uinet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                    shape=(self.n_user, self.m_item))
        # pre-calculate
        lil = uinet.tolil()
        #print(lil[0,1])
        #print(int(self.trainDataSize * noisy_rate))

        if noisy_rate == 0:
            return 

        noisy_labels_index = np.random.choice(self.trainDataSize, size=int(self.trainDataSize * noisy_rate),replace=False)
        #print(f"noisy labels: {len(set(noisy_labels_index))}")
        for index in noisy_labels_index:
            user = self.trainUser[index]
            item = self.trainItem[index]
            while True:
                noisy_item = np.random.randint(0, self.m_item)
                if lil[user, noisy_item] == 0:
                    lil[user, noisy_item] = 1
                    #print(user,item,noisy_item)
                    self.trainItem[index] = noisy_item
                    self.noisy_interactions[user, noisy_item] = 1
                    break

    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict
    
    @property
    def valDict(self):
        return self.__valDict

    @property
    def allPos(self):
        return self._allPos
        

    
    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(world.device))
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                s = time()
                #adj_mat = sp.load_npz(self.path + '/train_coo_adj_graph.npz')
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])


                data = np.ones(self.traindataSize*2)
                row = np.concatenate((self.trainUser,self.trainItem+self.n_user))
                col = np.concatenate((self.trainItem+self.n_user,self.trainUser))
                adj_mat = sp.coo_matrix((data,(row,col)),shape=(self.n_user+self.m_item,self.n_user+self.m_item))
                """
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                
                """
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))

                d_inv_left = np.power(rowsum, -0.5).flatten()
                d_inv_left[np.isinf(d_inv_left)] = 0.
                d_mat_left = sp.diags(d_inv_left)

                d_inv_right = np.power(rowsum, -0.5).flatten()
                d_inv_right[np.isinf(d_inv_right)] = 0.
                d_mat_right = sp.diags(d_inv_right)
                
                norm_adj = d_mat_left.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat_right)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def __build_val(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.valItem):
            user = self.valUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

class NoisyDataset(MyDataset):
    def __init__(self,config = world.config,path = None):
        super().__init__(path=path)

        self.c_ui_matrix = self.precompute_c_ui()


    # ユーザ数*ユーザ数の共起行列を計算
    # ユーザiとユーザjが共にアイテムを評価した回数を共起行列の(i, j)成分に格納
    def calc_co_occurrence(self, block_size=10000):
        # lil_matrix を使ってスパース構造を変更する
        uinet_csr = self.UserItemNet.astype(np.uint32).tocsr()
        n_users = uinet_csr.shape[0]

        # lil_matrix を使って共起行列を初期化
        co_occurrence_matrix = lil_matrix((n_users, n_users), dtype=np.uint32)

        # ブロックごとに計算
        for start in tqdm(range(0, n_users, block_size)):
            end = min(start + block_size, n_users)
            uinet_block = uinet_csr[start:end, :]
            co_occurrence_block = uinet_block.dot(uinet_csr.T)

            # 共起行列に部分結果を保存（lil_matrix への挿入が高速）
            co_occurrence_matrix[start:end, :] = co_occurrence_block

        # 最後に csr_matrix に変換して返す
        return co_occurrence_matrix.tocsr()


    # 共起頻度スコアは事前に計算し、ファイルに保存しておく
    def precompute_c_ui(self,thres = 1):

        if os.path.exists(self.path + "/c_ui_matrix.npz"):
            print("load precomputed c_ui")
            c_ui_matrix = sp.load_npz(self.path + "/c_ui_matrix.npz")
            return c_ui_matrix

        print("precompute c_ui")

        co_occurrence_matrix = self.calc_co_occurrence()

        c_ui_matrix = lil_matrix((self.n_user, self.m_item))

        # 相互作用が存在するユーザーとアイテムのみ処理
        interaction_indices = self.UserItemNet.nonzero()
        # アイテムごとにユーザーのインタラクションをあらかじめ取得しておく
        all_item_user_interactions = {
            i: self.UserItemNet[:, i].nonzero()[0] for i in range(self.m_item)
        }

        for u, i in tqdm(zip(interaction_indices[0], interaction_indices[1]), total=len(interaction_indices[0])):
            users_interacted_with_item_i = all_item_user_interactions[i]
            co_occurrences_u = co_occurrence_matrix[u, users_interacted_with_item_i]
            c_ui_matrix[u, i] = np.sum(co_occurrences_u > thres)

        c_ui_matrix = c_ui_matrix.tocsr()
        sp.save_npz(self.path + '/c_ui_matrix.npz', c_ui_matrix)

        print("precompute c_ui done")

        return c_ui_matrix.tocsr()



    def get_c_ui(self, u, i):
        # 事前に計算されたc_uiを返す
        return self.c_ui_matrix[u, i]


class DDRMDataset(NoisyDataset):
    def __init__(self,config = world.config,path = None):
        super().__init__(path=path)

    def load_data(self,path):
        train_file = path + '/train_list.npy'
        valid_file = path + '/valid_list.npy'
        test_file = path + '/test_list.npy'

        self.path = path

        train_list = np.load(train_file, allow_pickle=True)
        valid_list = np.load(valid_file, allow_pickle=True)
        test_list = np.load(test_file, allow_pickle=True)

        data = [1]*len(train_list)
        train_npz = sp.coo_matrix((data,(train_list[:,0],train_list[:,1])))

        data = [1]*len(valid_list)
        valid_npz = sp.coo_matrix((data,(valid_list[:,0],valid_list[:,1])))

        data = [1]*len(test_list)
        test_npz = sp.coo_matrix((data,(test_list[:,0],test_list[:,1])))

        user_num = max(max(train_list[:,0]),max(valid_list[:,0]),max(test_list[:,0]))+1
        item_num = max(max(train_list[:,1]),max(valid_list[:,1]),max(test_list[:,1]))+1

        #self.noisy_interactions = lil_matrix((user_num, item_num))

        return train_npz,test_npz,valid_npz

