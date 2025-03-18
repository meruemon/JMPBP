"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

import time

#import dgl
import scipy.sparse as sp

import torch.nn.functional as F
import utils

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items

        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        self.weight_decay = config['decay']
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        
    def getUsersRating(self, users,get_emb=False):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())

        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg,prob=None):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, self.weight_decay*reg_loss

        

    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)


class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()
        self.weight_decay = config['decay']

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        world.cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()



    def update_graph(self, graph):
        self.Graph = graph


    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb] 
        
        g_droped = self.Graph
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users, get_emb=False):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))

        #print(rating.shape)

        if get_emb:
            return torch.matmul(users_emb, items_emb.t()),users_emb,items_emb

        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    


    def bpr_loss(self, users, pos, neg,prob=None):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        #print(users_emb)

        losses = torch.nn.functional.softplus(neg_scores - pos_scores)

        if prob is not None:
            #print(prob)
            
            full_loss = torch.mean(losses)
            losses = losses * prob

        loss = torch.mean(losses)
        

        return loss, self.weight_decay*reg_loss
    



    def get_sample_scores(self, users, items):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        pos_emb = all_items[items.long()]
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        return pos_scores

    def get_sample_features(self, users, items,neg_items):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb] 

        g_droped = self.Graph
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)


        if world.model_name in ["simgcl","xsimgcl"]:
            embs = torch.stack(embs[1:], dim=1)
        else:
            embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        all_users, all_items = torch.split(light_out, [self.num_users, self.num_items])

        users_femb = all_users[users.long()]
        pos_femb = all_items[items.long()]
        pos_scores = torch.mul(users_femb, pos_femb)
        pos_scores = torch.sum(pos_scores, dim=1)
        
        return pos_scores

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma



class SGL(LightGCN):
    def __init__(self, config, dataset):
        super(SGL, self).__init__(config, dataset)

        self.cl_rate = config['simgcl_lambda']
        self.eps = 0.1
        self.temperature = config['simgcl_tau']

        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)


    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()


    def computer(self,graph=None):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb] 
        if graph is not None:
            g_droped = graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def set_subgraph(self):
        self.subgraph1 = self.generate_subgraph()
        self.subgraph2 = self.generate_subgraph()


    def rand_sample(self, high, size=None, replace=True):
        r"""Randomly discard some points or edges.

        Args:
            high (int): Upper limit of index value
            size (int): Array size after sampling

        Returns:
            numpy.ndarray: Array index after sampling, shape: [size]
        """

        a = np.arange(high)
        sample = np.random.choice(a, size=size, replace=replace)
        return sample

    def generate_subgraph(self):
        self.drop_ratio = 0.2
        # SGL-ED
        keep_item = self.rand_sample(
                    len(self.dataset.trainUser),
                    size=int(len(self.dataset.trainUser) * (1 - self.drop_ratio)),
                    replace=False,
                )
        user = self.dataset.trainUser[keep_item]
        item = self.dataset.trainItem[keep_item]

        matrix = sp.csr_matrix(
            (np.ones_like(user), (user, item + self.num_users)),
            shape=(self.num_users + self.num_items, self.num_users + self.num_items),
        )

        matrix = matrix + matrix.T
        D = np.array(matrix.sum(axis=1)) + 1e-7
        D = np.power(D, -0.5).flatten()
        D = sp.diags(D)
        matrix = D.dot(matrix).dot(D)

        matrix = matrix.tocoo()
        x = torch.sparse.FloatTensor(
            torch.LongTensor(np.array([matrix.row, matrix.col])),
            torch.FloatTensor(matrix.data.astype(np.float32)),
            matrix.shape,
        ).to(world.device)
        
        return x

    def calculate_loss(self, users, pos_items, neg_items, prob = None):

        loss,reg_loss = self.bpr_loss(users,pos_items,neg_items,prob=prob)
        loss = loss + reg_loss


        user = torch.unique(users)
        pos_item = torch.unique(pos_items)
        
        perturbed_user_embs_1, perturbed_item_embs_1 = self.computer(self.subgraph1)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.computer(self.subgraph2)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])

        cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)

        return loss + cl_loss



class SimGCL(LightGCN):
    def __init__(self, config, dataset):
        super(SimGCL, self).__init__(config, dataset)

        self.cl_rate = config['simgcl_lambda']
        self.eps = config['simgcl_eps']
        self.temperature = config['simgcl_tau']

        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

    def computer(self, perturbed=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = []
        g_droped = self.Graph    
        

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb, device=world.device)
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps

            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def calculate_cl_loss(self, x1, x2):
        x1, x2 = F.normalize(x1, dim=-1), F.normalize(x2, dim=-1)
        pos_score = (x1 * x2).sum(dim=-1)
        pos_score = torch.exp(pos_score / self.temperature)
        ttl_score = torch.matmul(x1, x2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / self.temperature).sum(dim=1)
        return -torch.log(pos_score / ttl_score).sum()

    def simgcl_loss(self, users, pos_items, neg_items, prob = None):

        loss,reg_loss = self.bpr_loss(users,pos_items,neg_items,prob=prob)
        loss = loss + reg_loss


        user = torch.unique(users)
        pos_item = torch.unique(pos_items)
        
        perturbed_user_embs_1, perturbed_item_embs_1 = self.computer(perturbed=True)
        perturbed_user_embs_2, perturbed_item_embs_2 = self.computer(perturbed=True)

        user_cl_loss = self.calculate_cl_loss(perturbed_user_embs_1[user], perturbed_user_embs_2[user])
        item_cl_loss = self.calculate_cl_loss(perturbed_item_embs_1[pos_item], perturbed_item_embs_2[pos_item])

        cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)

        return loss + cl_loss
   
class XSimGCL(SimGCL):
    def __init__(self, config, dataset):
        super(XSimGCL, self).__init__(config, dataset)
        
        self.layer_cl = config['xsimgcl_layer_cl']


        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

    def computer(self, perturbed=False):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        all_embs_cl = all_emb

        embs = []

        rand_layer_cl = self.layer_cl

        g_droped = self.Graph    
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            if perturbed:
                random_noise = torch.rand_like(all_emb, device=world.device)
                all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
            embs.append(all_emb)
            if layer == rand_layer_cl:
                all_embs_cl = all_emb
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(light_out, [self.num_users, self.num_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embs_cl, [self.num_users, self.num_items])
        if perturbed:
            return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl
        return user_all_embeddings, item_all_embeddings

    def xsimgcl_loss(self, users, pos_items, neg_items, prob = None):
        # 0層目の埋め込みベクトル
        loss,reg_loss = self.bpr_loss(users,pos_items,neg_items,prob=prob)
        loss = loss + reg_loss

        user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl = self.computer(perturbed=True)

        user = torch.unique(users)
        pos_item = torch.unique(pos_items)

        user_cl_loss = self.calculate_cl_loss(user_all_embeddings[user], user_all_embeddings_cl[user])
        item_cl_loss = self.calculate_cl_loss(item_all_embeddings[pos_item], item_all_embeddings_cl[pos_item])

        cl_loss = self.cl_rate * (user_cl_loss + item_cl_loss)

        return loss + cl_loss

