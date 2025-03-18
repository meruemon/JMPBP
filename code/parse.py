'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=32,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=1,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=float,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='taobao',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[300]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--neg_sample', type=int, default=-1,
                        help='negative sample ratio.')    


    parser.add_argument('--pth', type=str,help="checkpoint path")
    parser.add_argument('--pth2', type=str,help="checkpoint path2")

    # JMPBP
    parser.add_argument('--thres', type=float, default=0.5,
                    help='sample selection threshold')
    parser.add_argument('--warmup', type=int, default=50,
                help='warmup epoch')
    parser.add_argument('--ensemble_prediction',type=int,default=1)
    
    parser.add_argument('--alpha', type=float,default=1,
                    help="proposed method")
    parser.add_argument('--gmm_input', nargs='*',default=["score","cui"], help="gmm input")


    #SimGCL
    parser.add_argument('--simgcl_lambda', type=float,default=1e-5)
    parser.add_argument('--simgcl_eps', type=float,default=0.1)
    parser.add_argument('--simgcl_tau', type=float,default=0.2)

    parser.add_argument('--xsimgcl_layer_cl', type=int,default=0)


    parser.add_argument('--test_interval', type=int, default=0,
                        help='test_interval')
    parser.add_argument('--early_stop',type=int,default=0,help='use earlystop')
    parser.add_argument('--noisy_rate',type=float,default=0,help='noisy_rate')
    parser.add_argument('--eps',type=float,default=0.1,help='eps')
    parser.add_argument('--patience',type=int,default=5,help='patience')

    parser.add_argument('--gpu', type=int,default=0,
                        help="set device id")

    parser.add_argument('--full_log', type=int,default=0,
                        help="whether save full log or not")

    return parser.parse_args()
