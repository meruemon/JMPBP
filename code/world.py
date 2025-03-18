'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

#torch.autograd.set_detect_anomaly(True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

torch.cuda.set_device(args.gpu)

ROOT_PATH = "./"
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
RESULT_PATH = join(CODE_PATH, 'results')
RERANK_PATH = join(CODE_PATH, 'rerank')
RECOMMEND_PATH = join(CODE_PATH, 'recommends')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
rerank_config = {}

all_dataset = ['lastfm', 'gowalla', 'yelp2018', 'amazon-book','taobao','spotify','yahoo','beauty','adressa_noisy','yelp_noisy','book_noisy','ml_noisy','book_ddrm','yelp_ddrm','ml_ddrm']
all_models  = ['mf', 'lgn','simgcl','xsimgcl',"sgl"]

# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False



config['eps'] = args.eps

config['warmup'] = args.warmup
config['thres'] = args.thres
config['ensemble_prediction'] = args.ensemble_prediction


config["alpha"] = args.alpha
config["gmm_input"] = args.gmm_input


config['patience'] = args.patience

config["simgcl_tau"] = args.simgcl_tau
config["simgcl_eps"] = args.simgcl_eps
config["simgcl_lambda"] = args.simgcl_lambda

config["xsimgcl_layer_cl"] = args.xsimgcl_layer_cl

GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
#CORES = multiprocessing.cpu_count() // 2
CORES = 6
seed = args.seed
early_stop = args.early_stop
full_log = args.full_log
#neg_ratio = 4
neg_ratio = 128

dataset = args.dataset
model_name = args.model
if dataset not in all_dataset:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment

pth = args.pth
pth2 = args.pth2


config['noisy_rate'] = args.noisy_rate

test_interval = args.test_interval

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
