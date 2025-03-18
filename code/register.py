import world
import dataloader
import model
import utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:
    dataset = dataloader.Loader(path="../data/"+world.dataset)
elif world.dataset == 'yelp_ddrm':
    dataset = dataloader.DDRMDataset(path="../data/yelp_ddrm")
elif world.dataset == 'ml_ddrm':
    dataset = dataloader.DDRMDataset(path="../data/ml_ddrm")
elif world.dataset == 'book_ddrm':
    dataset = dataloader.DDRMDataset(path="../data/book_ddrm")

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN,
    'sgl':model.SGL,
    'simgcl':model.SimGCL,
    'xsimgcl':model.XSimGCL,
}