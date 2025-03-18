import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import datetime 
import json
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import earlystop

import copy


Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

bpr = utils.Loss(Recmodel, world.config)


now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

check_distribute = False

#ランキングの確認
if world.pth:
    Recmodel.load_state_dict(torch.load(utils.getFileName_Recommend(world.pth),map_location=torch.device('cpu')))
    
    # 全体の結果だけ見る場合
    result = Procedure.Test(dataset, Recmodel, 0) 
    result_dict = dict([(k,v.tolist()) for k,v in result.items()])
    full_filename = now + world.comment + ".txt"
    path = join(world.RERANK_PATH,full_filename)
    with open(path,mode='w') as f:
        f.write(json.dumps(world.rerank_config)+"\n")
        f.write(json.dumps(world.topks)+"\n")
        f.write(json.dumps(result_dict)+"\n")
    # 各ユーザの推薦結果を見る場合
    #Procedure.Recommend(dataset,Recmodel,now=now) 
    
else:

    wf = utils.getFileName()
    weight_file = wf + ".pth"
    
    print(f"load and save to {weight_file}")
    Neg_k = 1


    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, now + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")


    results = [world.config,world.topks]

    best_recall = 0
    coverage = 0
    best_model_weight = None
    best_model_epoch = 0

    if world.early_stop:
        early_stop = earlystop.EarlyStoppingCriterion(save_path=wf,patience=world.config['patience'])
        early_stop(0,Recmodel)

    try:

        test_epoch = 10


        if world.test_interval != 0:
            test_epoch = world.test_interval

        for epoch in range(world.TRAIN_epochs):
            start = time.time()
            

            if epoch % test_epoch == 0:
                
                if world.early_stop:
                    result = Procedure.Val(dataset, Recmodel, epoch, w, world.config['multicore'])
                    early_stop(result['recall'][0],Recmodel)
                    
                    result_dict = dict([(k,v.tolist()) for k,v in result.items()])
                    result_dict['epoch'] = epoch
                    results.append(result_dict)
                    if early_stop.early_stop:
                        break

                else:
                    cprint("[TEST]")
                    result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                    print(time.time()-start)

                    if best_recall < result['recall'][0]:
                        best_recall = result['recall'][0]
                        #coverage = result['coverage'][0]
                        best_model_weight = copy.deepcopy(Recmodel.state_dict())
                        best_model_epoch = epoch
                    result_dict = dict([(k,v.tolist()) for k,v in result.items()])
                    result_dict['epoch'] = epoch
                    results.append(result_dict)

            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)


            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}  |time:{int(time.time()-start)}|')
            torch.save(Recmodel.state_dict(), weight_file)
        cprint("[TEST]")
        

        if world.early_stop:
            Recmodel.load_state_dict(torch.load(early_stop.save_path,map_location=torch.device('cpu')))
        result = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore']) 
        if best_recall < result['recall'][0]:
            best_recall = result['recall'][0]
            #coverage = result['coverage'][0]
        result_dict = dict([(k,v.tolist()) for k,v in result.items()])
        result_dict['epoch'] = epoch+1
        results.append(result_dict)
            
    except Exception as e:
        print(traceback.format_exc())

    finally:
        if world.tensorboard:
            w.close()

        full_filename = now + world.comment + ".txt"
        path = join(world.RESULT_PATH,full_filename)
        with open(path,mode='w') as f:
            for result in results:
                f.write(json.dumps(result)+"\n")
        
        if not world.early_stop:
            torch.save(best_model_weight, wf + "[best="+str(best_model_epoch)+"epoch].pth")
