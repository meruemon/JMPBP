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
import os
import copy
import utils


#2つのネットワークのロード
Recmodel1 = register.MODELS[world.model_name](world.config, dataset)
Recmodel2 = register.MODELS[world.model_name](world.config, dataset)

Recmodel1 = Recmodel1.to(world.device)
Recmodel2 = Recmodel2.to(world.device)

bpr1 = utils.Loss(Recmodel1, world.config)
bpr2 = utils.Loss(Recmodel2, world.config)

now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f"{now}{world.comment}"
base_path = f"code/logs/{filename}"
os.makedirs(base_path,exist_ok=True)
os.makedirs(join(base_path,"distributions","net1"),exist_ok=True)
os.makedirs(join(base_path,"distributions","net2"),exist_ok=True)
os.makedirs(join(base_path,"certainties","net1"),exist_ok=True)
os.makedirs(join(base_path,"certainties","net2"),exist_ok=True)
os.makedirs(join(base_path,"samples","net1"),exist_ok=True)
os.makedirs(join(base_path,"samples","net2"),exist_ok=True)

#ランキングの確認
if world.pth:
    Recmodel1.load_state_dict(torch.load(utils.getFileName_Recommend(world.pth),map_location=torch.device('cpu')))

    if world.pth2:
        Recmodel2.load_state_dict(torch.load(utils.getFileName_Recommend(world.pth2),map_location=torch.device('cpu')))


    #全体の結果だけ見る場合
    result = Procedure.Test(dataset, Recmodel1, 0, dualnet=Recmodel2) 
    result_dict = dict([(k,v.tolist()) for k,v in result.items()])

else:

    wf = utils.getFileName()
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
        early_stop(0,Recmodel1,dualnet=Recmodel2)

    try:
        test_epoch = 10

        if world.test_interval != 0:
            test_epoch = world.test_interval

        for epoch in range(world.TRAIN_epochs):
            start = time.time()
            
            # 検証/テストデータの結果
            if epoch % test_epoch == 0:
                
                if world.early_stop:
                    result = Procedure.Val(dataset, Recmodel1, epoch, w, world.config['multicore'],dualnet=Recmodel2)
                    early_stop(result['recall'][0],Recmodel1,dualnet=Recmodel2)
                    
                    result_dict = dict([(k,v.tolist()) for k,v in result.items()])
                    result_dict['epoch'] = epoch
                    results.append(result_dict)
                    if early_stop.early_stop:
                        break

                else:
                    cprint("[TEST]")
                    result = Procedure.Test(dataset, Recmodel1, epoch, w, world.config['multicore'],dualnet=Recmodel2)
                    print(time.time()-start)

                    if best_recall < result['recall'][0]:
                        best_recall = result['recall'][0]
                        #coverage = result['coverage'][0]
                        best_model_weight = copy.deepcopy(Recmodel1.state_dict())
                        best_model_epoch = epoch

                    result_dict = dict([(k,v.tolist()) for k,v in result.items()])
                    result_dict['epoch'] = epoch
                    results.append(result_dict)

            if epoch < world.config['warmup']:
                output_information1 = Procedure.BPR_train_original(dataset, Recmodel1, bpr1, epoch, neg_k=Neg_k,w=w)
                output_information2 = Procedure.BPR_train_original(dataset, Recmodel2, bpr2, epoch, neg_k=Neg_k,w=w)

            else:

                users1, posItems1, _, criterions1 = Procedure.caluculate_criterion(dataset, Recmodel1, bpr1, epoch, neg_k=Neg_k,w=w)
                prob1 = Procedure.calculate_certainty(criterions1)

                users2, posItems2, _, criterions2 = Procedure.caluculate_criterion(dataset, Recmodel2, bpr2, epoch, neg_k=Neg_k,w=w)
                prob2 = Procedure.calculate_certainty(criterions2)


                sample1 = (users1,posItems1,prob1)
                sample2 = (users2,posItems2,prob2)

                output_information1 = Procedure.Certainty_aware_train(dataset, Recmodel1, bpr1, epoch, neg_k=Neg_k,w=w,sample=sample2)
                output_information2 = Procedure.Certainty_aware_train(dataset, Recmodel2, bpr2, epoch, neg_k=Neg_k,w=w,sample=sample1)

                if world.full_log:
                    np.savez(join(base_path,"samples","net1",f"{epoch}.npz"),criterion=criterions1,prob=prob1)
                    np.savez(join(base_path,"samples","net2",f"{epoch}.npz"),criterion=criterions2,prob=prob2)

            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] Recmodel1:{output_information1} Recmodel2:{output_information2} |time:{time.time()-start:.2f}|')

        cprint("[TEST]")
        
        if world.early_stop:
            Recmodel1.load_state_dict(torch.load(early_stop.save_net1_path,map_location=torch.device('cpu')))
            Recmodel2.load_state_dict(torch.load(early_stop.save_net2_path,map_location=torch.device('cpu')))

        result = Procedure.Test(dataset, Recmodel1, epoch, w, world.config['multicore'],dualnet=Recmodel2) 

        if best_recall < result['recall'][0]:
            best_recall = result['recall'][0]
            
        result_dict = dict([(k,v.tolist()) for k,v in result.items()])
        result_dict['epoch'] = epoch+1
        results.append(result_dict)

    except Exception as e:
        print(traceback.format_exc())

    finally:
        if world.tensorboard:
            w.close()

        with open(join(base_path,"result.txt"),mode='w') as f:
            for result in results:
                f.write(json.dumps(result)+"\n")
        
        if not world.early_stop:
            # TODO 検証データを使わないときの2つのモデルの保存
            torch.save(best_model_weight, wf + "[best="+str(best_model_epoch)+"epoch].pth")
