import json
import matplotlib.pyplot as plt
import os 
import numpy as np
import seaborn as sns
import pandas as pd


class PlotManager:
    def __init__(self,filename):
        self.base_path = "code/logs/"+filename

    # サンプル選択における欲しい評価指標を指定して取得する関数
    # 出力: x(epoch),y(value)
    def get_value_x_y(self,d,value):
        warmup = d["warmup"]
        info = d["info"]
        x = range(warmup,warmup+len(info))
        #2つ目のネットワークの値を取得
        y = [t[value][-1] for t in info]
        
        return x,y

    # サンプル選択の精度評価を取得する関数
    def get_sample_selection_eval(self):
        with open(os.path.join(self.base_path,'sample_selection_info.json')) as f:
            d = json.load(f)
        value = "precision"
        prec_x,prec_y = self.get_value_x_y(d,value)
        value = "accuracy"
        acc_x,acc_y = self.get_value_x_y(d,value)
        
        return prec_x,prec_y,acc_x,acc_y


    # 指標に対して割り当てられる重みを取得する関数
    def get_samples(self,epoch,net="net2"):
        return np.load(os.path.join(self.base_path,"samples",net,f"{epoch}.npz"))


    # 全エポックの精度評価評価を取得する関数
    def get_results(self):

        with open(os.path.join(self.base_path,"result.txt")) as f:
            results_txt = [s.strip() for s in f.readlines()]

            results = [json.loads(one_epoch) for one_epoch in results_txt[2:]]

            return results
