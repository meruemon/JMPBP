import numpy as np
import torch

class EarlyStoppingCriterion:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0, save_path=None):
        """
        Args:
        ¦   patience (int): How long to wait after last time validation loss improved.
        ¦   ¦   ¦   ¦   ¦   Default: 5
        ¦   verbose (bool): If True, prints a message for each validation loss improvement.
        ¦   ¦   ¦   ¦   ¦   Default: False
        ¦   delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        ¦   ¦   ¦   ¦   ¦   Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_result = None
        self.early_stop = False
        self.val_result_max = 0
        self.delta = delta
        self.save_path = save_path +"[earlystop].pth"

        #ネットワークが2つの場合
        self.save_net1_path = save_path +"_[net1]_[earlystop].pth"
        self.save_net2_path = save_path +"_[net2]_[earlystop].pth"

    def __call__(self, val_result, model, dualnet=None):

        if self.best_result is None:
            self.best_result = val_result

            self.save_checkpoint(val_result, model,dualnet)

        elif val_result < self.best_result - self.delta:
            self.counter += 1
            print(f'recall: {val_result}, EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_result = val_result
            print(f'recall: {self.best_result}, Performance is better... saving the model')
            self.save_checkpoint(val_result, model,dualnet)
            self.counter = 0

    def save_checkpoint(self, val_result, model,dualnet=None):
        '''Saves model when validation loss decrease.'''

        print(f'Validation recall increased ({self.val_result_max:.6f} --> {val_result:.6f}).Saving model ...')

        if dualnet:
            torch.save(model.state_dict(), self.save_net1_path)
            torch.save(dualnet.state_dict(), self.save_net2_path)
        else:
            torch.save(model.state_dict(), self.save_path)
        self.val_result_max = val_result