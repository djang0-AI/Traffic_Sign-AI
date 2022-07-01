import torch
import numpy as np

from copy import deepcopy

from torch.utils.data import DataLoader
from fastprogress.fastprogress import master_bar, progress_bar


class Trainer():
    def __init__(self, model, crit, optimizer, epochs, device):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

        super().__init__()


    def train(self, traindata, valdata):
        trainloader = DataLoader(traindata, batch_size=256, shuffle=True, drop_last = False)
        valloader = DataLoader(valdata, batch_size= 512, shuffle=False, drop_last= False)
        scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.99, end_factor=0.0001, total_iters=len(trainloader)*self.epochs, last_epoch=- 1, verbose=False)

        val_losses = []
        val_accuracy = []
        train_losses = []
        train_accuracy = []
        mb = master_bar(range(self.epochs))
        for epoch in mb:

            correct_train = 0
            correct_val = 0
            loss_train = 0
            loss_val = 0
            lowest_loss = np.inf
            self.model.train()

            for x, y in progress_bar(trainloader, parent = mb):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                t_loss = self.crit(y_hat, y)
                loss_train += t_loss.item()
                t_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                _,pred_idxs = torch.topk(y_hat, 1)
                correct_train += torch.eq(y, pred_idxs.squeeze()).sum().item()
                scheduler.step()

            self.model.eval()
            with torch.no_grad():
                for x, y in progress_bar(valloader, parent = mb):
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)
                    v_loss = self.crit(y_hat, y)
                    loss_val += v_loss.item()

                    _,pred_idxs = torch.topk(y_hat, 1)
                    correct_val += torch.eq(y, pred_idxs.squeeze()).sum().item()

                    if lowest_loss >= loss_val:
                        lowest_loss = loss_val
                        best_model = deepcopy(self.model.state_dict())

            acc_train = correct_train/len(traindata)
            acc_val = correct_val/len(valdata)
            loss_train = loss_train/len(trainloader)
            loss_val = loss_val/len(valloader)

            val_losses.append(loss_val)
            val_accuracy.append(acc_val)
            train_losses.append(loss_train)
            train_accuracy.append(acc_train)

            print("Epoch: {: d}  train_accuracy: {: .4f}  valid_accuracy: {: .4f} train_loss: {: .4f} valid_loss: {: .4f} lowest_loss: {: .4f}".format(epoch, acc_train, acc_val, loss_train, loss_val, lowest_loss))
            self.model.load_state_dict(best_model)
