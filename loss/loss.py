import torch.nn as nn
from log.log import Log
import numpy as np


class LossLog(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_recon = nn.L1Loss()#nn.SmoothL1Loss()#nn.MSELoss()#nn.L1Loss()
        self.log = Log()

        self.history_one_epoch = {
            'loss_recon': list(),
            'loss_total': list(),
        }

    def forward(self, data, target):
        loss_recon = self.loss_recon(data, target)
        loss_total = loss_recon

        self.history_one_epoch['loss_recon'].append(float(loss_recon))
        self.history_one_epoch['loss_total'].append(float(loss_total))

        return loss_total

    def write_save(self, epoch, fn):
        for k, v in self.history_one_epoch.items():
            self.log.write(k, np.mean(v))
            self.history_one_epoch[k] = list()

        self.log.write('epoch', epoch)
        self.log.save(fn)

