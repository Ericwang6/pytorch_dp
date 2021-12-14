#!/usr/bin/python

from typing import Any
import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from common import get_device_data
from dataloader import SystemData
from model import DeepPot
import json


def get_batch_size(batch_size: Any) -> int:
    if not isinstance(batch_size, int):
        print("Do not support non-int batch_size currently")
        return 1
    else:
        return batch_size


class Prefactor(object):
    def __init__(self, start: float, end: float, numb_steps: int, type: str = "linear"):
        self.start = start
        self.end = end
        self.diff = end - start
        self.type = type
        self.numb_steps = numb_steps
    
    def __getitem__(self, index):
        if self.type == "linear":
            return self.start + self.diff * index / (self.numb_steps - 1)


class Trainer(object):
    def __init__(self, jdata):
        self.type_map = jdata['model']['type_map']
        self.sel_list = jdata["model"]["descriptor"]["sel"]
        self.logfile = open(jdata["training"].get("disp_file", "lcurve.out"), 'w')

        self.train_data = SystemData(jdata["training"]["training_data"]["systems"], self.type_map, self.sel_list)
        self.test_data = SystemData(jdata["training"]['validation_data']["systems"], self.type_map, self.sel_list)
        self.train_batch_size = get_batch_size(jdata["training"]["training_data"]["batch_size"])
        self.test_batch_size = get_batch_size(jdata["training"]["validation_data"]["batch_size"])
        self.train_loader = DataLoader(self.train_data,
                                       batch_size = self.train_batch_size,
                                       shuffle = False, 
                                       pin_memory = True, 
                                       num_workers = 4)
        self.test_loader = DataLoader(self.test_data,
                                      batch_size = self.test_batch_size,
                                      shuffle = False, 
                                      pin_memory = True, 
                                      num_workers = 4)

        if torch.cuda.is_available():
            self.model = DeepPot(jdata['model']).cuda()
        else:
            self.model = DeepPot(jdata["model"])


        self.start_lr = jdata["learning_rate"]["start_lr"]
        self.stop_lr = jdata["learning_rate"]["stop_lr"]
        self.decay_steps = jdata["learning_rate"]["decay_steps"]
        self.numb_steps = jdata["training"]["numb_steps"]
        self.optimizer = Adam(self.model.parameters(), lr = self.start_lr)
        self.loss_fn = nn.MSELoss(reduction = "mean")

        self.e_pref = 0.0
        self.f_pref = 1.0

    def train(self):
        stop_flag = False
        step = 0
        while not stop_flag:
            for i, (batch_train_data, batch_test_data) in enumerate(zip(self.train_loader, self.test_loader)):
                self.model.train()
                e_pref = self.e_pref
                f_pref = self.f_pref
                # batch_loss = []

                for n in range(self.train_batch_size):
                    train_data = dict([(k, batch_train_data[k][n]) for k in batch_train_data if k != "sel"])
                    train_data["sel"] = [[x[n] for x in sel] for sel in batch_train_data["sel"]]
                    e_pred, f_pred = self.model(train_data)
                    e_true = get_device_data(train_data["energy"])
                    f_true = get_device_data(train_data["force"])

                    e_loss = self.loss_fn(e_true, e_pred)
                    f_loss = self.loss_fn(f_true, f_pred)
                    # print(e_true, e_pred)
                    # print(f_true, f_pred)
                    print(f_true)
                    print(f_pred)
                    # batch_loss.append(e_pref * e_loss + f_pref * f_loss)
                    batch_loss = e_pref * e_loss + f_pref * f_loss
                    print(f"Batch Loss: {batch_loss}  Energy Loss: {e_loss}  Force Loss: {f_loss}")
                # batch_loss = 
                # batch_loss = torch.sum(batch_loss) / self.train_batch_size

                    self.optimizer.zero_grad()
                    batch_loss.backward()

                    # for param in self.model.parameters():
                        # print(param.size(), param.grad)
                    # exit(1)
                    self.optimizer.step()

                step += 1
                stop_flag = step >= self.numb_steps

                if stop_flag: break
        self.logfile.close()
                    


        #         pred_energy, pred_force = self.model(batch_data)  # (batch_size, 1) (batch_size, atom_num, 3)
        #         gt_energy = get_device_data(batch_data['frame_energy'])
        #         gt_force = get_device_data(batch_data['frame_force'])
        #         energy_loss = self.config.loss['pref_e'] * self.loss_fn(pred_energy, gt_energy)
        #         if self.require_force:
        #             force_loss = self.config.loss['pref_f'] * self.loss_fn(pred_force, gt_force)
        #             loss = energy_loss + force_loss
        #             self.writer.add_scalars('training_loss', {'energy_loss': energy_loss, 'force_loss': force_loss}, i+epoch*self.train_num)
        #             if i % self.config.training['disp_freq'] == 0:
        #                 print('epoch{}:batch{}  energy_loss:{}  force_loss:{}  training loss:{}'.format(epoch, i, energy_loss, force_loss, loss))
        #         else:
        #             loss = energy_loss
        #             self.writer.add_scalars('training_loss', {'energy_loss': energy_loss}, i+epoch*self.train_num)
        #             if i % self.config.training['disp_freq'] == 0:
        #                 print('epoch{}:batch{}  energy_loss:{}  training loss:{}'.format(epoch, i, energy_loss, loss))

        #         ##TODO Stage_1##
        #         ##### change the lr of optimizer #####
        #         for p in self.optimizer.param_groups:
        #             p['lr'] *= 1
        #         ##### change the lr of optimizer #####

        #         self.optimizer.zero_grad()
        #         loss.backward()
        #         self.optimizer.step()

        #         if (i+1) % self.config.training['test_freq'] == 0:
        #             self.test(epoch)
        #             self.model.train()
        # self.writer.close()

    # def test(self, epoch):

    #     self.model.eval()
    #     total_mse = 0.
    #     for (i, batch_data) in enumerate(tqdm(self.test_loader)):
    #         pred_energy, pred_force = self.model(batch_data)  # (batch_size, 1) (batch_size, atom_num, 3)
    #         gt_energy = get_device_data(batch_data['frame_energy'])
    #         gt_force = get_device_data(batch_data['frame_force'])
    #         energy_loss = self.loss_fn(pred_energy, gt_energy)
    #         if self.require_force:
    #             force_loss = self.loss_fn(pred_force, gt_force)
    #             loss = self.config.loss['pref_e'] * energy_loss + self.config.loss['pref_f'] * force_loss
    #         else:
    #             loss = self.config.loss['pref_e'] * energy_loss
    #         total_mse += loss.cpu()
    #     total_mse = (total_mse / self.train_num).item()
    #     self.writer.add_scalars('testing_loss', {'total_mse': total_mse}, epoch)
    #     print('test total_mse:', total_mse)
    #     if (not self.test_only) and total_mse <= self.best_mse:
    #         self.best_mse = total_mse
    #         print('best model found!')
    #         torch.save(self.model.state_dict(), './checkpoint/best_model.pth')
    #         print('best model saved!')

if __name__ == "__main__":
    input_json = sys.argv[1]
    with open(input_json, 'r') as f:
        jdata = json.load(f)
    trainer = Trainer(jdata)
    trainer.train()








