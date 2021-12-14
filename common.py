#!/usr/bin/python
import os
import json
import torch

def get_device_data(x):
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x

class Config(object):
    def __init__(self, json_path):
        self.wdir = os.path.abspath(os.path.dirname(json_path))
        with open(json_path, 'r') as f:
            self.jdata = json.load(f)
        self.data_path = self.jdata['data_path']
        self.model_parameter = self.jdata['model']
        self.type_map = self.model_parameter['type_map']
        self.type_number = len(self.type_map)
        self.descriptor_parameter = self.model_parameter['descriptor']
        self.fitting_parameter = self.model_parameter['fitting_net']
        self.learning_rate = self.jdata['learning_rate']
        self.training = self.jdata['training']
        self.loss = self.jdata['loss']
        self.mode = self.jdata['mode']
        self.model_path = self.jdata['model_path']
        self.require_force = self.jdata['require_force']


