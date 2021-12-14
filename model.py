#!/usr/bin/python

import numpy as np
import torch
import torch.nn as nn
from typing import List, Union
from common import get_device_data


activation_funcs = {
    "tanh": torch.tanh,
    "relu": torch.relu
}

pi = torch.tensor(np.pi)


class NormalBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation_func: str = "tanh"):
        super(NormalBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activate_func = activation_funcs[activation_func]
        self.linear = nn.Linear(self.in_dim, self.out_dim)
    
    def forward(self, x):
        out = self.linear(x)
        out = self.activate_func(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, resnet_dt: bool = True, activation_func: str = "tanh"):
        super(ResBlock, self).__init__()
        info = f"Dimension not match in ResNet: {in_dim} vs {out_dim}, out_dim has to be equal to in_dim or be twice in_dim"
        assert in_dim == out_dim or in_dim * 2 == out_dim, info

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.activate_func = activation_funcs[activation_func]
        self.resnet_dt = resnet_dt
        if resnet_dt:
            self.dt = nn.Parameter(torch.FloatTensor(1), requires_grad = True)
        else:
            self.dt = torch.tensor(1.)

    def forward(self, x):
        features = self.activate_func(self.linear(x))
        if self.in_dim * 2 == self.out_dim:
            identity = torch.cat([x, x], dim = 1)
        else:
            identity = x
        
        out = torch.add(self.dt * features, identity)
        return out


class EmbeddingNet(nn.Module):
    def __init__(self, neuron_list: List[int], resnet_dt: bool = True, activation_func: str = "tanh"):
        super(EmbeddingNet, self).__init__()
        self.n_layers = len(neuron_list)
        self.neuron_list = neuron_list
        assert self.n_layers >= 1, "Embedding net must have at least 1 layer"
        layers = [NormalBlock(1, self.neuron_list[0], activation_func)]
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                layers.append(ResBlock(neuron_list[i], neuron_list[i + 1], resnet_dt, activation_func))

        self.linear_layers = nn.Sequential(*layers)

    def forward(self, x):  # input must be reshaped as (1,1)
        return self.linear_layers(x)


class FittingNet(nn.Module):
    def __init__(self, in_dim: int, neuron_list: List[int], resnet_dt: bool = True, activation_func: str = "tanh"):
        super(FittingNet, self).__init__()
        self.n_layers = len(neuron_list)
        self.neuron_list = neuron_list
        assert self.n_layers >= 1, "Fitting net must have at least 1 layer"
        layers = [NormalBlock(in_dim, self.neuron_list[0], activation_func)]
        if self.n_layers > 1:
            for i in range(self.n_layers - 1):
                layers.append(ResBlock(neuron_list[i], neuron_list[i + 1], resnet_dt, activation_func))
        
        self.linear_layers = nn.Sequential(*layers)
        self.out_layer = nn.Linear(self.neuron_list[-1], 1)

    def forward(self, x):
        return self.out_layer(self.linear_layers(x))


class DeepPot(nn.Module):
    def __init__(self, config):
        super(DeepPot, self).__init__()
        self.type_map = config["type_map"]

        self.descprt_params = config["descriptor"]
        self.rcs = torch.tensor(self.descprt_params["rcut_smth"])
        self.rc = torch.tensor(self.descprt_params["rcut"])
        self.type_one_side = self.descprt_params.get("type_one_side", True)
        self.sel_list = self.descprt_params["sel"]

        self.fitting_params = config["fitting_net"]


        self.M1 = self.descprt_params["neuron"][-1]
        self.M2 = self.descprt_params["axis_neuron"]

        if not self.type_one_side:
            embed_nets = []
            for i in range(len(self.type_map)):
                tmp = []
                for j in range(len(self.type_map)):
                    tmp.append(EmbeddingNet(
                                    neuron_list = self.descprt_params["neuron"],
                                    resnet_dt = self.descprt_params.get("resnet_dt", False),
                                    activation_func = self.descprt_params.get("activation_func", "tanh")
                    ))
                embed_nets.append(nn.ModuleList(tmp))
            embed_nets = nn.ModuleList(embed_nets)

            fit_nets = []
            for i in range(len(self.type_map)):
                tmp = []
                for j in range(len(self.type_map)):
                    tmp.append(FittingNet(
                                    in_dim = self.M1 * self.M2,
                                    neuron_list = self.fitting_params["neuron"],
                                    resnet_dt = self.fitting_params.get("resnet_dt", False),
                                    activation_func = self.fitting_params.get("activation_func", "tanh")
                    ))
                fit_nets.append(nn.ModuleList(tmp))
            fit_nets = nn.ModuleList(fit_nets)
        else:
            embed_nets = []
            for i in range(len(self.type_map)):
                embed_nets.append(EmbeddingNet(
                                    neuron_list = self.descprt_params["neuron"],
                                    resnet_dt = self.descprt_params.get("resnet_dt", False),
                                    activation_func = self.descprt_params.get("activation_func", "tanh")
                    ))
            embed_nets = nn.ModuleList(embed_nets)

            fit_nets = []
            for i in range(len(self.type_map)):
                fit_nets.append(FittingNet(
                                    in_dim = self.M1 * self.M2,
                                    neuron_list = self.fitting_params["neuron"],
                                    resnet_dt = self.fitting_params.get("resnet_dt", False),
                                    activation_func = self.fitting_params.get("activation_func", "tanh")
                    ))
            fit_nets = nn.ModuleList(fit_nets)

        self.embed_nets = embed_nets
        self.fit_nets = fit_nets

    def forward(self, data):
        coord = get_device_data(torch.tensor(data["coord"], requires_grad = True))
        atom_types = data["atom_types"]
        sel_idx = data["sel"]
        atom_energies = []
        for i in range(len(atom_types)):
            sel = sel_idx[i]
            type_i = atom_types[i]
            matR = compute_generalized_relative_coord(coord, i, self.rcs, self.rc, sel)
            s_ij = matR[:, 0]
            matG1 = []
            for j in range(len(sel)):
                type_j = atom_types[sel[j]]
                input_s_ij = s_ij[j].reshape(1, 1)
                if not self.type_one_side:
                    matG1.append(self.embed_nets[type_i][type_j](input_s_ij))
                else:
                    matG1.append(self.embed_nets[type_i](input_s_ij))
            
            matG1 = torch.cat(tuple(matG1), dim = 0)
            matG2 = matG1[:, :self.M2]
            # print(matG1.size())
            # print(matG2.size())
            # print(matR.size())
            coord_filter = torch.matmul(matG1.t(), matR)
            axis_filter = torch.matmul(matR.t(), matG2)
            fitting_input = torch.matmul(coord_filter, axis_filter).reshape(1, -1)

            if not self.type_one_side:
                atomic_energy = self.fit_nets[type_i][type_j](fitting_input)
            else:
                atomic_energy = self.fit_nets[type_i](fitting_input)
            
            atom_energies.append(atomic_energy)
        
        atom_energies = torch.cat(tuple(atom_energies))
        energy = torch.sum(atom_energies).unsqueeze(0).unsqueeze(0)

        force = -torch.autograd.grad(outputs = energy, inputs = coord, create_graph = True, retain_graph = True)[0]

        # print(force)

        return energy, force


def compute_generalized_relative_coord(coord: torch.Tensor,
                                       atom_idx: int,
                                       rcs: Union[float, torch.FloatTensor],
                                       rc: Union[float, torch.FloatTensor],
                                       sel_idx: List[int]):
    curr_coord = coord[atom_idx]
    # rest_coord = torch.cat((coord[:atom_idx], coord[atom_idx + 1:]), dim = 0)
    rest_coord = coord[sel_idx, :]
    # rest_idx = coord.size()[0]
    rel_coord = rest_coord - curr_coord
    r_ij = torch.sqrt(torch.sum(torch.pow(rel_coord, 2), dim = 1))
    
    stage_1_idx = (r_ij < rcs).float()
    stage_2_idx = ((r_ij < rc) & (r_ij >= rcs)).float()
    s_ij_1 = 1 / r_ij
    s_ij_2 = 1 / r_ij * (torch.cos(pi * (r_ij - rcs) / (rc - rcs)) / 2 + 0.5)
    s_ij = s_ij_1 * stage_1_idx + s_ij_2 * stage_2_idx
    rel_coord_general = torch.cat((s_ij.reshape(-1, 1), s_ij.reshape(-1, 1) * rel_coord / r_ij.reshape(-1, 1)), dim = 1)
    # print(rel_coord_general.size())
    return rel_coord_general