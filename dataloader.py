#!/usr/bin/python
from typing import List
from torch.utils.data import Dataset
import numpy as np
import dpdata


class SingleSystemData(Dataset):
    def __init__(self, dirname: str, type_map: List[str], sel_list: List[int]):
        self.system = dpdata.LabeledSystem(dirname, fmt = "deepmd/npy", type_map = type_map)
        self.atom_types = self.system["atom_types"]
        self.n_atoms = self.system.get_natoms()
        self.sel_list = sel_list
        
        self.sel_idxs = [self.get_sel_idx(i) for i in range(self.system.get_nframes())]
        
    
    def get_sel_idx(self, frame_idx: int):
        coord = self.system["coords"][frame_idx]
        sel_idxs = []
        for i in range(self.n_atoms):
            atom_type = self.atom_types[i]
            max_sel = self.sel_list[atom_type]
            rel_coord = coord - coord[i]
            r_ij = np.linalg.norm(rel_coord, axis = 1).flatten()
            argsort = np.lexsort((self.atom_types, r_ij))[1: max_sel + 1]
            sel_idxs.append(argsort.tolist())
        return sel_idxs
    
    def __len__(self):
        return self.system.get_nframes()

    def __getitem__(self, index: int):
        data = {}
        data["coord"] = self.system["coords"][index]
        data["energy"] = self.system["energies"][index]
        data["force"] = self.system["forces"][index]
        data["sel"] = self.sel_idxs[index]
        data["atom_types"] = self.system["atom_types"]
        return data


class SystemData(Dataset):
    def __init__(self, dirnames: List[str], type_map: List[str], sel_list: List[int]):
        self.n_systems = len(dirnames)
        self.systems = [
            SingleSystemData(dirname, type_map, sel_list) for dirname in dirnames
        ]
        self._map_system = []
        self._map_frames = []
        for i in range(self.n_systems):
            ss = self.systems[i]
            for j in range(len(ss)):
                self._map_system.append(i)
                self._map_frames.append(j)
    
    def __len__(self):
        return sum([len(system) for system in self.systems])
    
    def __getitem__(self, index):
        return self.systems[self._map_system[index]][self._map_frames[index]]

        
