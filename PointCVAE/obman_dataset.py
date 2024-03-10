from torch.utils.data import Dataset
import torch
import os
import numpy as np


class obman(Dataset):
    def __init__(self, obj_pc_path, obj_cls_path):
        self.obj_pc_path = obj_pc_path
        self.obj_cls_path = obj_cls_path
        self.file_list = os.listdir(self.obj_pc_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # obj_pc
        obj_pc = np.load(os.path.join(self.obj_pc_path, self.file_list[idx]))
        obj_cls = np.load(os.path.join(self.obj_cls_path, self.file_list[idx]))
        obj_pc = torch.tensor(obj_pc, dtype=torch.float32)
        obj_cls = torch.tensor(obj_cls, dtype=torch.long)
        return (obj_pc, obj_cls)
