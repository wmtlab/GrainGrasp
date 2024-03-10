from torch.utils.data import Dataset
import torch
from pytorch3d import transforms
import numpy as np
import os


class obman(Dataset):
    def __init__(self, obj_pc_path, hand_pc_path):

        self.obj_pc_path = obj_pc_path
        self.hand_pc_path = hand_pc_path
        self.file_list = os.listdir(self.obj_pc_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        obj_pc = np.load(os.path.join(self.obj_pc_path, self.file_list[idx]))
        hand_pc = np.load(os.path.join(self.hand_pc_path, self.file_list[idx]))
        obj_pc = torch.tensor(obj_pc, dtype=torch.float32)  # 3*3000
        hand_pc = torch.tensor(hand_pc, dtype=torch.float32)

        if np.random.rand() >= 0.5:
            return (obj_pc, hand_pc, torch.tensor(1))
        else:
            if np.random.rand() > 0.5:
                hand_pc = hand_pc.mm(transforms.random_rotation())
            else:
                hand_pc = hand_pc + 0.05 * torch.randn((1, 3))
            return (obj_pc, hand_pc, torch.tensor(0))
