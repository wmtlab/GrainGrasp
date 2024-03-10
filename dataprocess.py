import torch
import numpy as np
from utils import annotate, tools, Load_obman
import argparse
import os
from config import cfgs


class ObmanProcess:
    def __init__(self, cfg):
        self.mode = cfg.mode
        self.obman_path = cfg.obman_path
        self.shapeNet_path = cfg.shapeNet_path
        self.save_path = cfg.save_path
        self.start = cfg.start
        self.end = cfg.end
        self.sample_points_num = cfg.sample_points_num
        self.K = cfg.K
        self.load_obman = Load_obman(self.shapeNet_path, self.obman_path, self.mode)

    def process(self, save=True):
        actual_number_files = len(self.load_obman.pklNameList)
        if self.end == -1 or self.end > actual_number_files - 1:
            self.end = actual_number_files - 1

        if save:
            tools.check_dir(self.save_path)
            hand_pc_path = os.path.join(self.save_path, "hand_pc")
            hand_param_path = os.path.join(self.save_path, "hand_param")
            obj_pc_path = os.path.join(self.save_path, "obj_pc")
            obj_cls_path = os.path.join(self.save_path, "obj_cls")
            obj_mesh_path = os.path.join(self.save_path, "obj_mesh")
            tools.check_dir(hand_pc_path)
            tools.check_dir(obj_pc_path)
            tools.check_dir(obj_cls_path)
            tools.check_dir(hand_param_path)
            tools.check_dir(obj_mesh_path)

        print("Processing total {} files".format(self.end - self.start + 1))
        for idx in range(self.start, self.end + 1):
            print("Processing {}/{}".format(idx, self.end))
            meta = self.load_obman.get_meta(idx)
            hand_pc = self.load_obman.get_hand_pc(meta)
            hand_pose = self.load_obman.get_hand_pose(meta)
            obj_mesh = self.load_obman.get_obj_mesh(meta)
            obj_pc = tools.pc_sample(obj_mesh, self.sample_points_num)
            obj_cls, _ = annotate.get_obj_cls_and_colors(torch.Tensor(hand_pc), torch.Tensor(obj_pc), K=self.K)
            obj_cls = obj_cls.squeeze().cpu().detach().numpy()

            hand_pc_idx_path = os.path.join(hand_pc_path, "{}.npy".format(idx))
            np.save(hand_pc_idx_path, hand_pc)
            hand_param_idx_path = os.path.join(hand_param_path, "{}.npy".format(idx))
            np.save(hand_param_idx_path, hand_pose)
            obj_pc_idx_path = os.path.join(obj_pc_path, "{}.npy".format(idx))
            np.save(obj_pc_idx_path, obj_pc.T)  # [3, 3000]
            obj_cls_idx_path = os.path.join(obj_cls_path, "{}.npy".format(idx))
            np.save(obj_cls_idx_path, obj_cls)
            obj_mesh_idx_path = os.path.join(obj_mesh_path, "{}.obj".format(idx))
            obj_mesh.export(obj_mesh_idx_path)
            print("Saved idx = {} in directory {}".format(idx, self.save_path))


if __name__ == "__main__":
    cfg = cfgs.obman_config
    del cfgs
    parser = argparse.ArgumentParser(description="Process data")
    parser.add_argument("--start", "-s", type=int, default=0, help="Start index")
    parser.add_argument("--end", "-e", type=int, default=-1, help="End index, -1 means the last index")
    args = parser.parse_args()
    cfg.save_path = os.path.join(cfg.save_path, str(cfg.K), cfg.mode)
    cfg.start = args.start
    cfg.end = args.end
    obman_process = ObmanProcess(cfg)
    obman_process.process()
