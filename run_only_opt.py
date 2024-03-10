import os
import time
import torch
import trimesh
import argparse
import numpy as np
import open3d as o3d
from utils import annotate
from utils import vis
from utils import tools
from utils import Load_obman
from config import cfgs
from GrainGrasp import GrainGrasp


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parge = argparse.ArgumentParser(description="GrainGrasp")
    parge.add_argument("--idx", "-i", type=int, default=1, help="The idx of the object")
    parge.add_argument("--epochs", "-e", type=int, default=300, help="The epochs of the optimization")
    parge.add_argument("--K", "-k", type=int, default=50)
    parge.add_argument("--threshold", "-t", type=float, default=0.0)
    parge.add_argument("--select", "-s", type=str, default="12345")
    parge.add_argument("--vis_pc", "-vp", type=bool, default=True)
    parge.add_argument("--vis_mesh", "-vm", type=bool, default=True)
    parge.add_argument("--vis_process", "-vprocess", type=bool, default=False)
    args = parge.parse_args()
    select_finger_idx = list(map(lambda x: int(x), args.select))
    sample_points_num = cfgs.obman_config.sample_points_num
    obj_path = os.path.join("sample", str(args.idx), "obj_mesh.obj")
    obj_mesh = trimesh.load_mesh(obj_path)
    obj_pc = tools.pc_sample(obj_mesh, sample_points_num)
    hand_pc_path = os.path.join("sample", str(args.idx), "hand_pc.npy")  # [3,3000]
    hand_pc = np.load(hand_pc_path)
    obj_pc = torch.Tensor(obj_pc)
    hand_pc = torch.Tensor(hand_pc)

    # load the GrainGrasp model
    grain_grasp = GrainGrasp(cfgs.dcog_config, None, device)
    time_start = time.time()
    result = grain_grasp.inference_only_opt(
        obj_pc,
        hand_pc=hand_pc,
        K=args.K,
        epochs=args.epochs,
        select_finger_idx=select_finger_idx,
        threshold=args.threshold,
    )
    print("The running time is {:.2f}s".format(time.time() - time_start))
    print("The Epen is ", result.E_pen)
    print("The min_idx is ", result.min_idx)
    hand_pc_final = result.min_idx_hand_pc
    hand_face = grain_grasp.dcog_model.rh_faces[0].cpu()
    hand_color = annotate.get_finger_colors(hand_pc_final)
    hand_mesh_o3d = vis.get_o3d_mesh(hand_pc_final, hand_face, [0, 0.8, 1], hand_color)
    obj_colors_true = annotate.get_obj_colors(result.obj_cls.cpu())
    obj_pcd = vis.get_o3d_pcd(obj_pc.cpu().detach(), obj_colors_true)
    obj_mesh_o3d = vis.trimesh2o3d(obj_mesh)

    if args.vis_pc:
        vis.vis_HandObject([hand_mesh_o3d], [obj_pcd])
    if args.vis_mesh:
        vis.vis_HandObject([hand_mesh_o3d], [obj_mesh_o3d])

    # vis the process of the optimization
    if args.vis_process:
        record_hand_pc = result.min_idx_record_hand_pc
        record_handmesh_o3d = hand_mesh_o3d = vis.get_o3d_mesh(record_hand_pc[0], hand_face, [0, 0.8, 1], hand_color)
        vis.vis_GraspProcess(record_handmesh_o3d, record_hand_pc[1:], obj_mesh_o3d)
