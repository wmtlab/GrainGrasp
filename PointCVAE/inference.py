import sys
import os

sys.path.append(os.getcwd())
import torch
import numpy as np
from PointCVAE import PointCVAENet
from utils import annotate


def load_model(model_path, requires_grad=False):
    model = PointCVAENet([1024, 512, 256], 1024, [512, 256, 128, 64, 6], 6, 64)
    param = torch.load(model_path)
    weights_dict = {}
    for k, v in param["network"].items():
        if "obj_encoder" in k:
            if "module" in k:
                k = k.replace("module.", "")
                weights_dict[k] = v
    weights_dict = {}
    for k, v in param["network"].items():
        if "cls_decoder" in k:
            continue
        elif "recon" in k:
            continue
        else:
            new_k = k.replace("module.", "") if "module" in k else k
            weights_dict[new_k] = v
    model.load_state_dict(weights_dict)
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model


def inference(model, obj_pc):
    # obj_pc : [B, 3, N]
    input_dim = obj_pc.dim()
    if input_dim == 2:
        obj_pc = obj_pc.unsqueeze(0)
    if obj_pc.shape[1] != 3:
        obj_pc = obj_pc.transpose(2, 1).contiguous()
    with torch.no_grad():
        _, _, cls_pred = model(obj_pc.detach())
    cls_pred = cls_pred.max(dim=1, keepdim=False)[1]
    if input_dim == 2:
        cls_pred = cls_pred.squeeze(0)
    return cls_pred.detach()


if __name__ == "__main__":
    model_path = "PointCVAE/model_best_val.pth"
    model = load_model(model_path)
    model.eval()

    save_root = "pre_results"
    K = 50
    mode = "train"
    idx = 50
    obj_pc = np.load("Data/processed/{}/{}/obj_pc/{}.npy".format(K, mode, idx))
    obj_pc_cls = np.load("Data/processed/{}/{}/obj_cls/{}.npy".format(K, mode, idx))
    hand_pc = np.load("Data/processed/{}/{}/hand_pc/{}.npy".format(K, mode, idx))
    obj_pc = torch.Tensor(obj_pc).unsqueeze_(0)
    hand_pc = torch.Tensor(hand_pc)
    obj_pc_cls = torch.Tensor(obj_pc_cls)

    cls_pred = inference(model, obj_pc)
    obj_pc_ = obj_pc.transpose(2, 1).detach()
    obj_colors = annotate.get_obj_colors(cls_pred)
    obj_colors_true = annotate.get_obj_colors(obj_pc_cls)
    pcd_hand = annotate.get_o3d_pcd(hand_pc, vis=False)
    pcd_obj = annotate.get_o3d_pcd(obj_pc_, obj_colors, vis=False)
    pcd_obj_true = annotate.get_o3d_pcd(obj_pc_, obj_colors_true, vis=False)
    annotate.vis_HandObject(pcd_hand, pcd_obj, window_name="pred_cls")

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    np.save(os.path.join(save_root, "{}.npy".format(idx)), cls_pred.cpu().numpy())
