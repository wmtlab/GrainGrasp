import torch
import numpy as np
from pytorch3d.ops import knn_points

# device = "cuda" if torch.cuda.is_available() else "cpu"

tip_index = {
    "thumb": [740, 743, 756, 760, 762, 763, 768, 767, 739],
    "index": [328, 329, 332, 343, 347, 349, 350, 354, 355],
    "middle": [455, 459, 461, 462, 466, 435, 436, 467, 438, 439, 442],
    "ring": [549, 550, 553, 566, 569, 570, 571, 572, 573, 577, 578],
    "pinky": [687, 689, 690, 683],
}

finger_colors = {
    "thumb": [1.0, 0.0, 0.0],
    "index": [0.0, 1.0, 0.0],
    "middle": [0.0, 0.0, 1.0],
    "ring": [1.0, 1.0, 0.0],
    "pinky": [1.0, 0.0, 1.0],
}

finger_cls = {"thumb": 1, "index": 2, "middle": 3, "ring": 4, "pinky": 5}
finger_names = list(finger_cls.keys())


def get_obj_cls_and_colors(hand_points, obj_points, K=50, input_finger_colors=None, device="cuda"):
    """
    :param hand_points: Tensor,(B,778,3)
    :param obj_points: Tensor,(B,N,3)
    :param K: Int, for input of Knn.
    :return: obj_cls: Tensor-(B,N), colors: Tensor-(B,N,3)
    """
    input_finger_colors = finger_colors if input_finger_colors == None else input_finger_colors

    if len(hand_points.shape) == 2:
        hand_points = hand_points.unsqueeze(0)
    if len(obj_points.shape) == 2:
        obj_points = obj_points.unsqueeze(0)
    assert hand_points.shape[0] == obj_points.shape[0]

    obj_cls = np.zeros((*obj_points.shape[:2],), dtype=np.int32)
    colors = np.zeros_like(obj_points.cpu())
    _, idx_Batch, _ = knn_points(hand_points.to(device), obj_points.to(device), K=K)  # cuda
    idx_Batch = idx_Batch.cpu().numpy()

    for b in range(hand_points.shape[0]):
        idx = idx_Batch[b]  # (N, K)
        obj_index = dict()
        for i in range(len(finger_cls)):
            finger = finger_names[i]
            obj_index_finger = np.unique(idx[tip_index[finger]])  # (len(index),K)
            obj_index[finger] = obj_index_finger
            for j in range(i):
                inter_index = np.intersect1d(obj_index[finger_names[j]], obj_index_finger)  # (?,)
                split = len(tip_index[finger])
                if len(inter_index) > 0:

                    """Calculate the minimum distance between the current finger and
                    another finger with intersection using these points, and assign
                    them to the corresponding finger based on the smaller distance."""
                    two_index = tip_index[finger] + tip_index[finger_names[j]]
                    _, idx_min, _ = knn_points(
                        obj_points[b : b + 1, inter_index].cpu(),
                        hand_points[b : b + 1, two_index].cpu(),
                    )
                    idx_min = idx_min.numpy().reshape((-1,))
                    now_index = idx_min < split
                    past_index = np.logical_not(now_index)
                    # Remove the current set from the previous set
                    obj_index[finger_names[j]] = np.setdiff1d(obj_index[finger_names[j]], inter_index[now_index])
                    # Remove the previous set from the current set
                    obj_index[finger] = np.setdiff1d(obj_index[finger], inter_index[past_index])

        for finger in obj_index:
            index = obj_index[finger]
            obj_cls[b][index] = finger_cls[finger]
            colors[b][index] = finger_colors[finger]

    return torch.tensor(obj_cls), torch.tensor(colors)


def get_finger_colors(hand_points, input_finger_colors=None):
    """
    Get colors of fingertips
    :param hand_points: (N,3) or (B,N,3)
    :return: colors: (N,3) or (B,N,3)
    """
    input_finger_colors = finger_colors if input_finger_colors == None else input_finger_colors
    colors = np.zeros_like(hand_points)
    for finger in finger_colors:
        colors[tip_index[finger]] = finger_colors[finger]
    return colors


def get_obj_colors(obj_cls, cls_colors=None):

    cls_colors = finger_colors if cls_colors == None else cls_colors
    # if obj_cls.shape[-1] == 1:
    #     obj_cls = obj_cls.reshape()
    # obj_cls = obj_cls.squeeze()
    # if len(obj_cls.shape) == 1:
    #     obj_cls = obj_cls.reshape(1, -1)
    if obj_cls.shape[-1] == 1:  # (B,N,1) or (N,1)
        obj_cls = obj_cls.reshape(*obj_cls.shape[:-1])  # (B,N) or (N,)
    obj_colors = np.zeros((*obj_cls.shape, 3))  # (B,N,3) or (N,3)

    for finger in tip_index:
        obj_colors[obj_cls == finger_cls[finger]] = cls_colors[finger]
    return obj_colors
