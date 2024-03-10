import sys
import os

sys.path.append(os.getcwd())
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.ops.knn import knn_points
import mano
from attrdict import AttrDict
from pytorch3d import transforms
from utils import tools
from SupNet import load_model as load_supnet_model


class DCoGModel:
    def __init__(
        self,
        mano_path,
        init_handpose_path,
        init_quat_path,
        finger_index_path,
        tip_index_path,
        supnet_path,
        init_move_finger_idx=3,
        weights=None,
        device="cuda",
    ):
        self.device = device
        self.rh_mano, self.rh_faces = self.load_mano(mano_path)
        self.init_handpose = torch.tensor(np.load(init_handpose_path)).to(self.device)
        self.init_quat = torch.tensor(np.load(init_quat_path)).to(self.device)
        self.init_handpose = self.init_handpose.repeat(self.init_quat.shape[0], 1)
        self.rh_faces = self.rh_faces.repeat(self.init_quat.shape[0], 1, 1)
        self.finger_index = tools.fingerName2fingerId(tools.readJson(finger_index_path))
        self.tip_index = tools.fingerName2fingerId(tools.readJson(tip_index_path))

        self.hand_normal = None
        sup_net = load_supnet_model(supnet_path, requires_grad=False)
        self.sup_net = sup_net.eval().to(device)
        self.init_move_finger_idx = init_move_finger_idx
        if weights is None:
            weights = AttrDict(
                {"w_dis": 0.5, "w_dct": 0.8, "w_dcf": 0.6, "w_net": 0.6, "w_pen": 10}
            )
        self.weights = weights

    def run(self, obj_pc, obj_cls, epochs=300, select_finger_idx=[1, 2, 3, 4, 5]):
        """
        obj_pc: [N, 3]
        obj_cls: [N]
        """
        record_hand_pc = []
        concat_center = self.get_center_contactmaps(obj_cls, obj_pc)
        init_tran = self.get_init_translation(concat_center)
        init_tran = init_tran * torch.tensor([[1.2, 1.2, 1.2]]).to(self.device)
        init_tran = torch.autograd.Variable(init_tran, requires_grad=True)
        quat_rt = torch.autograd.Variable(
            torch.Tensor([[1, 0, 0, 0, 0, 0, 0]])
            .repeat(init_tran.shape[0], 1)
            .to(self.device),
            requires_grad=True,
        )
        init_quat = torch.autograd.Variable(self.init_quat, requires_grad=True)
        handpose = torch.autograd.Variable(self.init_handpose, requires_grad=True)
        optimizer = torch.optim.Adam([handpose, quat_rt, init_quat, init_tran], lr=0.01)

        for epoch in range(epochs):
            optimizer.zero_grad()
            q_rot, tran = quat_rt[:, :4], quat_rt[:, 4:]
            q_rot = transforms.quaternion_multiply(q_rot, init_quat)
            tran = tran + init_tran
            m_rot = transforms.quaternion_to_matrix(q_rot)
            hand_pc_opt = self.get_hand_pc(handpose, m_rot, tran)
            record_hand_pc.append(hand_pc_opt.clone().detach().cpu())
            mesh_p3d = Meshes(verts=hand_pc_opt, faces=self.rh_faces)
            self.hand_normal = mesh_p3d.verts_normals_packed().view(-1, 778, 3)
            E_dis = self.cal_Edis(obj_pc, obj_cls, hand_pc_opt, select_finger_idx)
            E_dct = self.cal_Edct(obj_pc, obj_cls, hand_pc_opt)
            E_dcf = self.cal_Edcf(obj_pc, hand_pc_opt)
            E_net = self.cal_Enet(obj_pc, hand_pc_opt)
            E_pen = self.cal_Epen(obj_pc, hand_pc_opt)
            E = (
                (epoch + 1) * self.weights.w_dis * E_dis
                + max(0, epochs - epoch)
                * (self.weights.w_dct * E_dct + self.weights.w_dcf * E_dcf)
                + self.weights.w_net * E_net
                + self.weights.w_pen * E_pen
            )
            E.backward()
            optimizer.step()
        hand_pc_opt = self.get_hand_pc(handpose, m_rot, tran)
        record_hand_pc.append(hand_pc_opt.clone().detach().cpu())
        result = AttrDict()
        result.record_hand_pc = torch.stack(record_hand_pc)  # [epochs+1, B, 778, 3]
        result.hand_pc = hand_pc_opt.detach().cpu()
        result.handpose = handpose.detach().cpu()
        result.rot = m_rot.detach().cpu()
        result.translation = tran.detach().cpu()

        return result

    def get_idx_minEpen(self, obj_pc, hand_pose, threshold=0.0):
        self.hand_normal = None  # recalculate hand normal
        E_pen = self.cal_Epen(obj_pc, hand_pose.to(self.device), reuturn_batch=True)
        re_E_pen = E_pen.tolist()
        E_pen[E_pen <= threshold] = torch.inf
        min_idx = E_pen.argmin().item()
        return re_E_pen, min_idx

    def load_mano(self, mano_model_path):
        with torch.no_grad():
            rh_mano = mano.load(
                model_path=mano_model_path,
                model_type="mano",
                use_pca=True,
                num_pca_comps=45,
                batch_size=6,
                flat_hand_mean=True,
            ).to(self.device)
            rh_faces = (
                torch.tensor(rh_mano.faces.astype(int)).unsqueeze(0).to(self.device)
            )
        return rh_mano, rh_faces

    def get_hand_pc(self, pose, m_rot, tran=None):
        hand_pc = self.rh_mano(hand_pose=pose).vertices
        if tran == None:
            return torch.bmm(hand_pc, m_rot)
        else:
            return torch.bmm(hand_pc, m_rot) + tran.reshape(-1, 1, 3)

    def get_center_contactmaps(self, obj_cls, obj_pc):
        concat_center = torch.zeros((5, 3)).to(obj_pc.device)
        for i in range(1, 6):
            concat_center[i - 1] = obj_pc[obj_cls == i].mean(dim=0)
        return concat_center

    def get_init_translation(self, concat_center, init_move_finger_idx=-1):
        if init_move_finger_idx == -1:
            init_move_finger_idx = self.init_move_finger_idx
        hand_pc = self.rh_mano(hand_pose=self.init_handpose).vertices
        m_rot = transforms.quaternion_to_matrix(self.init_quat)
        hand_pc = torch.bmm(hand_pc, m_rot)
        select_finger_index = self.finger_index[init_move_finger_idx]
        concat_center = concat_center[init_move_finger_idx].repeat(6, 1)
        tran = concat_center - hand_pc[:, select_finger_index].mean(
            dim=1, keepdim=False
        )
        return tran

    def cal_Edis(self, obj_pc, obj_cls, hand_pc, select_idx=[1, 2, 3, 4, 5]):
        E = 0
        for fingerId, ft_idx in self.tip_index.items():
            if fingerId in select_idx:
                obj_idx_pc = obj_pc[obj_cls == fingerId].repeat(hand_pc.shape[0], 1, 1)
                if obj_idx_pc.shape[1] == 0:
                    continue
                e = knn_points(hand_pc[:, ft_idx], obj_idx_pc, K=1).dists
                e = torch.dropout(e, p=0.2, train=True).sum(dim=0).mean()
            E += e
        return E * 500

    def cal_Edct(self, obj_pc, obj_cls, hand_pc):
        if self.hand_normal is None:
            mesh = Meshes(verts=hand_pc, faces=self.rh_faces[: hand_pc.shape[0]])
            self.hand_normal = (
                mesh.verts_normals_packed().view(-1, 778, 3).to(self.device)
            )

        E = 0
        for fingerId, ft_idx in self.tip_index.items():
            tip_idx_pc = hand_pc[:, ft_idx]
            obj_idx_pc = obj_pc[obj_cls == fingerId].repeat(hand_pc.shape[0], 1, 1)
            if obj_idx_pc.shape[1] == 0:
                continue
            _, _, nn = knn_points(tip_idx_pc, obj_idx_pc, K=1, return_nn=True)
            idxtip2obj_normal = F.normalize(nn.squeeze(-2) - tip_idx_pc, dim=2)
            idxtip_normal = F.normalize(self.hand_normal[:, ft_idx], dim=2)
            e = torch.square(idxtip2obj_normal - idxtip_normal)
            e = torch.dropout(e, p=0.2, train=True).sum(dim=0).mean()
            E += e

        return E * 0.5

    def cal_Edcf(self, obj_pc, hand_pc):
        if self.hand_normal is None:
            mesh = Meshes(verts=hand_pc, faces=self.rh_faces[: hand_pc.shape[0]])
            self.hand_normal = (
                mesh.verts_normals_packed().view(-1, 778, 3).to(self.device)
            )

        finger_index_all = []
        for _, ft_idx in self.finger_index.items():
            finger_index_all.extend(ft_idx)
        finger_pc = hand_pc[:, finger_index_all]
        finger_normal = F.normalize(self.hand_normal[:, finger_index_all], dim=2)
        _, _, nn = knn_points(
            finger_pc,
            obj_pc.repeat(hand_pc.shape[0], 1, 1).contiguous(),
            return_nn=True,
        )
        finger2obj_normal = F.normalize(nn.squeeze(-2) - finger_pc, dim=2)
        E = torch.square(finger2obj_normal - finger_normal)
        E = torch.dropout(E, p=0.2, train=True).sum(dim=0).mean()
        return E

    def cal_Enet(self, obj_pc, hand_pc):
        obj_pc = obj_pc.T
        obj_pc = obj_pc.repeat(hand_pc.shape[0], 1, 1)
        net_pred, _ = self.sup_net(obj_pc, hand_pc)
        E = torch.nn.functional.cross_entropy(
            net_pred,
            torch.ones((net_pred.shape[0]), dtype=torch.long).to(net_pred.device),
            reduction="sum",
        )
        return E / obj_pc.shape[0]

    def cal_Epen(self, obj_pc, hand_pc, reuturn_batch=False):
        """

        get penetrate object xyz and the distance to its NN
        :param hand_pc: [B, 778, 3]
        :param hand_face: [B, 1538, 3], hand faces vertex index in [0:778]
        :param obj_pc: [B, 3000, 3]
        :return: inter penetration loss
        """
        if self.hand_normal is None:
            mesh = Meshes(verts=hand_pc, faces=self.rh_faces[: hand_pc.shape[0]])
            self.hand_normal = (
                mesh.verts_normals_packed().view(-1, 778, 3).to(self.device)
            )
        obj_pc = obj_pc.repeat(hand_pc.shape[0], 1, 1)
        B = hand_pc.size(0)
        nn_dist, nn_idx, _ = knn_points(obj_pc, hand_pc)
        nn_idx = nn_idx.repeat(1, 1, 3)
        hand_idx_pc = hand_pc.gather(dim=1, index=nn_idx)
        obj2hand_normal = hand_idx_pc - obj_pc
        hand_idx_normal = self.hand_normal.gather(dim=1, index=nn_idx)
        interior = (obj2hand_normal * hand_idx_normal).sum(
            dim=-1
        ) > 0  # interior as true, exterior as false
        E = nn_dist.squeeze(-1) * interior * 1e4
        if reuturn_batch:
            return E.sum(dim=1) / B
        else:
            return E.sum() / B

    def cal_Edc_Edis(
        self,
        obj_pc,
        obj_cls,
        hand_pc,
        select_idx=[1, 2, 3, 4, 5],
        weight=[0.5, 0.8, 0.6],
    ):
        E = 0
        mesh = Meshes(verts=hand_pc, faces=self.rh_faces)
        self.hand_normal = mesh.verts_normals_packed().view(-1, 778, 3)
        obj_pc = obj_pc.repeat(hand_pc.shape[0], 1, 1).contiguous()
        finger_index_all = []
        for fingerId, ft_idx in self.tip_index.items():
            finger_index_all.extend(self.finger_index[fingerId])
            # E_dct
            tip_idx_pc = hand_pc[:, ft_idx]
            obj_idx_pc = obj_pc[:, obj_cls == fingerId]
            if obj_idx_pc.shape[1] == 0:
                continue
            dists, _, nn = knn_points(tip_idx_pc, obj_idx_pc, K=1, return_nn=True)
            tipidx2obj_normal = F.normalize(nn.squeeze(-2) - tip_idx_pc, dim=2)
            tip_idx_normal = F.normalize(self.hand_normal[:, ft_idx], dim=2)
            e_dct = torch.square(tipidx2obj_normal - tip_idx_normal)
            e_dct = torch.dropout(e_dct, p=0.2, train=True).sum(dim=0).mean() * 0.5
            E += e_dct * weight[0]
            # E_dis
            if fingerId in select_idx:
                e_dis = torch.dropout(dists, p=0.2, train=True).sum(dim=0).mean() * 500
                E += e_dis * weight[1]
        # E_dcf
        finger_pc = hand_pc[:, finger_index_all]
        finger_normal = F.normalize(self.hand_normal[:, finger_index_all], dim=2)
        _, _, nn = knn_points(finger_pc, obj_pc, return_nn=True)
        finger2obj_normal = F.normalize(nn.squeeze(-2) - finger_pc, dim=2)
        e_dcf = torch.square(finger2obj_normal - finger_normal)
        e_dcf = torch.dropout(e_dcf, p=0.2, train=True).sum(dim=0).mean()
        E += e_dcf * weight[2]
        return E
