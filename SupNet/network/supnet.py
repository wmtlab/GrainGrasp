import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from SupNet import PointNetEncoder


class SupNet(nn.Module):
    def __init__(self):
        super(SupNet, self).__init__()
        self.cls_embedding = nn.Embedding(2, 128)
        self.ho_encoder = PointNetEncoder(
            global_feat=True, feature_transform=False, channel=3 + 128
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, obj_pc, hand_pc):
        """
        :param obj_pc: [B, 3, N]
        :param hand_pc: [B, 778, 3]
        :return: reconstructed object class
        """
        B, _, N = obj_pc.shape
        obj_cls = torch.zeros(B, N).to(torch.long).to(obj_pc.device)
        hand_cls = torch.zeros(B, 778).to(torch.long).to(hand_pc.device)
        obj_emb = self.cls_embedding(obj_cls).transpose(1, 2)  # B*e*3000
        hand_emb = self.cls_embedding(hand_cls)  # B*778*e
        obj_feature = torch.cat((obj_pc, obj_emb), dim=1)
        hand_feature = torch.cat((hand_pc, hand_emb), dim=2).transpose(
            1, 2
        )  # B*(3+e)*778
        torch.cat((obj_feature, hand_feature), dim=2)
        ho_feature, quat, _, _ = self.ho_encoder(
            torch.cat((obj_feature, hand_feature), dim=2), stn=True
        )
        cls = self.decoder(ho_feature)
        return cls, quat


if __name__ == "__main__":
    # import time
    device = "cuda"
