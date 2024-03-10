import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from PointCVAE import PointNetEncoder
from PointCVAE import VAE


class PointCVAENet(nn.Module):
    def __init__(
        self,
        cvae_encoder_sizes=[512, 512, 256],
        cvae_latent_size=1024,
        cvae_decoder_sizes=[512, 256, 128, 64, 6],
        cls_num=6,
        emb_dim=128,
    ):
        super(PointCVAENet, self).__init__()
        self.cvae_encoder_sizes = cvae_encoder_sizes
        self.cvae_encoder_sizes[0] = emb_dim
        self.cvae_latent_size = cvae_latent_size
        self.cvae_decoder_sizes = cvae_decoder_sizes
        self.cvae_decoder_sizes[0] = cvae_latent_size
        self.cvae_decoder_sizes[-1] = cls_num
        self.cvae_condition_size = 576
        self.cls_num = cls_num
        self.emb_dim = emb_dim

        self.cls_embedding = nn.Embedding(cls_num, emb_dim)
        self.obj_encoder = PointNetEncoder(
            global_feat=False, feature_transform=False, channel=3
        )

        self.cvae = VAE(
            encoder_layer_sizes=self.cvae_encoder_sizes,
            latent_size=self.cvae_latent_size,
            decoder_layer_sizes=self.cvae_decoder_sizes,
            condition_size=self.cvae_condition_size,
        )

    def forward(self, obj_pc, obj_cls=None):
        """
        :param obj_pc: [B, 3, N]
        :return: reconstructed object class
        """
        if len(obj_pc.shape) == 2:
            obj_pc = obj_pc.unsqueeze(0)

        x_feature, rot, tran, _ = self.obj_encoder(obj_pc)
        if self.training:
            obj_cls_emb = (
                self.cls_embedding(obj_cls).permute(0, 2, 1).contiguous()
            )  # [B,N]->[B,emb_dim,N]
            obj_cls_pred, means, log_var, z = self.cvae(obj_cls_emb, x_feature)
            return rot, tran, obj_cls_pred, means, log_var, z
        else:
            # inference
            obj_cls_pred = self.cvae.inference(obj_pc.shape[0], x_feature)
            return rot, tran, obj_cls_pred


if __name__ == "__main__":
    pass
