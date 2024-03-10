import torch
import torch.nn.functional as F


class DiceLoss(torch.nn.Module):
    def __init__(self, n_classes, weight=None):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        if weight == None:
            self.weights = [i for i in range(n_classes)]
        else:
            self.weights = weight

    def forward(self, input, target):
        target = F.one_hot(target).transpose(2, 1)
        input = F.softmax(input, dim=1)
        total_dice_loss = 0
        for i in range(self.n_classes):
            numerator = 2 * torch.sum(input[:, i] * target[:, i])
            denominator = torch.sum(input[:, i]) + torch.sum(target[:, i])
            loss = 1 - numerator / denominator
            total_dice_loss += loss * self.weights[i]

        return total_dice_loss / self.n_classes


dice_loss = DiceLoss(6)


def CVAE_loss(recon_cls, cls, mean, log_var, w_cross, w_dice, mode="train"):
    """
    :param recon_x: reconstructed hand xyz [B,778,3]
    :param x: ground truth hand xyz [B,778,3]
    :param recon_cls: reconstructed cls [B,6,N]
    :param cls: ground truth hand xyz [B,N]
    :param mean: [B,z]
    :param log_var: [B,z]
    :return:
    """
    cross_loss = F.cross_entropy(
        recon_cls,
        cls,
        weight=torch.Tensor([0.05, 0.4, 0.4, 0.5, 0.6, 0.6]).to(cls.device),
    ).sum()
    recon_cls_loss = w_cross * cross_loss + w_dice * dice_loss(recon_cls, cls)

    if mode != "train":
        return recon_cls_loss, None
    else:
        # KLD loss
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) / log_var.shape[0] * 2
        return recon_cls_loss, KLD


def transform_loss(rot):
    d = rot.size()[1]
    I = torch.eye(d)[None, :, :]
    if rot.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(rot, rot.transpose(2, 1)) - I, dim=(1, 2)))
    return loss
