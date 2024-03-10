import sys
import os

sys.path.append(os.getcwd())
import time
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from PointCVAE import obman
from PointCVAE import PointCVAENet
import numpy as np
import random
from utils import tools
from utils.loss import CVAE_loss, transform_loss
from config import cfgs


def train(cfg, epoch, model, train_loader, device, optimizer, log_root):
    since = time.time()
    logs = defaultdict(list)
    w_cross, w_dice, w_kd, w_rot = cfg.loss_weight
    model.train()
    for batch_idx, (obj_pc, obj_cls) in enumerate(train_loader):
        obj_pc, obj_cls = obj_pc.to(device), obj_cls.to(device)
        optimizer.zero_grad()
        rot, tran, obj_cls_pred, means, log_var, z = model(obj_pc, obj_cls)
        recon_cls_loss, KLD_loss = CVAE_loss(obj_cls_pred, obj_cls, means, log_var, w_cross, w_dice, "train")
        rot_loss = transform_loss(rot)
        loss = recon_cls_loss + w_kd * KLD_loss + w_rot * rot_loss  # rot_loss from training pointnet
        loss.backward()
        optimizer.step()
        logs["loss"].append(loss.item())
        logs["recon_cls_loss"].append(recon_cls_loss.item())
        logs["KLD_loss"].append(KLD_loss.item())
        logs["rot_loss"].append(rot_loss.item())
        if batch_idx % cfg.print_every == 0 or batch_idx == len(train_loader) - 1:
            print(
                "Train Epoch {:02d}/{:02d}, Batch {:04d}/{:d}, Total Loss {:9.5f}, Cls Loss {:9.5f}, KLD_loss {:9.5f}, rot_loss {:9.5f}".format(
                    epoch,
                    cfg.epochs,
                    batch_idx,
                    len(train_loader) - 1,
                    loss.item(),
                    recon_cls_loss.item(),
                    w_kd * KLD_loss.item(),
                    w_rot * rot_loss.item(),
                )
            )

    time_elapsed = time.time() - since
    out_str = "Epoch: {:02d}/{:02d}, train, time {:.0f}m, Mean Toal Loss {:9.5f}, Cls Loss {:9.5f}, KLD_loss {:9.5f}, rot_loss {:9.5f}".format(
        epoch,
        cfg.epochs,
        time_elapsed // 60,
        sum(logs["loss"]) / len(logs["loss"]),
        sum(logs["recon_cls_loss"]) / len(logs["recon_cls_loss"]),
        sum(logs["KLD_loss"]) / len(logs["KLD_loss"]),
        sum(logs["rot_loss"]) / len(logs["rot_loss"]),
    )
    with open(log_root, "a") as f:
        f.write(out_str + "\n")


def val(cfg, epoch, model, val_loader, device, log_root, checkpoint_root, best_val_loss, mode="val"):
    # validation
    w_cross, w_dice, w_kd, w_rot = cfg.loss_weight
    total_recon_cls_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (obj_pc, obj_cls) in enumerate(val_loader):
            obj_pc, obj_cls = obj_pc.to(device), obj_cls.to(device)
            _, _, obj_cls_pred = model(obj_pc, obj_cls)  # recon [B,61] mano params
            recon_cls_loss, _ = CVAE_loss(obj_cls_pred, obj_cls, None, None, w_cross, w_dice, "test")
            total_recon_cls_loss += recon_cls_loss.item()
    mean_recon_cls_loss = total_recon_cls_loss / len(val_loader)
    if mean_recon_cls_loss < best_val_loss:
        best_eval_loss = mean_recon_cls_loss
        save_name = os.path.join(checkpoint_root, "model_best_{}.pth".format(str(best_eval_loss)))
        torch.save({"network": model.state_dict(), "epoch": epoch}, save_name)

    out_str = "Epoch: {:02d}/{:02d}, {},  Best Recon cls Loss: {:9.5f}".format(epoch, cfg.epochs, mode, best_eval_loss)
    print(out_str)
    with open(log_root, "a") as f:
        f.write(out_str + "\n")
    return best_val_loss


if __name__ == "__main__":
    # config
    cfg = cfgs.cvae_config
    cfg.K = cfgs.obman_config.K
    del cfgs
    # log file
    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + "_" + str(local_time[2]) + "_" + str(local_time[3])
    model_info = "W{}".format(str(cfg.loss_weight))
    save_root = os.path.join("logs", cfg.model_type, time_str + "_" + model_info)
    tools.check_dir(save_root)
    log_root = save_root + "/log.txt"
    log_file = open(log_root, "w+")
    log_file.write(str(cfg) + "\n")
    log_file.write("weights for recon_cls_loss, KLD_loss, rot_loss are {}".format(str(cfg.loss_weight)) + "\n")
    log_file.close()

    # seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # device
    use_cuda = cfg.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device", device)
    device_num = 1

    # network
    model = PointCVAENet(
        cvae_encoder_sizes=list(cfg.encoder_layer_sizes),
        cvae_latent_size=cfg.latent_size,
        cvae_decoder_sizes=list(cfg.decoder_layer_sizes),
        cls_num=cfg.cls_num,
        emb_dim=cfg.emb_dim,
    ).to(device)

    # multi-gpu
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model)
            device_num = len(device_ids)

    # dataset
    if "Train" in cfg.train_mode:
        obj_pc_path = "Data/processed/{}/{}/obj_pc".format(cfg.K, "train")
        obj_cls_path = "Data/processed/{}/{}/obj_cls".format(cfg.K, "train")
        train_dataset = obman(obj_pc_path, obj_cls_path)
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.dataloader_workers)
    if "Val" in cfg.train_mode:
        obj_pc_path = "Data/processed/{}/{}/obj_pc".format(cfg.K, "val")
        obj_cls_path = "Data/processed/{}/{}/obj_cls".format(cfg.K, "val")
        val_dataset = obman(obj_pc_path, obj_cls_path)
        val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.dataloader_workers)
    if "Test" in cfg.train_mode:
        obj_pc_path = "Data/processed/{}/{}/obj_pc".format(cfg.K, "test")
        obj_cls_path = "Data/processed/{}/{}/obj_cls".format(cfg.K, "test")
        eval_dataset = obman(obj_pc_path, obj_cls_path)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.dataloader_workers)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[round(cfg.epochs * x) for x in [0.3, 0.6, 0.8, 0.9]],
        gamma=0.5,
    )

    best_val_loss = float("inf")
    best_eval_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        print("Begin Trian epoch={}".format(epoch))
        if "Train" in cfg.train_mode:
            train(cfg, epoch, model, train_loader, device, optimizer, log_root)
            scheduler.step()
        if "Val" in cfg.train_mode or "Test" in cfg.train_mode:
            print("Begin Val epoch={}".format(epoch))
            best_val_loss = val(cfg, epoch, model, val_loader, device, log_root, save_root, best_val_loss, "val")
        if "Test" in cfg.train_mode:
            best_val_loss = val(cfg, epoch, model, eval_loader, device, log_root, save_root, best_val_loss, "test")
