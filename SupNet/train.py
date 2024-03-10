import sys
import os

sys.path.append(os.getcwd())
import time
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from collections import defaultdict
from SupNet import obman
from SupNet import SupNet
import numpy as np
import random
from config import cfgs


def train(cfg, epoch, model, train_loader, device, optimizer, log_root):
    since = time.time()
    logs = defaultdict(list)
    model.train()

    for batch_idx, (obj_pc, hand_pc, true_cls) in enumerate(train_loader):
        obj_pc, hand_pc, true_cls = (
            obj_pc.to(device),
            hand_pc.to(device),
            true_cls.to(device),
        )
        optimizer.zero_grad()

        pred_cls, quat = model(obj_pc, hand_pc)
        rot_loss = torch.square(quat.norm(dim=1) - 1).sum() / obj_pc.shape[0]
        cls_loss = torch.nn.functional.cross_entropy(pred_cls, true_cls)
        loss = rot_loss + cls_loss
        loss.backward()
        optimizer.step()
        acc = (pred_cls.max(1)[1] == true_cls).sum() / true_cls.shape[0]
        logs["loss"].append(loss.item())
        logs["rot_loss"].append(rot_loss.item())
        logs["cls_loss"].append(cls_loss.item())
        logs["acc"].append(acc.item())
        if batch_idx % cfg.print_every == 0 or batch_idx == len(train_loader) - 1:
            print(
                "Train Epoch {:02d}/{:02d}, Batch {:04d}/{:d}, Total Loss {:9.5f}, rot loss {:9.5f}, cls loss{:9.5f}, acc{:9.5f}".format(
                    epoch,
                    cfg.epochs,
                    batch_idx,
                    len(train_loader) - 1,
                    loss.item(),
                    rot_loss.item(),
                    cls_loss.item(),
                    acc.item(),
                )
            )

    time_elapsed = time.time() - since
    out_str = "Epoch: {:02d}/{:02d}, train, time {:.0f}m, Mean Toal Loss {:9.5f}, rot loss {:9.5f}, cls loss{:9.5f}, acc{:9.5f}".format(
        epoch,
        cfg.epochs,
        time_elapsed // 60,
        sum(logs["loss"]) / len(logs["loss"]),
        sum(logs["rot_loss"]) / len(logs["rot_loss"]),
        sum(logs["cls_loss"]) / len(logs["cls_loss"]),
        sum(logs["acc"]) / len(logs["acc"]),
    )
    with open(log_root, "a") as f:
        f.write(out_str + "\n")


def val(cfg, epoch, model, val_loader, device, log_root, checkpoint_root, best_val_acc, mode="val"):
    # validation
    total_loss, total_rot_loss, total_cls_loss = 0.0, 0.0, 0.0
    acc_num, total_num = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (obj_pc, hand_pc, true_cls) in enumerate(val_loader):
            # obj_pc, hand_param, obj_cmap = obj_pc.to(device), hand_param.to(device), obj_cmap.to(device)
            obj_pc, hand_pc, true_cls = (
                obj_pc.to(device),
                hand_pc.to(device),
                true_cls.to(device),
            )
            optimizer.zero_grad()
            pred_cls, quat = model(obj_pc, hand_pc)  # recon [B,61] mano params
            rot_loss = torch.square(quat.norm(dim=1) - 1).sum() / obj_pc.shape[0]
            cls_loss = torch.nn.functional.cross_entropy(pred_cls, true_cls)
            loss = rot_loss + cls_loss
            acc_num += (pred_cls.max(1)[1] == true_cls).sum()
            total_num += true_cls.shape[0]
            total_loss += loss.item()
            total_rot_loss += rot_loss.item()
            total_cls_loss += cls_loss.item()

    mean_loss = total_loss / len(val_loader)
    mean_rot_loss = total_rot_loss / len(val_loader)
    mean_cls_loss = total_cls_loss / len(val_loader)
    acc = (acc_num / total_num).item()
    if acc > best_val_acc:
        best_val_acc = acc
        save_name = os.path.join(checkpoint_root, "model_best_{}.pth".format(str(acc)))
        torch.save({"network": model.state_dict(), "epoch": epoch}, save_name)

    out_str = "Epoch: {:02d}/{:02d}, {}, mean_loss {:9.5f}, mean_rot_loss {:9.5f}, mean_cls_loss {:9.5f}, acc {:9.5f}, Best Acc: {:9.5f},".format(
        epoch, cfg.epochs, mode, mean_loss, mean_rot_loss, mean_cls_loss, acc, best_val_acc
    )
    print(out_str)
    with open(log_root, "a") as f:
        f.write(out_str + "\n")

    return max(best_val_acc, acc)


if __name__ == "__main__":
    cfg = cfgs.supnet_config
    cfg.K = cfgs.obman_config.K
    del cfgs

    local_time = time.localtime(time.time())
    time_str = str(local_time[1]) + "_" + str(local_time[2]) + "_" + str(local_time[3])
    save_root = os.path.join("logs", cfg.model_type, time_str)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    log_root = save_root + "/log.txt"
    log_file = open(log_root, "w+")
    log_file.write(str(cfg) + "\n")
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
    model = SupNet().to(device)
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
        hand_pc_path = "Data/processed/{}/{}/hand_pc".format(cfg.K, "train")
        train_dataset = obman(obj_pc_path, hand_pc_path)
        train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.dataloader_workers)
    if "Val" in cfg.train_mode:
        obj_pc_path = "Data/processed/{}/{}/obj_pc".format(cfg.K, "val")
        hand_pc_path = "Data/processed/{}/{}/hand_pc".format(cfg.K, "val")
        val_dataset = obman(obj_pc_path, hand_pc_path)
        val_loader = DataLoader(dataset=val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.dataloader_workers)
    if "Test" in cfg.train_mode:
        obj_pc_path = "Data/processed/{}/{}/obj_pc".format(cfg.K, "test")
        hand_pc_path = "Data/processed/{}/{}/hand_pc".format(cfg.K, "test")
        eval_dataset = obman(obj_pc_path, hand_pc_path)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.dataloader_workers)

    # optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[round(cfg.epochs * x) for x in [0.3, 0.6, 0.8, 0.9]],
        gamma=0.5,
    )

    best_val_acc = 0
    best_eval_loss = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        print("Begin Trian epoch={}".format(epoch))
        if "Train" in cfg.train_mode:
            train(cfg, epoch, model, train_loader, device, optimizer, log_root)
            scheduler.step()
        if "Val" in cfg.train_mode:
            print("Begin Val epoch={}".format(epoch))
            best_val_acc = val(cfg, epoch, model, val_loader, device, log_root, save_root, best_val_acc, "val")
        if "Test" in cfg.train_mode:
            print("Begin Test epoch={}".format(epoch))
            best_val_acc = val(cfg, epoch, model, val_loader, device, log_root, save_root, best_val_acc, "test")
