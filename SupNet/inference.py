import sys
import os

sys.path.append(os.getcwd())
from SupNet import SupNet
from utils.loss import *


def load_model(model_path, requires_grad=False):
    param = torch.load(model_path)
    weights_dict = {}
    for k, v in param["network"].items():
        new_k = k.replace("module.", "") if "module" in k else k
        weights_dict[new_k] = v
    model = SupNet()
    model.load_state_dict(weights_dict)
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model


if __name__ == "__main__":

    model_path = "SupNet/model.pth"
    model = load_model(model_path)
    print(model)
