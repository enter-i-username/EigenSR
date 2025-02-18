from .svd import SVD
from .norm import Norm, MinMax
import copy
import os
from datetime import datetime
from torchvision.transforms import functional as F


def minmax(x):
    return (x - x.min()) / (x.max() - x.min())


def dict_shallow_copy(_dict):
    return {_k: _v for _k, _v in _dict.items()}


def copy_freeze_model(model, device):
    model_copy = copy.deepcopy(model)

    for param in model_copy.parameters():
        param.requires_grad = False

    model_copy = model_copy.to(device)

    return model_copy


def make_dirs_if_not_exist(path):
    _, extension = os.path.splitext(path)

    if extension:
        parent_directory = os.path.dirname(path)
        if not os.path.exists(parent_directory):
            os.makedirs(parent_directory)
            return True
        else:
            return False
    else:
        if not os.path.exists(path):
            os.makedirs(path)
            return True
        else:
            return False


def get_current_time_str():
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{current_time}".replace('-', '_')
    return folder_name


def pad_if_smaller(img, size):
    *_, ih, iw = img.shape
    sh, sw = size

    if ih < sh:
        ph = sh - ih
        pt = ph // 2
        pb = ph - pt
    else:
        pt = pb = 0

    if iw < sw:
        pw = sw - iw
        pl = pw // 2
        pr = pw - pl
    else:
        pl = pr = 0

    img = F.pad(img, padding=[pl, pt, pr, pb], padding_mode='reflect')
    return img


def crop_center_if_larger(img, size, scale=1):
    *_, ih, iw = img.shape
    sh, sw = size

    if ih > sh:
        ch = ih - sh
        ch //= scale
        h_start = ch // 2 * scale
        h_end = h_start + sh
    else:
        h_start = h_end = None

    if iw > sw:
        cw = iw - sw
        cw //= scale
        w_start = cw // 2 * scale
        w_end = w_start + sw
    else:
        w_start = w_end = None

    return img[..., h_start: h_end, w_start: w_end]

