from torch.nn import functional as F
import torch
import utils
from utils import imresize
from collections import deque
from torchvision.transforms import functional
import numpy as np


@torch.no_grad()
def iterative_sr(
        deg_hsi_batch: torch.Tensor,
        iters: int,
        beta: float,
        **hsi_inference_kwargs
):
    if hsi_inference_kwargs['transform']:
        hsi_inference_kwargs['transform'] = _ImageCycleTransforms()
    else:
        hsi_inference_kwargs['transform'] = None

    Y_ = deg_hsi_batch

    for _i in range(iters):
        Y = hsi_inference(hsi_batch=deg_hsi_batch, ref_batch=Y_, **hsi_inference_kwargs)
        if iters == 1:
            return Y

        b, c, h, w = Y.shape
        b, c, h_, w_ = Y_.shape
        if h_ != h or w_ != w:
            Y_ = imresize.imresize(Y_, sizes=(h, w))

        Y_ = beta * Y + (1 - beta) * Y_

    return Y_


@torch.no_grad()
def hsi_inference(
        hsi_batch,  # SR source image
        ref_batch,  # SVD reference image
        model,
        scale,
        batch_size,
        patch_size=(48, 48),
        step_size=(32, 32),
        eigen_dims=None,
        transform=None
):

    b, c, h, w = hsi_batch.shape
    b, c, rh, rw = ref_batch.shape

    if isinstance(eigen_dims, float) and 0 < eigen_dims < 1:
        eigen_dims = int(np.ceil(eigen_dims * c))

    # SVD transform
    svd_objs, batch_f = [], []
    for _b in range(b):
        hsi_hwc = hsi_batch[_b].permute(1, 2, 0)
        ref_hwc = ref_batch[_b].permute(1, 2, 0)
        svd = utils.SVD(ref_hwc, svd_solver=torch.linalg.svd)
        # eigen_hwf.shape = (h, w, eigen_dims)
        eigen_hwf = svd.transform(hsi_hwc, eigen_dims)
        svd_objs.append(svd)
        batch_f.append(eigen_hwf)
    # batch_f.shape = (b, eigen_dims, h, w)
    batch_hsi = []
    batch_f = torch.stack(batch_f).permute(0, 3, 1, 2)

    # Norm transform
    # norm_obj = utils.Norm(
    #     batch_f,
    #     mean_func=torch.mean, std_func=torch.std, dims=[-2, -1]
    # )
    # batch_f = norm_obj.transform(batch_f)

    # Geometric transform
    if transform is not None:
        batch_f = transform.transform(batch_f)

    # SVD eigen inference
    batch_f = gray_batch_sr(batch_f, model, scale, batch_size, patch_size, step_size)

    # Geometric inverse
    if transform is not None:
        batch_f = transform.inverse_transform(batch_f)

    # Norm inverse
    # batch_f = norm_obj.inverse_transform(batch_f)

    # SVD inverse
    for _b in range(b):
        eigen_hwf = batch_f[_b].permute(1, 2, 0)
        svd = svd_objs[_b]
        # hsi_hwc.shape = (h, w, c)
        hsi_hwc = svd.inverse_transform(eigen_hwf)
        batch_hsi.append(hsi_hwc)
    batch_hsi = torch.stack(batch_hsi).permute(0, 3, 1, 2)

    return batch_hsi


@torch.no_grad()
def gray_batch_sr(
        gray_batch,
        model,
        scale,
        batch_size,
        patch_size=(48, 48),
        step_size=(32, 32)
):
    B, C, H, W = gray_batch.shape
    model.eval()

    # Right values
    gray_batch_r = gray_batch

    B, C, total_h_r, total_w_r = gray_batch_r.shape
    patch_h_r, patch_w_r = patch_size
    step_h_r, step_w_r = step_size
    pad_t_r, pad_b_r = _get_padding_sizes(total_h_r, patch_h_r, step_h_r)
    pad_l_r, pad_r_r = _get_padding_sizes(total_w_r, patch_w_r, step_w_r)

    # Left values
    total_h_l, total_w_l, \
    patch_h_l, patch_w_l, \
    step_h_l, step_w_l, \
    pad_t_l, pad_b_l, pad_l_l, pad_r_l = _get_left_sizes(
        scale,
        total_h_r, total_w_r,
        patch_h_r, patch_w_r,
        step_h_r, step_w_r,
        pad_t_r, pad_b_r, pad_l_r, pad_r_r
    )

    gray_batch_l = torch.zeros(
        B, C,
        total_h_l + pad_t_l + pad_b_l, total_w_l + pad_l_l + pad_r_l,
        device=gray_batch_r.device
    )

    # Unfold the right
    gray_batch_r = F.pad(gray_batch_r, (pad_l_r, pad_r_r, pad_t_r, pad_b_r), mode='reflect')
    num_patch_h = (gray_batch_r.shape[-2] - patch_h_r) // step_h_r + 1
    num_patch_w = (gray_batch_r.shape[-1] - patch_w_r) // step_w_r + 1

    # gray_batch_r.shape = (B, C * patch_h_r * patch_w_r, num_patch_h * num_patch_w)
    gray_batch_r = F.unfold(gray_batch_r, kernel_size=(patch_h_r, patch_w_r), stride=(step_h_r, step_w_r))
    # gray_batch_r.shape = (B, C, patch_h_r, patch_w_r, num_patch_h * num_patch_w)
    gray_batch_r = gray_batch_r.reshape((B, C, patch_h_r, patch_w_r, num_patch_h * num_patch_w))
    # gray_batch_r.shape = (num_patch_h * num_patch_w * B * C, patch_h_r, patch_w_r)
    gray_batch_r = gray_batch_r.permute((4, 0, 1, 2, 3)).reshape((-1, patch_h_r, patch_w_r))

    # SR the right with batch_size to the left
    BC = B * C
    queue = _BatchQueue()
    assign_add = _PatchAssignAdd(
        total_h_l, total_w_l,
        step_h_l, step_w_l,
        pad_t_l, pad_b_l, pad_l_l, pad_r_l
    )
    for _b in range(0, gray_batch_r.shape[0], batch_size):
        # _gray_batch_l.shape = (batch_size, patch_h_l, patch_w_l)
        _gray_batch_l = gray_batch_r[_b: _b + batch_size, ...]
        _norm_obj = utils.Norm(
            _gray_batch_l,
            mean_func=torch.mean, std_func=torch.std, dims=[-2, -1]
        )
        _gray_batch_l = _norm_obj.transform(_gray_batch_l)
        # import matplotlib.pyplot as plt
        # plt.subplot(1, 2, 1)
        # plt.imshow(_gray_batch_l[0, ...].cpu())
        _gray_batch_l = model(_gray_batch_l.unsqueeze(1)).squeeze(1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(_gray_batch_l[0, ...].cpu())
        # plt.show()
        _gray_batch_l = _norm_obj.inverse_transform(_gray_batch_l)

        queue.append(_gray_batch_l)
        while queue.num >= BC:
            # _gray_BC_l.shape = (B * C, patch_h_l, patch_w_l)
            _gray_BC_l = queue.pop(BC)
            # _gray_BC_l.shape = (B, C, patch_h_l, patch_w_l)
            _gray_BC_l = _gray_BC_l.reshape((B, C, patch_h_l, patch_w_l))

            assign_add.iadd_(gray_batch_l, _gray_BC_l)

    assign_add.avg_(gray_batch_l)
    return gray_batch_l[..., pad_t_l: pad_t_l + total_h_l, pad_l_l: pad_l_l + total_w_l]


def _get_padding_sizes(_total, _patch, _step):
    if _total > _patch:
        _pad = (_total - _patch) % _step
        if _pad:
            _pad = _step - _pad
    else:
        _pad = _patch - _total

    _pad_1 = _pad // 2
    _pad_2 = _pad - _pad_1
    return _pad_1, _pad_2


def _get_left_sizes(_scale, *_right_sizes):
    return tuple(_s_r * _scale for _s_r in _right_sizes)


class _BatchQueue:

    def __init__(self):
        self.num = 0
        self.queue = deque()

    def append(self, _tensor):
        self.num += _tensor.shape[0]
        self.queue.append(_tensor)

    def pop(self, _batch_size):
        _lst = list()
        while _batch_size > 0:
            _tensor = self.queue.popleft()
            if _tensor.shape[0] > _batch_size:
                _lst.append(_tensor[:_batch_size, ...])
                self.queue.appendleft(_tensor[_batch_size:, ...])
                self.num -= _batch_size
                _batch_size = 0
            else:
                _lst.append(_tensor)
                self.num -= _tensor.shape[0]
                _batch_size -= _tensor.shape[0]
        return torch.cat(_lst, 0)


class _PatchAssignAdd:

    def __init__(self,
                 t_h_size, t_w_size,
                 t_h_step, t_w_step,
                 t_t_pad, t_b_pad, t_l_pad, t_r_pad):

        self.t_h_size = t_h_size
        self.t_w_size = t_w_size

        self.t_h_step = t_h_step
        self.t_w_step = t_w_step

        self.t_h_ptr = 0
        self.t_w_ptr = 0

        self.t_t_pad = t_t_pad
        self.t_b_pad = t_b_pad
        self.t_l_pad = t_l_pad
        self.t_r_pad = t_r_pad

        self.avg_map = torch.zeros(
            t_h_size + t_t_pad + t_b_pad,
            t_w_size + t_l_pad + t_r_pad
        )

    def iadd_(self, _target, _patch):
        *_, _patch_h, _patch_w = _patch.shape

        _h_slice = slice(self.t_h_ptr, self.t_h_ptr + _patch_h)
        _w_slice = slice(self.t_w_ptr, self.t_w_ptr + _patch_w)
        _target[..., _h_slice, _w_slice] += _patch#= _patch#+= _patch
        self.avg_map[_h_slice, _w_slice] += 1

        self.t_w_ptr += self.t_w_step
        if self.t_w_ptr >= self.t_w_size + self.t_l_pad + self.t_r_pad - _patch_w + self.t_w_step:
            self.t_w_ptr = 0
            self.t_h_ptr += self.t_h_step

    def avg_(self, gray_batch_l):
        shape_len = len(gray_batch_l.shape)
        _map = self.avg_map.reshape(
            *(1 for _ in range(shape_len - 2)), *self.avg_map.shape
        ).to(gray_batch_l.device)

        gray_batch_l /= _map
        pass


# Improves the model's robustness
class _ImageCycleTransforms:

    def __init__(self):
        self.current_T = 0

        self.Ts = [
            self._do_nothing,
            self._rotate90,
            self._vflip,
            self._rotate180,
            self._hflip,
            self._rotate270,
            self._transpose,
        ]
        self.ITs = [
            self._do_nothing,
            self._rotate270,
            self._vflip,
            self._rotate180,
            self._hflip,
            self._rotate90,
            self._transpose,
        ]

    @staticmethod
    def _do_nothing(x):
        return x

    @staticmethod
    def _rotate90(x):
        *_, _h, _w = x.shape
        x = functional.rotate(x, 90, expand=True)
        *_, __w, __h = x.shape
        if __w > _w or __h > _h:
            x = x[..., __w-_w:, :_h]
        return x

    @staticmethod
    def _rotate180(x):
        return functional.rotate(x, 180, expand=True)

    @staticmethod
    def _rotate270(x):
        *_, _h, _w = x.shape
        x = functional.rotate(x, 270, expand=True)
        *_, __w, __h = x.shape
        if __w > _w or __h > _h:
            x = x[..., :_w, __h-_h:]
        return x

    @staticmethod
    def _hflip(x):
        return functional.hflip(x)

    @staticmethod
    def _vflip(x):
        return functional.vflip(x)

    @staticmethod
    def _transpose(x):
        return torch.transpose(x, -2, -1)

    def transform(self, x: torch.Tensor):
        x = self.Ts[self.current_T](x)
        return x

    def inverse_transform(self, x: torch.Tensor):
        x = self.ITs[self.current_T](x)
        self.step()
        return x

    def step(self):
        self.current_T += 1
        if self.current_T >= len(self.Ts):
            self.current_T = 0
