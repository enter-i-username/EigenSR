import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
import utils
from torch.utils.data import Dataset


class Transform(Dataset):

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


class Sequence:

    def __init__(self, transform_lst):
        self.transform_lst = transform_lst

    def __call__(self, data_dict):
        for _T in self.transform_lst:
            data_dict = _T(data_dict)

        return data_dict


class FromDataDictKeys:

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data_dict):
        return {_k: _v for _k, _v in data_dict.items() if _k in self.keys}


class ReturnKeys:

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data_dict):
        return tuple(data_dict[_k] for _k in self.keys)


class NumpyToTorch:

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data_dict):
        return {_k: (self._to_tensor(_v) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}

    @staticmethod
    def _to_tensor(_d):
        if isinstance(_d, np.ndarray):
            return torch.from_numpy(_d).float()
        elif isinstance(_d, tuple):
            return tuple(torch.from_numpy(__d).float() for __d in _d)


class TorchToNumpy:

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data_dict):
        return {_k: (self._to_numpy(_v) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}

    @staticmethod
    def _to_numpy(_d):
        if isinstance(_d, torch.Tensor):
            return _d.detach().cpu().numpy()
        elif isinstance(_d, tuple):
            return tuple(__d.detach().cpu().numpy() for __d in _d)


class NormPair:

    def __init__(self, src_key, tgt_key, Norm):
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.Norm = Norm

    def __call__(self, data_dict):
        src = data_dict[self.src_key]
        tgt = data_dict[self.tgt_key]

        norm = self.Norm(src, torch.mean, torch.std, dims=[-2, -1])
        data_dict_ = utils.dict_shallow_copy(data_dict)
        data_dict_[self.src_key] = norm.transform(src)
        data_dict_[self.tgt_key] = norm.transform(tgt)
        return data_dict_


class TensorChannelFirst:

    def __init__(self, keys=('hsi', )):
        self.keys = keys

    def __call__(self, data_dict):
        return {_k: (_v.permute(2, 0, 1) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}


class TensorChannelLast:

    def __init__(self, keys=('hsi', )):
        self.keys = keys

    def __call__(self, data_dict):
        return {_k: (_v.permute(1, 2, 0) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}


class TensorCrop:

    def __init__(self, params, keys=('hsi', )):
        self.params = params
        self.keys = keys

    def __call__(self, data_dict):
        i, j, h, w = self.params
        return {_k: (F.crop(_v, i, j, h, w) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}


# class TensorRandomCrop:
#
#     def __init__(self, size, keys=('hsi', )):
#         self.size = size
#         self.random_crop = transforms.RandomCrop(size=size)
#         self.keys = keys
#
#     def __call__(self, data_dict):
#         i, j, h, w = self.random_crop.get_params(data_dict[self.keys[0]], self.size)
#         return {_k: (F.crop(_v, i, j, h, w) if _k in self.keys else _v)
#                 for _k, _v in data_dict.items()}
class TensorRandomCrop:

    def __init__(self, size, keys=('hsi',)):
        self.size = size
        self.random_crop = transforms.RandomCrop(size=size)
        self.keys = keys

    def __call__(self, data_dict):
        padded_dict = {_k: (self._pad_img_if_patch_is_larger(_v, self.size) if _k in self.keys else _v)
                       for _k, _v in data_dict.items()}

        i, j, h, w = self.random_crop.get_params(padded_dict[self.keys[0]], self.size)

        return {_k: (F.crop(_v, i, j, h, w) if _k in self.keys else _v)
                for _k, _v in padded_dict.items()}

    @staticmethod
    def _pad_img_if_patch_is_larger(_i, _patch_size):
        _c, _h, _w = _i.shape
        _ph, _pw = _patch_size

        if _h < _ph:
            _pad_h = _ph - _h
            _pad_t = _pad_h // 2
            _pad_b = _pad_h - _pad_t
        else:
            _pad_t = _pad_b = 0

        if _w < _pw:
            _pad_w = _pw - _w
            _pad_l = _pad_w // 2
            _pad_r = _pad_w - _pad_l
        else:
            _pad_l = _pad_r = 0

        _i = F.pad(_i, padding=[_pad_l, _pad_t, _pad_r, _pad_b], padding_mode='reflect')
        return _i


class TensorRandomFlip:

    def __init__(self, p, keys=('hsi', )):
        flips = [
            self._all_hflip,
            self._all_vflip,
        ]
        self.random_choice = transforms.RandomChoice(flips)
        self.random_apply = transforms.RandomApply([self.random_choice], p=p)
        self.keys = keys

    def __call__(self, data_dict):
        return self.random_apply(data_dict)

    def _all_hflip(self, data_dict):
        return {_k: (F.hflip(_v) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}

    def _all_vflip(self, data_dict):
        return {_k: (F.vflip(_v) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}


class TensorRandomRotate:

    def __init__(self, keys=('hsi', )):
        degrees = [
            self._all_rotate_0,
            self._all_rotate_90,
            self._all_rotate_180,
            self._all_rotate_270
        ]
        self.random_choice = transforms.RandomChoice(degrees)
        self.keys = keys

    def __call__(self, data_dict):
        return self.random_choice(data_dict)

    def _all_rotate(self, data_dict, angle):
        return {_k: (F.rotate(_v, angle, expand=True) if _k in self.keys else _v)
                for _k, _v in data_dict.items()}

    def _all_rotate_0(self, data_dict):
        return data_dict

    def _all_rotate_90(self, data_dict):
        return self._all_rotate(data_dict, 90)

    def _all_rotate_180(self, data_dict):
        return self._all_rotate(data_dict, 180)

    def _all_rotate_270(self, data_dict):
        return self._all_rotate(data_dict, 270)


class GetRandomDegHSI:

    def __init__(self,
                 degradation_set,
                 hsi_key='hsi',
                 deg_hsi_key='deg_hsi'):
        self.random_choice = transforms.RandomChoice(degradation_set)
        self.hsi_key = hsi_key
        self.deg_hsi_key = deg_hsi_key

    def __call__(self, data_dict):
        hsi = data_dict[self.hsi_key]
        hsi = self.random_choice(hsi)

        data_dict_ = utils.dict_shallow_copy(data_dict)
        data_dict_[self.deg_hsi_key] = hsi
        return data_dict_


class GetSeqDegHSI:

    def __init__(self,
                 degradation_lst,
                 hsi_key='hsi',
                 deg_hsi_key='deg_hsi'):
        self.degradation_lst = degradation_lst
        self.hsi_key = hsi_key
        self.deg_hsi_key = deg_hsi_key

    def __call__(self, data_dict):
        hsi = data_dict[self.hsi_key]

        for _D in self.degradation_lst:
            hsi = _D(hsi)

        data_dict_ = utils.dict_shallow_copy(data_dict)
        data_dict_[self.deg_hsi_key] = hsi
        return data_dict_


class GetRandomSVDEigenimage:

    def __init__(self,
                 thr,
                 hsi_key='hsi',
                 deg_hsi_key='deg_hsi',
                 svd_key='svd',
                 eigen_key='eigen',
                 deg_eigen_key='deg_eigen'):
        assert 0. <= thr <= 1.
        self.thr = thr
        self.hsi_key = hsi_key
        self.deg_hsi_key = deg_hsi_key
        self.svd_key = svd_key
        self.eigen_key = eigen_key
        self.deg_eigen_key = deg_eigen_key

    def __call__(self, data_dict):
        deg_hsi = data_dict[self.deg_hsi_key]
        hsi = data_dict[self.hsi_key]
        u, s = data_dict[self.svd_key]

        quantile = np.cumsum(s / s.sum())
        cutoff = self._find_cutoff(quantile, self.thr)
        assert cutoff is not None
        rnd = self._random_choice(cutoff)

        _b, _f = u.shape
        _b, _h, _w = hsi.shape
        _b, _dh, _dw = deg_hsi.shape

        # eigen = (u.T @ hsi.reshape((_b, -1))).reshape((_f, _h, _w))
        # eigen = eigen[rnd: rnd + 1, ...]
        eigen = (u[:, rnd: rnd + 1].T @ hsi.reshape((_b, -1))).reshape((1, _h, _w))

        # deg_eigen = (u.T @ deg_hsi.reshape((_b, -1))).reshape((_f, _dh, _dw))
        # deg_eigen = deg_eigen[rnd: rnd + 1, ...]
        deg_eigen = (u[:, rnd: rnd + 1].T @ deg_hsi.reshape((_b, -1))).reshape((1, _dh, _dw))

        data_dict_ = utils.dict_shallow_copy(data_dict)
        data_dict_[self.eigen_key] = eigen
        data_dict_[self.deg_eigen_key] = deg_eigen
        return data_dict_

    @staticmethod
    def _find_cutoff(quantile, thr):
        for i in range(len(quantile) - 1, -1, -1):
            if quantile[i] <= thr:
                return i + 1
        return None

    @staticmethod
    def _random_choice(cutoff, weights=None):
        random_number = np.random.choice(np.arange(cutoff), p=weights)
        return random_number



class GetSVD:

    def __init__(self, hsi_key='hsi', svd_key='svd'):
        self.hsi_key = hsi_key
        self.svd_key = svd_key

    def __call__(self, data_dict):
        _hsi = data_dict[self.hsi_key]
        # _c, _h, _w = _hsi.shape
        _svd = utils.SVD(_hsi.permute(1, 2, 0), svd_solver=torch.linalg.svd)
        u, s = _svd.u, _svd.s

        data_dict_ = utils.dict_shallow_copy(data_dict)
        data_dict_[self.svd_key] = (u, s)
        return data_dict_



class GetSVDEigenimages:

    def __init__(self,
                 hsi_key='hsi',
                 deg_hsi_key='deg_hsi',
                 svd_key='svd',
                 eigen_key='eigen',
                 deg_eigen_key='deg_eigen'):
        self.hsi_key = hsi_key
        self.deg_hsi_key = deg_hsi_key
        self.svd_key = svd_key
        self.eigen_key = eigen_key
        self.deg_eigen_key = deg_eigen_key

    def __call__(self, data_dict):
        deg_hsi = data_dict[self.deg_hsi_key]
        hsi = data_dict[self.hsi_key]
        u, s = data_dict[self.svd_key]

        _b, _f = u.shape
        _b, _h, _w = hsi.shape
        _b, _dh, _dw = deg_hsi.shape

        eigen = (u.T @ hsi.reshape((_b, -1))).reshape((_f, _h, _w))
        # eigen = eigen[rnd: rnd + 1, ...]

        deg_eigen = (u.T @ deg_hsi.reshape((_b, -1))).reshape((_f, _dh, _dw))
        # deg_eigen = deg_eigen[rnd: rnd + 1, ...]

        data_dict_ = utils.dict_shallow_copy(data_dict)
        data_dict_[self.eigen_key] = eigen
        data_dict_[self.deg_eigen_key] = deg_eigen
        return data_dict_


class GetOneRandomChannel:

    def __init__(self, cutoff=None, keys=('hsi',)):
        self.cutoff = cutoff
        self.keys = keys

    def __call__(self, data_dict):
        if self.cutoff is None:
            _co = data_dict[self.keys[0]].shape[0]
        else:
            _co = self.cutoff
        _idx = self._random_choice(_co)

        return {_k: (_v[_idx: _idx + 1, ...] if _k in self.keys else _v)
                for _k, _v in data_dict.items()}

    @staticmethod
    def _random_choice(cutoff, weights=None):
        random_number = np.random.choice(np.arange(cutoff), p=weights)
        return random_number


class GetOneChannel:

    def __init__(self, idx, keys=('hsi',)):
        self.idx = idx
        self.keys = keys

    def __call__(self, data_dict):
        return {_k: (_v[self.idx: self.idx + 1, ...] if _k in self.keys else _v)
                for _k, _v in data_dict.items()}
