import os
from PIL import Image
import numpy as np
import scipy.io as sio
from utils import SVD
from utils.norm import MinMax
from .base_dataset import BaseSVDLoader, BaseRGBLoader, BaseHSILoader


class HSILoader(BaseHSILoader):

    def load(self, root_path: str, name: str):
        fn = os.path.join(root_path, 'hsi', name + '.mat')
        mat = sio.loadmat(fn)
        hsi = np.array(mat['data']).astype(np.float32)
        return hsi


class RGBLoader(BaseRGBLoader):

    def load(self, root_path, name):
        pseudo_rgb_dir = os.path.join(root_path, 'pseudo_rgb')
        if not os.path.exists(pseudo_rgb_dir):
            os.makedirs(pseudo_rgb_dir)

        file_name = os.path.join(pseudo_rgb_dir, name + '.jpg')
        if not os.path.exists(file_name):
            hsi = HSILoader().load(root_path, name)
            pseudo_rgb = self._get_pseudo_rgb_from_hsi(hsi)
            Image.fromarray(
                (pseudo_rgb * 255).astype('uint8')
            ).save(os.path.join(pseudo_rgb_dir, name + '.jpg'))
        else:
            pseudo_rgb = self._load_img(os.path.join(pseudo_rgb_dir, name + '.jpg'))

        return pseudo_rgb

    def _load_img(self, fn):
        img = Image.open(fn)
        rgb = np.array(img).astype(np.float32) / 255
        return rgb

    def _get_pseudo_rgb_from_hsi(self, hsi):
        bands = [64, 31, 1]
        pseudo_rgb = hsi[..., bands]
        pseudo_rgb = MinMax(pseudo_rgb).transform(pseudo_rgb)
        pseudo_rgb **= 0.5
        return pseudo_rgb


class SVDLoader(BaseSVDLoader):

    def load(self, root_path, name):
        path = os.path.join(root_path, 'svd')
        if not os.path.exists(path):
            os.makedirs(path)

        file_name = os.path.join(path, name + '_SVD.mat')
        if not os.path.exists(file_name):
            hsi = HSILoader().load(root_path, name)
            svd = SVD(hsi)
            u, s = svd.u, svd.s
            mat = {
                'description': 'u.shape=(bands, feats), s.shape=(feats, )',
                'u': u,
                's': s,
            }
            sio.savemat(file_name, mat)
        else:
            mat = sio.loadmat(file_name)
            u, s = mat['u'], mat['s']

        return u, s









