import os
from PIL import Image
import numpy as np
import scipy.io as sio
from utils import SVD
from utils.norm import MinMax
from .base_dataset import BaseSVDLoader, BaseRGBLoader, BaseHSILoader


class HSILoader(BaseHSILoader):

    def load(self, root_path: str, name: str):
        path = os.path.join(root_path, 'hsi', name, name)
        png_files = sorted([_fn for _fn in os.listdir(path) if _fn.endswith('.png')])

        imgs = [self._load_grayscale_png_to_nparray(os.path.join(path, _pf)) for _pf in png_files]
        hsi = np.array(imgs).transpose((1, 2, 0))
        return hsi

    def _load_grayscale_png_to_nparray(self, png_file):
        img = Image.open(png_file)
        img_array = np.array(img).astype(np.float32)

        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]

        return img_array


class RGBLoader(BaseRGBLoader):

    def load(self, root_path, name):
        end = name.rfind('_ms')
        rgb_name = name[:end]
        fn = os.path.join(root_path, 'hsi', name, name, rgb_name + '_RGB.bmp')

        if not os.path.exists(fn):
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
        else:
            rgb = self._load_img(fn)
            return rgb

    def _load_img(self, fn):
        img = Image.open(fn)
        rgb = np.array(img).astype(np.float32) / 255
        return rgb

    def _get_pseudo_rgb_from_hsi(self, hsi):
        bands = [26, 13, 5]
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

