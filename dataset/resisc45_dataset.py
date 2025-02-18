import os
from PIL import Image
import h5py
import numpy as np
import scipy.io as sio
from utils import SVD
from utils.norm import MinMax
from .base_dataset import BaseSVDLoader, BaseRGBLoader, BaseHSILoader


class RGBLoader(BaseRGBLoader):

    def load(self, root_path, name):
        rgb_dir = os.path.join(root_path, 'rgb')
        rgb = self._load_img(os.path.join(rgb_dir, name + '.jpg'))
        return rgb

    def _load_img(self, fn):
        img = Image.open(fn)
        rgb = np.array(img).astype(np.float32) / 255
        return rgb


def HSILoader():
    return None

def SVDLoader():
    return None
