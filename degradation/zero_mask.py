import torch
import numpy as np


class ZeroMask:

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, img):

        if type(img) == np.ndarray:
            mask = np.random.uniform(0, 1, img.shape) < self.rate
            mask = mask.astype(np.float32)
            img = img * mask

        elif type(img) == torch.Tensor:
            mask = torch.rand_like(img) < self.rate
            mask = mask.to(torch.float32)
            img = img * mask
        else:
            raise NotImplementedError

        return img

