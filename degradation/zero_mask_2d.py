import torch
import numpy as np


class ZeroMask2D:

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, img):

        if type(img) == np.ndarray:
            H, W, B = img.shape
            mask = np.random.uniform(0, 1, (H, W, 1)) < self.rate
            mask = mask.astype(np.float32)
            img = img * mask

        elif type(img) == torch.Tensor:
            *dims, H, W = img.shape
            mask = torch.rand(*[1 for _ in dims], H, W) < self.rate
            mask = mask.to(torch.float32).to(img.device)
            img = img * mask
        else:
            raise NotImplementedError

        return img

