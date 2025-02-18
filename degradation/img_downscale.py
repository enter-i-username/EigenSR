import torch
import numpy as np
from utils.imresize import imresize


class ImgDownscale:

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, img):

        if type(img) == np.ndarray:
            _h, _w, _c = img.shape
            _new_h = int(_h * self.scale_factor)
            _new_w = int(_w * self.scale_factor)

            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = imresize(img, sizes=(_new_h, _new_w))
            img = img.squeeze(0).permute(1, 2, 0).numpy()

        elif type(img) == torch.Tensor:
            if len(img.shape) == 3:
                _c, _h, _w = img.shape
                _new_h = int(_h * self.scale_factor)
                _new_w = int(_w * self.scale_factor)

                img = img.unsqueeze(0)
                img = imresize(img, sizes=(_new_h, _new_w))
                img = img.squeeze(0)
            else:
                _b, _c, _h, _w = img.shape
                _new_h = int(_h * self.scale_factor)
                _new_w = int(_w * self.scale_factor)
                img = imresize(img, sizes=(_new_h, _new_w))

        else:
            raise NotImplementedError

        return img#, {'type': 'downscale', 'original size': (_h, _w)}

