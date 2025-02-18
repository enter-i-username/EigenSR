import numpy as np
import torch


class GaussianNoise:

    def __init__(self, **kwargs):
        sigma = kwargs.get('sigma', None)
        if sigma is not None:
            self.sigma = sigma
        else:
            self.snr = kwargs.get('snr')

    def __call__(self, data):

        if hasattr(self, 'sigma'):
            sigma = self.sigma
        elif hasattr(self, 'snr'):
            sigma = self._snr2std(self.snr, data)
        else:
            raise KeyError

        noise = None
        if type(data) == np.ndarray:
            noise = torch.randn(data.shape) * sigma
            noise = noise.numpy()
        elif type(data) == torch.Tensor:
            noise = torch.randn_like(data) * sigma

        data = data + noise
        return data

    @staticmethod
    def _snr2std(snr, data):
        signal_power = (data ** 2).mean()
        noise_power = signal_power / (10 ** (snr / 10))
        std_dev = noise_power ** 0.5
        return std_dev




