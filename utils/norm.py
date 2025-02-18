import numpy as np
import torch


class Norm:

    def __init__(self, x, mean_func=np.mean, std_func=np.std, dims=None):
        self.mean = mean_func(x, dims, keepdims=True)
        self.std = std_func(x, dims, keepdims=True)

    def transform(self, x):
        x_norm = (x - self.mean) / self.std
        return x_norm

    def inverse_transform(self, x_norm):
        x = x_norm * self.std + self.mean
        return x


class MinMax:

    def __init__(self, x, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = x.min()
        self.max = x.max()

    def transform(self, x):
        x_std = (x - self.min) / (self.max - self.min)
        x_norm = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return x_norm

    def inverse_transform(self, x_norm):
        x_std = (x_norm - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        x = x_std * (self.max - self.min) + self.min
        return x


