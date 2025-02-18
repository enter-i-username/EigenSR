import numpy as np
import skimage.metrics as skm
from log.log import Log
import utils


class MetricLog:

    def __init__(self, **metrics):

        self.log = Log()
        self.history_one_epoch = {_metric_name: list() for _metric_name in metrics.keys()}
        self.metrics = metrics

        self.best_epoch = 0
        self.best_metric = None

    def __call__(self, data, target):
        b, c, h, w = data.shape

        for _b in range(b):
            data_numpy = data[_b, ...].detach().cpu().numpy().transpose((1, 2, 0))
            target_numpy = target[_b, ...].detach().cpu().numpy().transpose((1, 2, 0))

            minmax = utils.norm.MinMax(target_numpy)
            data_numpy = minmax.transform(data_numpy)
            target_numpy = minmax.transform(target_numpy)

            for _name, _metric in self.metrics.items():
                self.history_one_epoch[_name].append(
                    float(_metric(x=data_numpy, ref_x=target_numpy))
                )

    def write_save(self, epoch, fn, metric='mPSNR', optimal='max'):
        if optimal == 'max':
            _compare = np.greater
        elif optimal == 'min':
            _compare = np.less
        else:
            raise NotImplementedError

        metric = np.mean(self.history_one_epoch[metric])
        if self.best_metric is not None:
            if _compare(metric, self.best_metric):
                self.best_metric = metric
                self.best_epoch = epoch
        else:
            self.best_metric = metric
            self.best_epoch = epoch

        for k, v in self.history_one_epoch.items():
            self.log.write('mean ' + k, np.mean(v))
            self.log.write(k, v)
            self.history_one_epoch[k] = list()

        self.log.write('epoch', epoch)
        self.log.save(fn)

    def get_best_epoch(self):
        return self.best_epoch


def mSAM(x: np.ndarray, ref_x: np.ndarray):
    rows, cols, bands = x.shape
    r_rows, r_cols, r_bands = ref_x.shape
    assert rows == r_rows and cols == r_cols and bands == r_bands

    x_2d = x.reshape((rows * cols, bands))
    ref_x_2d = ref_x.reshape((rows * cols, bands))

    x_2d_norm = x_2d / np.linalg.norm(x_2d, axis=1, keepdims=True)
    ref_x_2d_norm = ref_x_2d / np.linalg.norm(ref_x_2d, axis=1, keepdims=True)

    dot_product = np.sum(x_2d_norm * ref_x_2d_norm, axis=1)
    dot_product = np.clip(dot_product, -1., 1.)
    mean_arccos_radians = np.mean(np.arccos(dot_product))
    mean_arccos_degrees = np.degrees(mean_arccos_radians)

    return mean_arccos_degrees


def mPSNR(x: np.ndarray, ref_x: np.ndarray):
    rows, cols, bands = x.shape
    r_rows, r_cols, r_bands = ref_x.shape
    assert rows == r_rows and cols == r_cols and bands == r_bands

    mean_lst = [skm.peak_signal_noise_ratio(ref_x[..., _b], x[..., _b], data_range=1.) for _b in range(bands)]

    mean_psnr = np.mean(mean_lst)

    return mean_psnr


def mSSIM(x: np.ndarray, ref_x: np.ndarray):
    rows, cols, bands = x.shape
    r_rows, r_cols, r_bands = ref_x.shape
    assert rows == r_rows and cols == r_cols and bands == r_bands

    mean_lst = [skm.structural_similarity(ref_x[..., _b], x[..., _b], data_range=1.) for _b in range(bands)]

    mean_ssim = np.mean(mean_lst)

    return mean_ssim


def RMSE(x: np.ndarray, ref_x: np.ndarray):
    # rows, cols, bands = x.shape
    # r_rows, r_cols, r_bands = ref_x.shape
    # assert rows == r_rows and cols == r_cols and bands == r_bands

    # total = rows * cols * bands
    mse = np.mean((x - ref_x) ** 2)
    rmse = mse ** 0.5

    return rmse


key2metric_dict = {
    'mPSNR': mPSNR,
    'mSSIM': mSSIM,
    'mSAM': mSAM,
    'RMSE': RMSE
}


def get_metric(key: str) -> callable:
    return key2metric_dict[key]
