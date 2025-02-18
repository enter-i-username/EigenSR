import numpy as np
import torch


class StripeMask:

    def __init__(self, droprate, rotation, return_mask=False):
        self.droprate = droprate
        self.rotation = rotation
        self.return_mask = return_mask

    def _get_degree(self):
        if isinstance(self.rotation, int):
            return self.rotation
        elif isinstance(self.rotation, str) and self.rotation.lower() == 'random':
            return np.random.randint(0, 360)
        else:
            raise NotImplementedError

    def __call__(self, img):

        data_type = type(img)
        if data_type == np.ndarray:
            H, W, *B = img.shape
        elif data_type == torch.Tensor:
            *B, H, W = img.shape
        else:
            raise NotImplementedError

        theta = np.radians(self._get_degree())

        # X.shape = [H, W], Y.shape = [H, W]
        Y, X = np.meshgrid(list(_ for _ in range(W)), list(_ for _ in range(H)))
        points = Points(
            np.concatenate(
                [X.reshape((H * W, 1)),
                 Y.reshape((H * W, 1))],
                axis=1
            )
        )

        samples = W
        rnd_idx = np.array([_ for _ in range(samples)])[np.random.uniform(0, 1, samples) < self.droprate]

        yi = rnd_idx
        xi = rnd_idx * (H - 1) / (W - 1)
        ABCs = np.ones((len(rnd_idx), 3))
        ABCs[:, 0] = np.cos(theta)
        ABCs[:, 1] = np.sin(theta)
        ABCs[:, 2] = -(xi * np.cos(theta) + yi * np.sin(theta))
        lines = Lines(ABCs)

        bool_matrices = lines.do_cross_points(points).reshape((len(rnd_idx), H, W))
        mask = np.logical_not(np.any(bool_matrices, axis=0)).astype(np.float32)

        if data_type == np.ndarray:
            img = img * mask.reshape((H, W, *(1 for _ in B)))
        elif data_type == torch.Tensor:
            img = img * mask.reshape((*(1 for _ in B), H, W))

        if self.return_mask:
            return img, mask
        else:
            return img


class Lines:
    # A1x + B1y + C1 = 0,
    # A2x + B2y + C2 = 0,
    # ...
    def __init__(self, ABCs):
        # An ABC is like [A, B, C]
        # self.lines.shape = [n, ABC]
        self.lines = np.array(ABCs)

    def do_cross_points(self, points_obj):
        return PointsLinesDis()(points_obj, self).T < 0.5


class Points:
    # (x1, y1),
    # (x2, y2),
    # ...
    def __init__(self, xys):
        # An xy is like [x, y]
        # self.points.shape = [n, xy]
        self.points = np.array(xys)


class PointsLinesDis:

    def __call__(self, points_n, lines_m):
        points_nm = self._projection_points(points_n, lines_m)
        dis = self._chebyshev_distance(points_n, points_nm)
        return dis

    def _chebyshev_distance(self, points_n, points_nm):
        # potins_n.points.shape = [n, xy]
        # points_nm.shape = [n, m, xy]
        pn = points_n.points[:, np.newaxis, :]
        # shape = [n, m]
        return np.abs(pn - points_nm).max(axis=2)

    def _projection_points(self, points_n, lines_m):
        # potins_n.points.shape = [n, xy]
        # lines_m.lines.shape = [m, ABC]
        pn = points_n.points[:, np.newaxis, :]
        lm = lines_m.lines[np.newaxis, :, :]

        # (n, 1, 1)
        x, y = pn[..., 0:1], pn[..., 1:2]
        # (1, m, 1)
        A, B, C = lm[..., 0:1], lm[..., 1:2], lm[..., 2:3]

        common = (A * x + B * y + C) / (A ** 2 + B ** 2)
        x_prj = x - A * common
        y_prj = y - B * common
        # shape = (n, m, xy)
        return np.concatenate([x_prj, y_prj], axis=2)
