import numpy as np


class SVD:

    def __init__(self, hsi, svd_solver=np.linalg.svd):
        h, w, c = hsi.shape
        hsi_2d = hsi.reshape((h * w, c))
        self.u, self.s, self.v = svd_solver(hsi_2d.T, full_matrices=False)

    def transform(self, hsi, dims=None):
        h, w, c = hsi.shape
        hsi_2d = hsi.reshape((h * w, c))

        if dims is None:
            dims = self.u.shape[1]

        svdeigen_2d = hsi_2d @ self.u[:, :dims]
        svdeigen = svdeigen_2d.reshape((h, w, dims))
        return svdeigen

    def inverse_transform(self, svdeigen):
        h, w, dims = svdeigen.shape
        c = self.u.shape[0]
        svdeigen_2d = svdeigen.reshape((h * w, dims))

        hsi_recon_2d = svdeigen_2d @ self.u[:, :dims].T
        hsi_recon = hsi_recon_2d.reshape((h, w, c))
        return hsi_recon

