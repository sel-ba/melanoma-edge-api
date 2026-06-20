from __future__ import annotations

import numpy as np


class MacenkoNormalizer:
    """Macenko stain normalization adapted for dermoscopy images."""

    def __init__(self, Io: int = 240, alpha: float = 1.0, beta: float = 0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.stain_matrix_target = None
        self.maxC_target = None

    def fit(self, target_image: np.ndarray) -> "MacenkoNormalizer":
        self.stain_matrix_target, self.maxC_target = self._get_stain_matrix(target_image)
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        if self.stain_matrix_target is None:
            raise RuntimeError("Normalizer not fitted. Call fit() first.")

        stain_matrix_source, _ = self._get_stain_matrix(image)
        source_concentrations = self._get_concentrations(image, stain_matrix_source)
        maxC_source = np.percentile(source_concentrations, 99, axis=0)
        source_concentrations *= self.maxC_target / (maxC_source + 1e-6)
        out = np.exp(-np.dot(source_concentrations, self.stain_matrix_target.T))
        out = np.clip(out * self.Io, 0, 255).astype(np.uint8)
        return out.reshape(image.shape)

    def _get_stain_matrix(self, image: np.ndarray):
        image = image.reshape(-1, 3).astype(np.float64)
        image = np.clip(image, 1, self.Io)
        od = -np.log(image / self.Io + 1e-6)
        od_hat = od[np.all(od > self.beta, axis=1)]

        if len(od_hat) == 0:
            return np.eye(2, 3), np.array([1.0, 1.0])

        _, _, v = np.linalg.svd(od_hat, full_matrices=False)
        v = v[:2, :]
        that = od_hat @ v.T
        phi = np.arctan2(that[:, 1], that[:, 0])
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100 - self.alpha)

        v1 = v.T @ np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = v.T @ np.array([np.cos(max_phi), np.sin(max_phi)])
        if v1[0] < 0:
            v1 = -v1
        if v2[0] < 0:
            v2 = -v2

        stain_matrix = np.array([v1, v2])
        concentrations = np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0].T
        max_c = np.percentile(concentrations, 99, axis=0)
        return stain_matrix, max_c

    def _get_concentrations(self, image: np.ndarray, stain_matrix: np.ndarray):
        od = -np.log(np.clip(image.reshape(-1, 3).astype(np.float64), 1, self.Io) / self.Io + 1e-6)
        return np.linalg.lstsq(stain_matrix.T, od.T, rcond=None)[0].T
