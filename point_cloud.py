#!/usr/bin/env python3
"""Point cloud module."""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import numpy.typing as npt

from time_delay_embedding import TimeDelayEmbedding


class PointCloud:
    """Point cloud created from the time series."""

    def __init__(self) -> None:
        """Construct a database without loading the data."""
        self._data: npt.NDArray[np.float_] = np.empty((0, 0, 0))

    def create(self, time_delay_embedding: TimeDelayEmbedding, window_size: int) -> None:
        """Create point clouds from database."""
        self._data = PointCloud._calc_sliding_window_point_cloud(time_delay_embedding.data, window_size)

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """Return the entire point clouds."""
        return self._data

    @staticmethod
    def _calc_sliding_window_point_cloud(
        time_delay_embedding: npt.NDArray[np.float_],
        window_size: int
    ) -> npt.NDArray[np.float_]:
        """Given a 2D array of shape [T, E], return an array of shape [T, W, E].

        - T: number of times
        - E: dimension of the embedding
        - W: window size
        """
        dimension_0_of_result = time_delay_embedding.shape[0] - window_size + 1

        embedding_size = time_delay_embedding.shape[1]
        s0, s1 = time_delay_embedding.strides

        result = as_strided(
            time_delay_embedding, shape=(dimension_0_of_result, window_size, embedding_size),
            strides=(s0, s0, s1)
        )
        return result
