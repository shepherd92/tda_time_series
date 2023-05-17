#!/usr/bin/env python3
"""Point cloud module."""

from datetime import date

import numpy as np
from numpy.lib.stride_tricks import as_strided
import numpy.typing as npt

from time_delay_embedding import TimeDelayEmbedding


class PointCloud:
    """Point cloud created from the time series."""

    def __init__(self) -> None:
        """Construct a database without loading the data."""
        self._data: npt.NDArray[np.float_] = np.empty((0, 0, 0))
        self._dates: list[date] = []

    def create(self, time_delay_embedding: TimeDelayEmbedding, window_size: int) -> None:
        """Given a 2D array of shape [T, E], create an array of shape [T, W, E].

        - T: number of times
        - E: dimension of the embedding
        - W: window size
        """
        dimension_0_of_result = time_delay_embedding.data.shape[0] - window_size + 1

        embedding_size = time_delay_embedding.data.shape[1]
        s0, s1 = time_delay_embedding.data.strides

        self._data = as_strided(
            time_delay_embedding.data, shape=(dimension_0_of_result, window_size, embedding_size),
            strides=(s0, s0, s1)
        )
        self._dates = time_delay_embedding.dates[-len(self._data):]

    def calculate_standard_deviations(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Calculate the standard deviations within each time window."""
        return self._dates, self._data.std(axis=(1, 2))

    def calculate_mean_difference_of_windows(self) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """Calculate the mean of the difference of consequtive sliding windows."""
        return self._dates[1:], (self._data[1:] - self._data[:-1]).mean(axis=(1, 2))

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """Return the entire point clouds."""
        return self._data

    @property
    def dates(self) -> list[date]:
        """Return the entire point clouds."""
        return self._dates
