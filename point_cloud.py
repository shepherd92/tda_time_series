#!/usr/bin/env python3
"""Point cloud module."""

from datetime import date

import numpy as np
from numpy.lib.stride_tricks import as_strided
import numpy.typing as npt
import pandas as pd

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

    def calculate_standard_deviations(self) -> pd.Series:
        """Calculate the standard deviations within each time window."""
        standard_deviations_data = self._data.std(axis=(1, 2))
        standard_deviations_series = pd.Series(
            standard_deviations_data,
            index=self.dates,
            name='standard_deviations',
        )
        standard_deviations_series.index.name = 'Date'
        return standard_deviations_series

    def calculate_mean_difference_of_windows(self) -> pd.Series:
        """Calculate the mean of the difference of consequtive sliding windows."""
        mean_difference_data = (self._data[1:] - self._data[:-1]).mean(axis=(1, 2))
        mean_difference_series = pd.Series(
            mean_difference_data,
            index=self.dates[1:],
            name='mean_difference_fo_time_windows',
        )
        mean_difference_series.index.name = 'Date'
        return mean_difference_series

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """Return the entire point clouds."""
        return self._data

    @property
    def window_size(self) -> int:
        """Return the window size."""
        return self._data.shape[1]

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding dimension."""
        return self._data.shape[2]

    @property
    def dates(self) -> list[date]:
        """Return the entire point clouds."""
        return self._dates
