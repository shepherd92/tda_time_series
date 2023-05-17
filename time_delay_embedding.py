#!/usr/bin/env python3
"""Point cloud module."""

import numpy as np
import numpy.typing as npt
import pandas as pd


class TimeDelayEmbedding:
    """Point cloud created from the time series."""

    def __init__(self) -> None:
        """Construct a database without loading the data."""
        self._data: npt.NDArray[np.float_] = np.empty((0, 0))

    def create(self, raw_data: pd.Series, embedding_dimension: int) -> None:
        """Create point clouds from database."""
        self._data = TimeDelayEmbedding._calc_time_delay_embedding(raw_data, embedding_dimension)

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """Return the entire point clouds."""
        return self._data

    @staticmethod
    def _calc_time_delay_embedding(data_series: pd.Series, dimension: int) -> npt.NDArray[np.float_]:
        """Return a numpy array with the time delay embedding of the data series."""
        embedding = pd.concat([
            data_series.shift(time_delay)
            for time_delay in range(dimension)
        ], axis=1)
        return embedding.to_numpy()
