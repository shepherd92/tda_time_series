#!/usr/bin/env python3
"""Point cloud module."""

from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd


class TimeDelayEmbedding:
    """Point cloud created from the time series."""

    def __init__(self) -> None:
        """Construct a database without loading the data."""
        self._data: npt.NDArray[np.float_] = np.empty((0, 0))
        self._dates: list[date] = []

    def create(self, raw_data: pd.Series, embedding_dimension: int) -> None:
        """Create point clouds from database."""
        embedding = pd.concat([
            raw_data.shift(time_delay)
            for time_delay in range(embedding_dimension)
        ], axis=1)[embedding_dimension-1:]
        self._dates = [date_time.date() for date_time in embedding.index.to_list()]
        self._data = embedding.to_numpy()

    @property
    def data(self) -> npt.NDArray[np.float_]:
        """Return the entire point clouds."""
        return self._data

    @property
    def dates(self) -> list[date]:
        """Return the entire point clouds."""
        return self._dates
