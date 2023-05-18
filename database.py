#!/usr/bin/env python3
"""Database module for storing and handling data."""

from logging import warning
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Database:
    """Database responsible for storing and handling data."""

    def __init__(self, location: Path) -> None:
        """Construct a database without loading the data."""
        self._data = pd.DataFrame()
        assert location.is_dir(), \
            f'Given data location {location} is not a directory.'
        self._location = location

    def load(self) -> None:
        """Load the database from the disk."""
        warning('The databse is already loaded, it will be overwritten.')
        columns: list[pd.Series] = []
        for data_file in self._location.glob('*'):
            current_file_contents = pd.read_csv(data_file, index_col='Date', parse_dates=['Date'])
            adjusted_close = current_file_contents['Adj Close']
            adjusted_close.rename(data_file.stem, inplace=True)
            columns.append(adjusted_close)

        self._data = pd.concat(columns, axis=1)

    def visualize(self) -> plt.Figure:
        """Visualize the data."""
        figure, (subfig_1, subfig_2, subfig_3) = plt.subplots(3, 1)
        self._visualize_data_frame(self.data, subfig_1)
        self._visualize_data_frame(self.returns, subfig_2)
        self._visualize_data_frame(self.log_returns, subfig_3)
        return figure

    def export(self, output_directory: Path) -> None:
        """Export the data."""
        self._data.to_csv(output_directory / 'raw_data.csv')
        self.returns.to_csv(output_directory / 'returns.csv')
        self.log_returns.to_csv(output_directory / 'log_returns.csv')

    @staticmethod
    def _visualize_data_frame(data_frame: pd.DataFrame, axes: plt.Axes) -> None:
        for column in data_frame.columns:
            axes.plot(data_frame.index, data_frame[column], label=column)
        axes.legend(loc='upper left')

    @property
    def data(self) -> pd.DataFrame:
        """Calculate the returns."""
        return self._data

    @property
    def returns(self) -> pd.DataFrame:
        """Calculate the returns."""
        return (self._data - self._data.shift(1)) / self._data.shift(1)

    @property
    def log_returns(self) -> pd.DataFrame:
        """Calculate the returns."""
        return np.log(self._data / self._data.shift(1))

    @property
    def loaded(self) -> bool:
        """Return if the data is loaded into the database."""
        return not self._data.empty
