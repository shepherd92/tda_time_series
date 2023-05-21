#!/usr/bin/env python3
"""Simplicial complex module. Handles a series of simplicial complexes."""

from datetime import date

import gudhi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PersistenceDiagram:
    """Persistence diagram creted from a simplicial complex."""

    def __init__(
        self,
        company: str,
        date: date,
        simplicial_complex: gudhi.SimplexTree
    ) -> None:
        """Construct a PersistenceDiagram object."""
        self._company = company
        self._date = date
        self._raw_persistence: list[tuple[int, tuple[float, float]]] = simplicial_complex.persistence()
        self._betti_numbers: list[int] = simplicial_complex.betti_numbers()

    def betti_number(self, dimension: int) -> int:
        """Return the Betti number for the given dimension."""
        return self._betti_numbers[dimension] if dimension < len(self._betti_numbers) else 0

    def plot(self, axes: plt.Axes) -> None:
        """Plot persistence diagram on the given axes."""
        gudhi.plot_persistence_diagram(self._raw_persistence, axes=axes)

        axes.set_title(f'Persistence diagram {self.company}, {self.date}')
        axes.set_aspect('equal')

    def points(self, dimension: int) -> pd.DataFrame:
        """Points of persistence diagram in a particular dimension."""
        points_array = np.array([
            dimension_point_pair[1]
            for dimension_point_pair in self._raw_persistence
            if dimension_point_pair[0] == dimension
        ])
        points_data = points_array if len(points_array) != 0 else np.empty((0, 2))
        points = pd.DataFrame(points_data, columns=['birth', 'death'])

        return points

    @property
    def company(self) -> str:
        """Return the company name."""
        return self._company

    @property
    def date(self) -> date:
        """Return the date of the point cloud."""
        return self._date
