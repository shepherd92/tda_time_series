#!/usr/bin/env python3
"""Simplicial complex module. Handles a series of simplicial complexes."""

from datetime import date
from pathlib import Path

import gudhi
import gudhi.representations
import matplotlib.pyplot as plt


RESOLUTION = 1000


class PersistenceLandscape:
    """Persistence diagram creted from a simplicial complex."""

    def __init__(self, company: str, date: date, simplicial_complex: gudhi.SimplexTree) -> None:
        """Construct a PersistenceLandscape object."""
        self._company = company
        self._date = date

        landscape = gudhi.representations.Landscape(resolution=RESOLUTION)
        self._landscape = landscape.fit_transform([simplicial_complex.persistence_intervals_in_dimension(1)])

    def plot(self, axes: plt.Axes, num_of_landscapes_to_plot: int) -> None:
        """Plot persistence landscape on the given axes."""
        for index in range(num_of_landscapes_to_plot):
            axes.plot(self._landscape[0][index * RESOLUTION:(index + 1) * RESOLUTION])

        axes.set_title(f'Persistence landscape {self.company}, {self.date}')

    def save(self, directory: Path) -> None:
        """Save the values of the persistence diagram."""
        raise NotImplementedError

    def norm(self, norm_order: int) -> float:
        """Return the norm of the persistence landscape."""

    @property
    def company(self) -> str:
        """Return the company name."""
        return self._company

    @property
    def date(self) -> date:
        """Return the date of the point cloud."""
        return self._date
