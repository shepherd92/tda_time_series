#!/usr/bin/env python3
"""Simplicial complex module. Handles a series of simplicial complexes."""

from datetime import date

import gudhi
import gudhi.representations
import numpy as np
import numpy.typing as npt


RESOLUTION = 1000
NUM_OF_LANDSCAPES = 4
DIMENSION = 1


class PersistenceLandscape:
    """Persistence diagram creted from a simplicial complex."""

    def __init__(self, company: str, date: date, simplicial_complex: gudhi.SimplexTree) -> None:
        """Construct a PersistenceLandscape object."""
        self._company = company
        self._date = date

        landscape: npt.NDArray[np.float_] = gudhi.representations.Landscape(
            num_landscapes=NUM_OF_LANDSCAPES,
            resolution=RESOLUTION
        )
        landscape = landscape.fit_transform(
            [simplicial_complex.persistence_intervals_in_dimension(DIMENSION)]
        )
        self._landscape = landscape.reshape((-1, RESOLUTION))

    def norm(self, norm_order: int) -> float:
        """Return the norm of the persistence landscape."""
        return (self.landscape_data**norm_order).sum()**(1. / norm_order)

    @property
    def company(self) -> str:
        """Return the company name."""
        return self._company

    @property
    def date(self) -> date:
        """Return the date of the point cloud."""
        return self._date

    @property
    def landscape_data(self) -> npt.NDArray[np.float_]:
        """Return the landscape data."""
        return self._landscape
