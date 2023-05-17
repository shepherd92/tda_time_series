#!/usr/bin/env python3
"""Simplicial complex module. Handles a series of simplicial complexes."""

from datetime import date
from typing import NamedTuple
from pathlib import Path

import gudhi
import matplotlib.pyplot as plt


class PersistenceDiagram:
    """Persistence diagram creted from a simplicial complex."""

    class Properties(NamedTuple):
        """Describe the properties of the persistence diagram."""

        company: str
        window_size: int
        embedding_dimension: int
        date: date

    def __init__(self, persistence_diagram_properties: Properties, simplicial_complex: gudhi.SimplexTree) -> None:
        """Construct a PersistenceDiagram object."""
        self._properties: PersistenceDiagram.Properties = persistence_diagram_properties
        self._raw_persistence: list[tuple[int, tuple[float, float]]] = simplicial_complex.persistence()
        self._betti_numbers: list[int] = simplicial_complex.betti_numbers()

    def betti_number(self, dimension: int) -> int:
        """Return the Betti number for the given dimension."""
        return self._betti_numbers[dimension] if dimension < len(self._betti_numbers) else 0

    def plot(self, axes: plt.Axes) -> None:
        """Plot persistence diagram on the given axes."""
        gudhi.plot_persistence_diagram(self._raw_persistence, axes=axes)

        axes.set_title(f'Persistence diagram {self._properties.company}, {self._properties.date}')
        axes.set_aspect('equal')

    def save(self, directory: Path) -> None:
        """Save the values of the persistence diagram."""
        raise NotImplementedError
