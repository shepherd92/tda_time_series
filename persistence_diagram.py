#!/usr/bin/env python3
"""Simplicial complex module. Handles a series of simplicial complexes."""

import gudhi
import matplotlib.pyplot as plt


class PersistenceDiagram:
    """Persistence diagram creted from a simplicial complex."""

    def __init__(self, simplicial_complex: gudhi.SimplexTree) -> None:
        """Construct a PersistenceDiagram object."""
        self._raw_persistence: list[tuple[int, tuple[float, float]]] = simplicial_complex.persistence()
        self._betti_numbers: list[int] = simplicial_complex.betti_numbers()

    def betti_number(self, dimension: int) -> int:
        """Return the Betti number for the given dimension."""
        return self._betti_numbers[dimension] if dimension < len(self._betti_numbers) else 0

    def plot(self, axes: plt.Axes) -> None:
        """Plot persistence diagram on the given axes."""
        gudhi.plot_persistence_diagram(self._raw_persistence, axes=axes)
