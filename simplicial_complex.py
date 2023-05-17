#!/usr/bin/env python3
"""Simplicial complex module. Handles a series of simplicial complexes."""

import gudhi
import numpy as np
import numpy.typing as npt

from persistence_diagram import PersistenceDiagram


class SimplicialComplexSet:
    """Simplicial complexes created from the time series."""

    MAX_DIMENSION = 8

    def __init__(self, point_cloud_data: npt.NDArray[np.float_]) -> None:
        """Construct a SimplicialComplex object."""
        self._point_clouds: npt.NDArray[np.float_] = point_cloud_data
        self._simplex_trees: list[gudhi.SimplexTree] = []

    def create(self) -> None:
        """Create point clouds from database."""
        for time_data in self._point_clouds:
            rips_complex = gudhi.RipsComplex(points=time_data, max_edge_length=100.)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=SimplicialComplexSet.MAX_DIMENSION)
            simplex_tree.compute_persistence()
            self._simplex_trees.append(simplex_tree)

    def simplex_trees(self, time_index: int) -> gudhi.SimplexTree:
        """Return the entire point clouds."""
        return self._simplex_trees[time_index]

    def calc_persistences(self) -> list[PersistenceDiagram]:
        """Create the peristence diagrams and export them."""
        persistences = [
            PersistenceDiagram(simplex_tree)
            for simplex_tree in self._simplex_trees
        ]

        return persistences
