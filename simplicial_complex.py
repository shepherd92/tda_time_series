#!/usr/bin/env python3
"""Simplicial complex module. Handles a series of simplicial complexes."""

from datetime import date

import gudhi

from persistence_diagram import PersistenceDiagram
from persistence_landscape import PersistenceLandscape
from point_cloud import PointCloud


class SimplicialComplexSet:
    """Simplicial complexes created from the time series."""

    MAX_DIMENSION = 8

    def __init__(self, company: str) -> None:
        """Construct a SimplicialComplex object."""
        self._name: str = company
        self._simplex_trees: list[gudhi.SimplexTree] = []
        self._embedding_dimension: int = 0
        self._window_size: int = 0
        self._dates: list[date] = []

    def create(self, point_cloud: PointCloud) -> None:
        """Create point clouds from database."""
        self._dates = point_cloud.dates
        self._embedding_dimension = point_cloud.embedding_dimension
        self._window_size = point_cloud.window_size

        for time_data in point_cloud.data:
            rips_complex = gudhi.RipsComplex(points=time_data, max_edge_length=100.)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=SimplicialComplexSet.MAX_DIMENSION)
            simplex_tree.compute_persistence()
            self._simplex_trees.append(simplex_tree)

    def simplex_trees(self, time_index: int) -> gudhi.SimplexTree:
        """Return the entire point clouds."""
        return self._simplex_trees[time_index]

    def calc_persistence_diagrams(self) -> list[PersistenceDiagram]:
        """Create the peristence diagrams."""
        persistence_diagrams: list[PersistenceDiagram] = []
        for date_, simplex_tree in zip(self.dates, self._simplex_trees):
            persistence_diagram = PersistenceDiagram(self.name, date_, simplex_tree)
            persistence_diagrams.append(persistence_diagram)

        return persistence_diagrams

    def calc_persistence_landscapes(self) -> list[PersistenceLandscape]:
        """Create the peristence landscapes."""
        persistence_landscapes: list[PersistenceLandscape] = []
        for date_, simplex_tree in zip(self.dates, self._simplex_trees):
            persistence_landscape = PersistenceLandscape(self.name, date_, simplex_tree)
            persistence_landscapes.append(persistence_landscape)

        return persistence_landscapes

    @property
    def name(self) -> str:
        """Return the name member."""
        return self._name

    @property
    def dates(self) -> list[date]:
        """Return the dates member."""
        return self._dates

    @property
    def window_size(self) -> int:
        """Return the window_size member."""
        return self._window_size

    @property
    def embedding_dimension(self) -> int:
        """Return the embedding_dimension member."""
        return self._embedding_dimension
