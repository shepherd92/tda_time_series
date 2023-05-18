#!/usr/bin/env python3
"""Main module."""

from argparse import ArgumentParser, Namespace
import cProfile
import io
import logging
from logging import basicConfig
from pathlib import Path
from pstats import Stats, SortKey
from subprocess import Popen, PIPE, call, check_output

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from configuration import Configuration
from database import Database
from persistence_diagram import PersistenceDiagram
from persistence_landscape import PersistenceLandscape
from point_cloud import PointCloud
from simplicial_complex import SimplicialComplexSet
from time_delay_embedding import TimeDelayEmbedding
from tools.debugger import debugger_is_active


def main(configuration: Configuration) -> None:
    """Run program - main function."""
    num_of_processes = 1 \
        if debugger_is_active() or configuration.profiling \
        else configuration.num_of_processes

    SimplicialComplexSet.MAX_DIMENSION = configuration.max_dimension

    database = Database(configuration.directories.data)
    database.load()
    database.export(configuration.directories.output)
    figure = database.visualize()
    figure.savefig(configuration.directories.output / 'data_series.png')

    pipeline(database.data, 'raw', configuration)
    pipeline(database.log_returns, 'log_returns', configuration)


def pipeline(data: pd.DataFrame, output_directory_prefix: str, configuration: Configuration) -> None:
    """Run the entire pipeline for a time series."""
    for company_ticker in data.columns:
        time_delay_embedding = create_time_delay_embedding(data[company_ticker], configuration.embedding_dimension)
        point_cloud = create_point_cloud(time_delay_embedding, configuration.window_size)
        std_plot = calculate_standard_deviations(point_cloud)
        std_plot.savefig(
            configuration.directories.output / f'{output_directory_prefix}_standard_deviations_{company_ticker}.png'
        )
        simplicial_complex_set = create_simplex_trees(point_cloud, company_ticker)

        mean_difference_of_windows_dates, mean_difference_of_windows = \
            point_cloud.calculate_mean_difference_of_windows()

        # persistence_diagrams = simplicial_complex_set.calc_persistence_diagrams()
        # export_persistence_diagrams(
        #     persistence_diagrams,
        #     configuration.directories.output / f'{output_directory_prefix}_persistence_diagrams' / company_ticker
        # )
        persistence_landscapes = simplicial_complex_set.calc_persistence_landscapes()
        export_persistence_landscapes(
            persistence_landscapes,
            configuration.directories.output / f'{output_directory_prefix}_persistence_landscapes' / company_ticker
        )


def create_time_delay_embedding(data: pd.Series, embedding_dimension: int) -> TimeDelayEmbedding:
    """Create a time delay embedding from a time series."""
    time_delay_embedding = TimeDelayEmbedding()
    time_delay_embedding.create(data, embedding_dimension=embedding_dimension)
    return time_delay_embedding


def create_point_cloud(time_delay_embedding: TimeDelayEmbedding, window_size: int) -> PointCloud:
    """Create a point cloud from a time delay embedded time series."""
    point_cloud = PointCloud()
    point_cloud.create(time_delay_embedding, window_size=window_size)
    return point_cloud


def create_simplex_trees(point_cloud: PointCloud, name: str) -> SimplicialComplexSet:
    """Create simplex trees from a series of point clouds."""
    simplicial_complex_set = SimplicialComplexSet(name)
    simplicial_complex_set.create(point_cloud)
    return simplicial_complex_set


def calculate_standard_deviations(point_cloud: PointCloud) -> tuple[plt.Figure]:
    """Create standard deviations plot."""
    figure, axes = plt.subplots(1, 1)
    dates, standard_deviations = point_cloud.calculate_standard_deviations()
    axes.plot(dates, standard_deviations)
    return figure

    mean_difference_of_windows_dates, mean_difference_of_windows = \
        point_cloud.calculate_mean_difference_of_windows()


def export_persistence_diagrams(persistence_diagrams: list[PersistenceDiagram], directory: Path) -> None:
    """Create the peristence diagrams and export them."""
    directory.mkdir(parents=True)

    for index, persistence_diagram in tqdm(enumerate(persistence_diagrams)):
        figure, axes = plt.subplots(1, 1)
        persistence_diagram.plot(axes)
        figure.savefig(directory / f'persistence_diagram_{index}.png')


def export_persistence_landscapes(persistence_landscapes: list[PersistenceLandscape], directory: Path) -> None:
    """Create the peristence diagrams and export them."""
    directory.mkdir(parents=True)

    for index, persistence_landscape in tqdm(enumerate(persistence_landscapes)):
        figure, axes = plt.subplots(1, 1)
        persistence_landscape.plot(axes, 3)
        figure.savefig(directory / f'persistence_landscape_{index}.png')


def create_parser() -> ArgumentParser:
    """Create parser for command line arguments."""
    parser = ArgumentParser(description='Load and prepare input data for training.')
    parser.add_argument('--config', type=Path, default=Path('config.json'), help='Path to the config file')
    return parser


def main_wrapper(params: Namespace) -> None:
    """Wrap the main function to separate profiling, logging, etc. from actual algorithm."""
    configuration = Configuration()
    configuration.load(params.config)
    configuration.directories.output.mkdir(parents=True, exist_ok=True)

    basicConfig(
        filename=configuration.directories.output / 'log.txt',
        filemode='w',
        encoding='utf-8',
        level=getattr(logging, configuration.log_level),
        format='%(asctime)s %(levelname)-8s %(filename)s.%(funcName)s%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
    logging.captureWarnings(True)

    if configuration.profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    main(configuration)

    if configuration.profiling:
        profile_output_dir = configuration.directories.output
        profiler.disable()
        stream = io.StringIO()
        statistics = Stats(profiler, stream=stream).sort_stats(SortKey.CUMULATIVE)
        statistics.print_stats(str(configuration.general.directories.root), 100)
        statistics.dump_stats(profile_output_dir / 'profile_results.prof')
        profile_log_path = profile_output_dir / 'profile_results.txt'

        with open(profile_log_path, 'w', encoding='utf-8') as profile_log_file:
            profile_log_file.write(stream.getvalue())

        with Popen(
            ('gprof2dot', '-f', 'pstats', profile_output_dir / 'profile_results.prof'),
            stdout=PIPE
        ) as gprof_process:
            check_output(
                ('dot', '-Tpng', '-o', profile_output_dir / 'profile_results.png'),
                stdin=gprof_process.stdout
            )
            gprof_process.wait()

        call(('snakeviz', '--server', profile_output_dir / 'profile_results.prof'))


if __name__ == '__main__':
    argument_parser = create_parser()
    program_arguments = argument_parser.parse_args()
    main_wrapper(program_arguments)
