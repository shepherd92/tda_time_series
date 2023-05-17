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

import gudhi
import matplotlib.pyplot as plt
from tqdm import tqdm

from configuration import Configuration
from database import Database
from persistence_diagram import PersistenceDiagram
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

    for column in database.data.columns:
        time_delay_embedding = TimeDelayEmbedding()
        time_delay_embedding.create(database.data[column], embedding_dimension=configuration.embedding_dimension)

        point_cloud = PointCloud()
        point_cloud.create(time_delay_embedding, window_size=configuration.window_size)

        simplicial_complex_set = SimplicialComplexSet(point_cloud.data)
        simplicial_complex_set.create()
        persistence_diagrams = simplicial_complex_set.calc_persistences()
        # write_persistence_diagrams(persistence_diagrams, configuration.directories.output / 'persistence_diagram_csvs')
        export_persistence_diagrams(persistence_diagrams, configuration.directories.output / column /
                                    'persistence_diagram_figures')


def write_persistence_diagrams(persistences, directory: Path) -> None:
    """Create the peristence diagrams and export them."""
    directory.mkdir(parents=True)

    for index, persistence_diagram in tqdm(enumerate(persistences)):
        gudhi.write_persistence_diagram(str(directory / f'persistence_diagram_{index}.csv'))


def export_persistence_diagrams(persistences: list[PersistenceDiagram], directory: Path) -> None:
    """Create the peristence diagrams and export them."""
    directory.mkdir(parents=True)

    for index, persistence_diagram in tqdm(enumerate(persistences)):
        if persistence_diagram.betti_number(1) > 1:
            figure, axes = plt.subplots(1, 1)
            persistence_diagram.plot(axes)
            figure.savefig(directory / f'persistence_diagram_{index}.png')


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
