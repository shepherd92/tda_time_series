#!/usr/bin/env python3
"""Main module."""

from argparse import ArgumentParser, Namespace
import cProfile
from datetime import date, datetime
import io
import logging
from logging import basicConfig
from pathlib import Path
from pstats import Stats, SortKey
from subprocess import Popen, PIPE, call, check_output

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import ttest_ind, shapiro, kstest, norm
from tqdm import tqdm

from configuration import Configuration
from database import Database
from persistence_diagram import PersistenceDiagram
from persistence_landscape import PersistenceLandscape
from point_cloud import PointCloud
from simplicial_complex import SimplicialComplexSet
from time_delay_embedding import TimeDelayEmbedding


def main(configuration: Configuration) -> None:
    """Run program - main function."""
    SimplicialComplexSet.MAX_DIMENSION = configuration.max_dimension

    database = Database(configuration.directories.data / 'prices')
    database.load()
    database.export(configuration.directories.output)
    database_figure = database.visualize()
    database_figure.savefig(configuration.directories.output / 'data_series.png')
    plt.close(database_figure)

    pipeline(database.data,        'raw',         configuration)
    pipeline(database.log_returns, 'log_returns', configuration)


def pipeline(
    data: pd.DataFrame,
    output_directory_prefix: str,
    configuration: Configuration
) -> None:
    """Run the entire pipeline for a time series."""
    for company_ticker in data.columns:
        time_delay_embedding = create_time_delay_embedding(data[company_ticker], configuration.embedding_dimension)
        point_cloud = create_point_cloud(time_delay_embedding, configuration.window_size)

        # standard deviations
        standard_deviations = point_cloud.calculate_standard_deviations()
        standard_deviations.to_csv(
            configuration.directories.output / f'{output_directory_prefix}_standard_deviations_{company_ticker}.csv'
        )
        std_plot = plot_time_series(standard_deviations)
        std_plot.savefig(
            configuration.directories.output / f'{output_directory_prefix}_standard_deviations_{company_ticker}.png'
        )

        # mean difference of windows
        mean_difference_of_windows = point_cloud.calculate_mean_difference_of_windows()
        mean_difference_of_windows.to_csv(
            configuration.directories.output / f'{output_directory_prefix}_mean_difference_{company_ticker}.csv'
        )
        mean_difference_of_windows_plot = plot_time_series(mean_difference_of_windows)
        mean_difference_of_windows_plot.savefig(
            configuration.directories.output / f'{output_directory_prefix}_mean_difference_{company_ticker}.png'
        )

        # simplicial complexes
        simplicial_complex_set = create_simplex_trees(point_cloud, company_ticker)

        # persistence diagrams
        persistence_diagrams = simplicial_complex_set.calc_persistence_diagrams()
        save_directory = configuration.directories.output / \
            f'{output_directory_prefix}_persistence_diagrams' / company_ticker
        export_persistence_diagrams(persistence_diagrams, save_directory)
        create_video(save_directory)

        # persistence landscapes
        persistence_landscapes = simplicial_complex_set.calc_persistence_landscapes()
        save_directory = configuration.directories.output / \
            f'{output_directory_prefix}_persistence_landscapes' / company_ticker
        export_persistence_landscapes(persistence_landscapes, save_directory)
        create_video(save_directory)

        # landscape norms
        landscape_norms_1 = calculate_landscape_norms(persistence_landscapes, 1)
        landscape_norms_1.to_csv(
            configuration.directories.output / f'{output_directory_prefix}_l1_norms_{company_ticker}.csv'
        )
        landscape_norms_1_plot = plot_time_series(landscape_norms_1)
        landscape_norms_1_plot.savefig(
            configuration.directories.output / f'{output_directory_prefix}_l1_norms_{company_ticker}.png'
        )
        landscape_norms_2 = calculate_landscape_norms(persistence_landscapes, 2)
        landscape_norms_2.to_csv(
            configuration.directories.output / f'{output_directory_prefix}_l2_norms_{company_ticker}.csv'
        )
        landscape_norms_2_plot = plot_time_series(landscape_norms_2)
        landscape_norms_2_plot.savefig(
            configuration.directories.output / f'{output_directory_prefix}_l2_norms_{company_ticker}.png'
        )

        # statistical tests
        bear_market_landscape_norms, bull_market_landscape_norms = separate_landscape_norms(
            landscape_norms_1,
            (datetime.strptime('2007-12-01', '%Y-%m-%d').date(), datetime.strptime('2009-06-30', '%Y-%m-%d').date()),
            (datetime.strptime('2017-12-01', '%Y-%m-%d').date(), datetime.strptime('2019-06-30', '%Y-%m-%d').date()),
        )
        bear_market_landscape_norms.to_csv(
            configuration.directories.output / f'{output_directory_prefix}_l1_norms_{company_ticker}_bear.csv'
        )
        bull_market_landscape_norms.to_csv(
            configuration.directories.output / f'{output_directory_prefix}_l1_norms_{company_ticker}_bull.csv'
        )
        _, shapiro_p_value = shapiro(landscape_norms_1)
        _, kstest_p_value = kstest(landscape_norms_1, norm.cdf)
        _, landscape_norm_t_p_value = t_test_landscape_norms(
            bear_market_landscape_norms, bull_market_landscape_norms
        )
        _, landscape_norm_bootstrap_p_value = bootstrap_test_landscape_norms(
            bear_market_landscape_norms, bull_market_landscape_norms, 100000
        )
        print(
            f'{company_ticker}: ' +
            f'Shapiro: {shapiro_p_value}, ' +
            f'Kolmogorov-Smirnov: {kstest_p_value}, ' +
            f't-test: {landscape_norm_t_p_value}, ' +
            f'bootstrap test: {landscape_norm_bootstrap_p_value}, '
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


def calculate_landscape_norms(persistence_landscapes: list[PersistenceLandscape], order: int) -> pd.Series:
    """Calculate landscape norms."""
    dates = [landscape.date for landscape in persistence_landscapes]
    landscape_norms = [landscape.norm(order) for landscape in persistence_landscapes]
    series = pd.Series(landscape_norms, index=dates, name='landscape_norm')
    series.index.name = 'Date'
    return series


def plot_time_series(series: pd.Series) -> plt.Figure:
    """Plot time series."""
    figure, axes = plt.subplots(1, 1)
    axes.plot(series.index, series.values)
    return figure


def export_persistence_diagrams(persistence_diagrams: list[PersistenceDiagram], directory: Path) -> None:
    """Create the peristence diagrams and export them."""
    directory.mkdir(parents=True)

    for index, persistence_diagram in tqdm(
        enumerate(persistence_diagrams),
        total=len(persistence_diagrams),
        desc='Plotting persistence diagrams...'
    ):
        points: pd.DataFrame = persistence_diagram.points(1)
        points.to_csv(directory / f'persistence_diagram_{index:04d}.csv', index=False)

        # figure, axes = plt.subplots(1, 1)
        # persistence_diagram.plot(axes)
        # points = persistence_diagram.points(1)
        # figure.savefig(directory / f'persistence_diagram_{index:04d}.png')
        # plt.close(figure)


def export_persistence_landscapes(
    persistence_landscapes: list[PersistenceLandscape],
    directory: Path
) -> None:
    """Create the peristence diagrams and export them."""
    directory.mkdir(parents=True)

    # determine maximum value
    max_y_value = 0.
    for persistence_landscape in persistence_landscapes:
        max_y_value = max(max_y_value, persistence_landscape.landscape_data.values.max())

    for index, persistence_landscape in tqdm(
        enumerate(persistence_landscapes),
        total=len(persistence_landscapes),
        desc='Plotting persistence landscapes...'
    ):
        persistence_landscape.landscape_data.to_csv(
            directory / f'persistence_landscape_{index:04d}.csv'
        )

        # figure, axes = plt.subplots(1, 1)
        # for plot in persistence_landscape.landscape_data:
        #     axes.plot(plot)
        # axes.get_xaxis().set_visible(False)
        # axes.get_yaxis().set_visible(False)
        # axes.set_ylim([0., max_y_value])
        # axes.set_title(f'Persistence landscape {persistence_landscape.company}, {persistence_landscape.date}')
        # figure.savefig(directory / f'persistence_landscape_{index:04d}.png')
        # plt.close(figure)


def separate_landscape_norms(
    landscape_norms: pd.Series,
    bear_market_interval: tuple[date, date],
    bull_market_interval: tuple[date, date],
) -> tuple[pd.Series, pd.Series]:
    """Separate landscape norms based on two intervals."""
    bear_market_norms = landscape_norms.loc[
        (landscape_norms.index >= bear_market_interval[0]) &
        (landscape_norms.index <= bear_market_interval[1])
    ]

    bull_market_norms = landscape_norms.loc[
        (landscape_norms.index >= bull_market_interval[0]) &
        (landscape_norms.index <= bull_market_interval[1])
    ]

    return bear_market_norms, bull_market_norms


def t_test_landscape_norms(bear_market_norms: pd.Series, bull_market_norms: pd.Series,) -> tuple[float, float]:
    """Perform a statistical test for the landscape norms in two different market conditions."""
    t_statistic, p_value = ttest_ind(bear_market_norms, bull_market_norms, equal_var=False)
    return t_statistic, p_value


def bootstrap_test_landscape_norms(
    bear_market_norms: pd.Series,
    bull_market_norms: pd.Series,
    bootstrap_samples: int
) -> tuple[float, float]:
    """Perform bootstrap test for means."""
    bear_values = bear_market_norms.values
    bull_values = bull_market_norms.values

    statistics: list[float] = []
    for _ in range(bootstrap_samples):
        sample: npt.NDArray[np.float_] = np.random.choice(bear_values, len(bear_values), replace=True)
        statistics.append(sample.mean())

    quantile = (np.array(statistics) < bull_values.mean()).mean()
    p_value = quantile if quantile < 0.5 else 1 - quantile

    return p_value


def create_video(image_directory: Path):
    """Create video from a set of images."""
    video_name = f'{image_directory.stem}.avi'
    images = sorted(list(image_directory.glob('*.png')))
    frame = cv2.imread(str(images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(str(image_directory / '..' / video_name), 0, 15, (width, height))
    cv2.VideoWriter()

    for image in images:
        video.write(cv2.imread(str(image)))

    cv2.destroyAllWindows()
    video.release()


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
