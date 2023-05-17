#!/usr/bin/env python
"""Configuration class."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
from typing import NamedTuple


class Directories(NamedTuple):
    """Directory configuration."""

    data: Path = Path()
    output: Path = Path()


@dataclass
class Configuration:
    """Class handling configuration parameters."""

    log_level = 'WARNING'
    profiling = False
    num_of_processes = 1
    directories = Directories()

    def load(self, path: Path) -> None:
        """Construct a high level data node."""
        with open(path, encoding='utf-8') as config_file:
            params = json.load(config_file)

        now = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.log_level = str(params['log_level'])
        self.profiling = bool(params['profiling'])
        self.num_of_processes = Configuration._determine_num_of_processes(int(params['num_of_processes']))
        self.embedding_dimension = int(params['embedding_dimension'])
        self.window_size = int(params['window_size'])
        self.max_dimension = int(params['max_dimension'])

        self.directories = Directories(
            data=Path(params['directories']['data']),
            output=Path(params['directories']['output']) / now,
        )

    @staticmethod
    def _determine_num_of_processes(desired_num_of_processes: int | None) -> int:

        cpu_count = os.cpu_count()
        if not cpu_count:
            num_of_processes = 1
        elif desired_num_of_processes:
            num_of_processes = min(cpu_count, desired_num_of_processes)
            num_of_processes = max(num_of_processes, 1)
        else:
            num_of_processes = max(cpu_count - 2, 1)

        return num_of_processes
