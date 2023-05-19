#!/usr/bin/env python3
"""Market conditions module."""

from datetime import date
from datetime import datetime
from enum import auto, Enum
from pathlib import Path

import pandas as pd


class MarketConditionCalculator:
    """Calculate market condtions for specific dates."""

    class Condition(Enum):
        """Enum type to show market condtions."""

        BEAR: int = auto()
        BULL: int = auto()

    def __init__(self, recession_indicator_file_path: Path) -> None:
        """Construct MarketConditionCalculator object."""
        self._data = pd.read_csv(
            recession_indicator_file_path,
            index_col='DATE',
            parse_dates=['DATE'],
            dtype={'JHDUSRGDPBR': int},
        ).squeeze()

    def assign_market_condition(self, dates: list[date]) -> list[Condition]:
        """Get market condition for a given date."""
        conditions: list[MarketConditionCalculator.Condition] = []
        for date_ in dates:
            time = datetime.combine(date_, datetime.min.time())
            relevant_index_date = self._data.index.asof(time)
            is_recession = self._data.loc[relevant_index_date] == 1
            if is_recession:
                conditions.append(MarketConditionCalculator.Condition.BEAR)
            else:
                conditions.append(MarketConditionCalculator.Condition.BULL)

        return conditions
