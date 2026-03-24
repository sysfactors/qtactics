# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple
import logging
from functools import reduce
from technical.indicators import cmf
import math


logger = logging.getLogger(__name__)

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib

# ==========================================
# Freqtrade - 100% Automated DCA Bot Strategy (Full Tutorial)
# https://youtu.be/lxlti_EDD28
# ==========================================

# ================================
# Freqtrade Version
# ================================

"""

freqtrade -V

Operating System:       Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.36
Python Version:         Python 3.13.8
CCXT Version:           4.5.20

Freqtrade Version:      freqtrade 2025.11
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_spot.json \
    --timerange 20200101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy DCA_RSI_30m_SO3 \
    --config user_data/binance_spot.json \
    --timeframe 30m \
    --timerange 20220101-20250101 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy \
    -e 100 \
    --job-workers 10 \
    --random-state 9319 \
    --max-open-trades 1 \
    -p SOL/USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --timerange 20250101-20260101 \
    --strategy DCA_RSI_30m_SO3 \
    -i 30m \
    --breakdown month \
    --config user_data/binance_spot.json \
    --strategy-path user_data/strategies \
    --cache none \
    --max-open-trades 1 \
    --timeframe-detail 5m \
    -p SOL/USDT
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_spot.json
"""

class DCA_RSI_30m_SO3(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "30m"
    
    # Can this strategy go short?
    can_short: bool = False
    
    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}
    
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.99

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured
    
    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # This enables the usage of adjust_trade_position() callback in the strategy.
    position_adjustment_enable = True 
    
    # Limit the number of additional entries per trade (on top of the first entry order) that the bot can execute.
    max_entry_position_adjustment = 3

    # Strategy parameters

    # This defines the exact USDT amount used for the initial entry.
    base_order_size = CategoricalParameter([50, 100], default=100, space='buy', optimize=False)
    
    # This determines the size of the first safety order, calculated as a multiple of the base order size.
    safety_order_ratio = DecimalParameter(1.8, 2.0, default=1.8, space='buy', decimals=1, optimize=True)
    
    # This sets a hard limit on how many safety orders can be placed within a single trade cycle.
    safety_order_max_count = IntParameter(3, 3, default=3, space='buy', optimize=False)
    
    # This is the percentage price drop required to trigger the first safety order.
    price_deviation_initial = DecimalParameter(0.02, 0.05, default=0.025, decimals=3, space='buy', optimize=True)
    
    # This multiplier increases the spacing between safety orders, controlling how aggressively the strategy averages down.
    safety_order_step_scale = DecimalParameter(1.6, 2.5, default=1.8, decimals=1, space='buy', optimize=True)
    
    # This is a multiplier that increases the size of each subsequent safety order, allowing the position size to grow as the price moves lower.
    safety_order_volume_scale = DecimalParameter(1.8, 2, default=1.7, decimals=1, space='buy', optimize=True)
    
    # This defines the profit target based on the combined position size across all orders.
    take_profit = DecimalParameter(0.025, 0.06, default=0.035, decimals=3, space='buy', optimize=True)
    
    # This specifies how the RSI signal is triggered, whether RSI is below a threshold, crossing below it, or crossing above it.
    rsi_condition = CategoricalParameter(['less_than', 'crossing_down', 'crossing_up'], default='less_than', optimize=False, space='buy')
    
    # This sets the lookback window used to calculate the RSI.
    rsi_period = CategoricalParameter([7, 14], default=14, space='buy', optimize=True)
    
    # This is the RSI value used to trigger trade entries.
    rsi_threshold = CategoricalParameter([25, 30, 35], default=30, space='buy', optimize=True)
    
    @property
    def plot_config(self):
        
        plot_config = {
            "main_plot": {},
            "subplots": {
                "RSI": {
                    "rsi": {
                        "color": "#9e57c2",
                        "type": "line"
                    }
                }
            }
        }
        
        return plot_config
    
    def custom_stake_amount(self, pair: str, current_time: 'datetime', current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """ 
        Forces the initial Base Order (BO) stake to the configured absolute value. 
        """
        
        return self.base_order_size.value

    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: Optional[float], max_stake: float,
                              **kwargs) -> float | None | Tuple[float | None, str | None]:
        """
        Safety Order (DCA) execution logic.

        so_level definition (based on successful entries):
        - so_level = 1 → Base Order already filled, next is Safety Order 1 (SO1)
        - so_level = 2 → Safety Order 1 filled, next is Safety Order 2 (SO2)
        - so_level = 3 → Safety Order 2 filled, next is Safety Order 3 (SO3)
        """

        # 1. Current DCA level (always >= 1 once base order is filled)
        so_level = trade.nr_of_successful_entries

        # 2. Stop placing orders if max safety order count is exceeded
        # safety_order_max_count refers to the number of SAFETY orders only (base order excluded)
        if so_level > self.safety_order_max_count.value:
            return None

        # 3. Prevent placing a new order if there is already an open order
        if trade.open_orders:
            return None

        # ------------------------------------------------------------------
        # 4. Calculate required price deviation from the BASE order price
        #
        # price_deviation_initial (d):
        #   Percentage drop required to trigger the FIRST safety order (SO1)
        #
        # safety_order_step_scale (s):
        #   Controls how much farther apart each subsequent safety order is placed
        #
        # Example:
        #   d = 1%, s = 2
        #
        #   Base Order →   0%
        #   SO1        →  -1%
        #   SO2        →  -3%
        #   SO3        →  -7%
        #
        # Formula creates either linear or geometric spacing
        # ------------------------------------------------------------------

        d = self.price_deviation_initial.value
        s = self.safety_order_step_scale.value

        if s == 1:
            # Linear spacing between safety order price levels
            required_deviation = -d * so_level
        else:
            # Geometric spacing between safety order price levels
            required_deviation = -d * ((s ** so_level - 1) / (s - 1))

        # 5. Calculate the actual trigger price based on the base order price
        trigger_price = trade.open_rate_requested * (1 + required_deviation)

        # 6. Only place a safety order if the current price has dropped enough
        if current_rate > trigger_price:
            return None

        # ------------------------------------------------------------------
        # 7. Calculate Safety Order stake size (volume scaling)
        #
        # safety_order_ratio:
        #   Defines the size of the FIRST safety order relative to the base order
        #
        # safety_order_volume_scale:
        #   Multiplier that increases each subsequent safety order size
        #
        # Example:
        #   SO1 → scale^0
        #   SO2 → scale^1
        #   SO3 → scale^2
        # ------------------------------------------------------------------

        base_order_size = self.base_order_size.value
        safety_order_ratio = self.safety_order_ratio.value
        volume_scale = self.safety_order_volume_scale.value

        # Base size for the first safety order
        so_base_amount = base_order_size * safety_order_ratio

        # Final safety order size with volume scaling applied
        so_amount = so_base_amount * (volume_scale ** (so_level - 1))

        # 8. Return the calculated stake amount and order tag
        return so_amount, f"SO{so_level}"


    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                 current_profit: float, **kwargs) -> bool:

        # Total invested capital
        total_invested_capital = trade.open_trade_value

        # Current value of the entire position at current market price
        current_position_value = current_rate * trade.amount

        # Absolute profit in quote currency (e.g. USDT)
        absolute_profit = current_position_value - total_invested_capital

        # Avoid division by zero if invested capital is zero or negative
        if total_invested_capital <= 0:
            return False

        # Profit percentage relative to invested capital
        profit_pct = absolute_profit / total_invested_capital

        # Exit signal when profit target is reached
        if profit_pct >= self.take_profit.value:
            return "tp_from_avg_entry"

        # Otherwise, keep holding
        return False


    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        for period in self.rsi_period.range:
            dataframe[f"rsi{period}"] = ta.RSI(dataframe, timperiod=period)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
 
        rsi_col = f"rsi{self.rsi_period.value}"

        # Generate trade entry condition based on RSI setting
        if self.rsi_condition.value == 'less_than':
            trade_start_condition = dataframe[rsi_col] < self.rsi_threshold.value
        elif self.rsi_condition.value == 'crossing_down':
            trade_start_condition = qtpylib.crossed_below(dataframe[rsi_col], self.rsi_threshold.value)
        elif self.rsi_condition.value == 'crossing_up':
            trade_start_condition = qtpylib.crossed_above(dataframe[rsi_col], self.rsi_threshold.value)
        else:
            # Default to no entry if condition is unrecognized
            trade_start_condition = False

        # Mark entries where conditions and volume > 0 are met
        dataframe.loc[
            (trade_start_condition) & (dataframe['volume'] > 0),
            ['enter_long', 'enter_tag']
        ] = (1, 'rsi_entry')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, 'exit_long'] = 0
        
        return dataframe