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
# Supertrend Strategy to Python Bot: 212% Profit (Full Code Inside!)
# https://youtu.be/Z_vIXP0mOJA
# ==========================================

# ================================
# Freqtrade Version
# ================================

"""
freqtrade -V

Operating System:       Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.36
Python Version:         Python 3.12.9
CCXT Version:           4.4.62

Freqtrade Version:      freqtrade 2025.2
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_SuperTrend_EMA.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy SuperTrend_EMA \
    --config user_data/binance_futures_SuperTrend_EMA.json \
    --timeframe 1h \
    --timerange 20240301-20241101 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 55 \
    --j -2 \
    --random-state 9319 \
    --min-trades 20 \
    --max-open-trades 1 \
    -p GRT/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy SuperTrend_EMA \
    --timeframe 1h \
    --timerange 20240301-20250301 \
    --breakdown month \
    -c user_data/binance_futures_SuperTrend_EMA.json \
    --max-open-trades 1 \
    --timeframe-detail 5m \
    -p GRT/USDT:USDT \
    --cache none
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_futures_SuperTrend_EMA.json
"""

class SuperTrend_EMA(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1h"
    informative_timeframe = "4h"

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}
    
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.20

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
    
    # Strategy parameters
    supertrend_length = IntParameter(10, 16, default=10, space='buy') 
    supertrend_multiplier = IntParameter(2, 5, default=4, space='buy') 
    
    ema_length = CategoricalParameter([50, 100, 200], default=50, space="buy")
    max_distance_percent = CategoricalParameter([0.02, 0.04, 0.06], default=0.02, space="buy")
    atr_window = CategoricalParameter([10, 30, 50, 70, 100], default=30, space="buy")
    atr_threshold_low_pct = CategoricalParameter([0.1, 0.2, 0.3, 0.4], default=0.5, space="buy")    
    
    leverage_level = IntParameter(1, 10, default=1, space='buy', optimize=False, load=False)
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                f"supertrend_{self.supertrend_length.value}_{self.supertrend_multiplier.value}": {
                    "color": "#4caf50",
                    "type": "line",
                    "fill_to": "close"
                },
                f"ema_{self.ema_length.value}": {
                    "color": "#2962ff",
                    "type": "line"
                }
            },
            "subplots": {
                "ATR": {
                    "atr_percent_4h": {"color": "#b71c1c"},
                    "atr_threshold_low": {"color": "#F7AC06"}
                },
                "ADX": {
                    "adx": {"color": "#f23645"},
                    "dynamic_adx_threshold": {"color": "#F7AC06"}
                }
            }
        }

    def informative_pairs(self):

        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        
        return informative_pairs
    
    @informative('4h')
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_percent"] = (dataframe["atr"] / dataframe["close"]) * 100
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        for multiplier in self.supertrend_multiplier.range:
            for period in self.supertrend_length.range:
                superTrend = pta.supertrend(dataframe['high'], dataframe['low'], dataframe['close'], length=period, multiplier=multiplier)
                dataframe[f'supertrend_{period}_{multiplier}'] = superTrend[f'SUPERT_{period}_{multiplier}.0']
                dataframe[f'supertrend_direction_{period}_{multiplier}'] = superTrend[f'SUPERTd_{period}_{multiplier}.0']
        
        for period in self.ema_length.range:
            dataframe[f'ema_{period}'] = ta.EMA(dataframe, timeperiod=period)
    
        dataframe["adx"] = ta.ADX(dataframe)
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe['atr_threshold_low'] = dataframe['atr_percent_4h'].rolling(window=self.atr_window.value).quantile(self.atr_threshold_low_pct.value)
        dataframe['dynamic_adx_threshold'] = 20 + (dataframe['atr_percent_4h'] * 2)
        
        supertrend_col = f'supertrend_direction_{self.supertrend_length.value}_{self.supertrend_multiplier.value}'
        
        ema_value = dataframe[f'ema_{self.ema_length.value}']
        distance = ema_value * self.max_distance_percent.value

        dataframe.loc[
                (
                    (dataframe[supertrend_col] == 1) &
                    (dataframe['close'] > (ema_value + distance)) &
                    (
                        (dataframe[supertrend_col].shift(1) == -1) |
                        (dataframe['close'].shift(1) < (ema_value.shift(1) + distance.shift(1)))
                    ) &
                    (dataframe['adx'] > dataframe['dynamic_adx_threshold']) &
                    (dataframe['atr_percent_4h'] >= dataframe['atr_threshold_low']) &
                    (dataframe['volume'] > 0) 
                ),
                'enter_long'] = 1
        
        
        dataframe.loc[
                (
                    (dataframe[supertrend_col] == -1) &
                    (dataframe['close'] < (ema_value - distance)) &
                    (
                        (dataframe[supertrend_col].shift(1) == 1) |
                        (dataframe['close'].shift(1) > (ema_value.shift(1) - distance.shift(1)))
                    ) &
                    (dataframe['adx'] > dataframe['dynamic_adx_threshold']) &
                    (dataframe['atr_percent_4h'] >= dataframe['atr_threshold_low']) &
                    (dataframe['volume'] > 0) 
                ),
                'enter_short'] = 1
        
        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        supertrend_col = f'supertrend_direction_{self.supertrend_length.value}_{self.supertrend_multiplier.value}'
        
        dataframe.loc[
            (
                (dataframe[supertrend_col] == -1)
            ),
            'exit_long'
        ] = 1
        
        dataframe.loc[
            (
                (dataframe[supertrend_col] == 1)
            ),
            'exit_short'
        ] = 1

        return dataframe
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value