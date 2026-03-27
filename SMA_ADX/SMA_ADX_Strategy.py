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
# ADX & Moving Averages: Freqtrade Crypto Strategy That Actually Works!
# YouTube Link: https://youtu.be/S6l3nZ3YkWw
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

freqtrade list-data -c user_data/config_binance_futures.json --show-timerange

freqtrade download-data \
    -c user_data/config_binance_futures.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy SMA_ADX_Strategy \
    --config user_data/binance_futures_SMA_ADX_Strategy.json \
    --timeframe 4h \
    --timerange 20240201-20241001 \
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
    --strategy SMA_ADX_Strategy \
    --timeframe 4h \
    --timerange 20240201-20250201 \
    --breakdown month \
    -c user_data/binance_futures_SMA_ADX_Strategy.json \
    --max-open-trades 1 \
    --cache none \
    --timeframe-detail 15m \
    -p GRT/USDT:USDT
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/config_binance_futures.json
"""

class SMA_ADX_Strategy(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "4h"
    informative_timeframe = '1d'

    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {}
    
    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.25

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
    short_sma_period = CategoricalParameter([10, 20], default=20, space="buy")
    long_sma_period = CategoricalParameter([100, 150, 200], default=200, space="buy")
    
    adx_period = CategoricalParameter([4, 5, 6], default=5, space="buy")
    adx_threshold = CategoricalParameter([25, 30, 35, 40], default=25, space="buy")
    
    atr_mult = CategoricalParameter([0.3, 0.5, 0.7, 1, 1.5, 2], default=0.5, space="buy")
    
    atr_price_threshold = CategoricalParameter([0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03], default=0.005, space="buy")
    
    cross_rolling_window = CategoricalParameter([2, 3, 4, 5], default=5, space="buy")

    leverage_level = IntParameter(1, 10, default=1, space='buy', optimize=False, load=False)
    
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                f"sma_high_{self.short_sma_period.value}": {"color": "#4CAF50"},
                f"sma_low_{self.short_sma_period.value}": {"color": "#f23645"},
                f"sma_long_{self.long_sma_period.value}": {"color": "#ffeb3b"}
                
            },
            "subplots": {
                # Subplots - each dict defines one additional plot 
                "ADX": {
                    f"adx_{self.adx_period.value}": {"color": "#2962ff"},
                },
                "ATR": {
                    "atr_1d": {"color": "#ff9800"}
                }
            }
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()

        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]

        return informative_pairs
    
    @informative('1d')
    def populate_indicators_1d(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        return dataframe
    

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        for period in self.short_sma_period.range:
            dataframe[f"sma_high_{period}"] = ta.SMA(dataframe["high"], timeperiod=period)
            dataframe[f"sma_low_{period}"] = ta.SMA(dataframe["low"], timeperiod=period)
            
        for period in self.long_sma_period.range:
            dataframe[f"sma_long_{period}"] = ta.SMA(dataframe["close"], timeperiod=period)

        for period in self.adx_period.range:
            dataframe[f"adx_{period}"] = ta.ADX(dataframe, timeperiod=period)
            
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
    
        # Define entry conditions - Long Position
        dataframe.loc[
            (
                # Close price is above the long-term SMA (confirming an overall uptrend)
                (dataframe['close'] > dataframe[f"sma_long_{self.long_sma_period.value}"]) &

                # Close price has crossed above the short-term high SMA within the given rolling window (confirming a breakout)
                (dataframe['close'].rolling(window=self.cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(
                        x, dataframe[f"sma_high_{self.short_sma_period.value}"].iloc[x.index[0]:x.index[-1]+1])) 
                )) &

                # ADX is above the threshold, confirming a strong trend
                (dataframe[f"adx_{self.adx_period.value}"] > self.adx_threshold.value) &

                # 1-day ATR is greater than a percentage of the close price, ensuring sufficient volatility
                (dataframe["atr_1d"] > dataframe["close"] * self.atr_price_threshold.value) &

                # Volume must be greater than zero to ensure valid data
                (dataframe["volume"] > 0) 
            ),
            "enter_long"] = 1

        # Define entry conditions - Short Position
        dataframe.loc[
            (
                # Close price is below the long-term SMA (confirming an overall downtrend)
                (dataframe['close'] < dataframe[f"sma_long_{self.long_sma_period.value}"]) &

                # Close price has crossed below the short-term low SMA within the given rolling window (confirming a breakdown)
                (dataframe['close'].rolling(window=self.cross_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(
                        x, dataframe[f"sma_low_{self.short_sma_period.value}"].iloc[x.index[0]:x.index[-1]+1])) 
                )) &

                # ADX is above the threshold, confirming a strong trend
                (dataframe[f"adx_{self.adx_period.value}"] > self.adx_threshold.value) &

                # 1-day ATR is greater than a percentage of the close price, ensuring sufficient volatility
                (dataframe["atr_1d"] > dataframe["close"] * self.atr_price_threshold.value) &

                # Volume must be greater than zero to ensure valid data
                (dataframe["volume"] > 0) 
            ),
            "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Define exit conditions - Long Position
        dataframe.loc[
            (
                # Price crosses below the adjusted short-term low SMA minus ATR multiplier (indicating a potential downtrend)
                (qtpylib.crossed_below(
                    dataframe['close'], 
                    dataframe[f"sma_low_{self.short_sma_period.value}"] - (self.atr_mult.value * dataframe["atr"])
                )) & 
                
                # Ensure there is trading volume (to avoid false signals)
                (dataframe["volume"] > 0) 
            ),
            "exit_long"] = 1

        # Define exit conditions - Short Position
        dataframe.loc[
            (
                # Price crosses above the adjusted short-term high SMA plus ATR multiplier (indicating a potential uptrend)
                (qtpylib.crossed_above(
                    dataframe['close'], 
                    dataframe[f"sma_high_{self.short_sma_period.value}"] + (dataframe['atr'] * self.atr_mult.value)
                )) & 
                
                # Ensure there is trading volume (to avoid false signals)
                (dataframe["volume"] > 0) 
            ),
            "exit_short"] = 1

        return dataframe

        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value