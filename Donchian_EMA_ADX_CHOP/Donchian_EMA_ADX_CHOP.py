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
# Donchian Channel Strategy made 140% Profit! (Full Tutorial)
# https://youtu.be/CgYdfwrL1VQ
# ==========================================

# ================================
# Freqtrade Version
# ================================

"""
freqtrade -V

Operating System:       Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.36
Python Version:         Python 3.13.5
CCXT Version:           4.4.96

Freqtrade Version:      freqtrade 2025.7
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_Donchian_EMA_ADX_CHOP.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy Donchian_EMA_ADX_CHOP \
    --config user_data/binance_futures_Donchian_EMA_ADX_CHOP.json \
    --timeframe 1h \
    --timerange 20240601-20250201 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 30 \
    --max-open-trades 1 \
    -p ETH/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy Donchian_EMA_ADX_CHOP \
    --timeframe 1h \
    --timerange 20240601-20250601 \
    --breakdown month \
    -c user_data/binance_futures_Donchian_EMA_ADX_CHOP.json \
    --max-open-trades 1 \
    --timeframe-detail 5m \
    -p ETH/USDT:USDT \
    --cache none
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_futures_Donchian_EMA_ADX_CHOP.json
"""

class Donchian_EMA_ADX_CHOP(IStrategy):

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1h"

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
    use_custom_stoploss = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200
    
    # Strategy parameters
    donchian_period = CategoricalParameter([15, 20, 25, 30], default=20, space="buy")
    
    ema_period = CategoricalParameter([50, 100, 150, 200], default=200, space="buy")
    adx_threshold = CategoricalParameter([20, 25, 30], default=30, space="buy") 
    chop_threshold = CategoricalParameter([30, 35, 40], default=40, space="buy") 
    atr_mult = CategoricalParameter([2, 2.5, 3], default=2.5, space="buy")    
    
    leverage_level = IntParameter(1, 10, default=1, space='buy', optimize=False, load=False)
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                f'dc_upper{self.donchian_period.value}': {"color": "#2962FF"},
                f'dc_lower{self.donchian_period.value}': {"color": "#2962FF"}
            },
            "subplots": {
                
                "ADX": {
                    "adx": {"color": "#f23645", "type": "line"
                    }
                },
                "CHOP": {
                    "chop": {"color": "#2962FF", "type": "line"
                    }
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
        
        return []

    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        

        for val in self.donchian_period.range: 

            dataframe[f'dc_upper{val}'] = dataframe['high'].rolling(window=val).max().shift(1)
            dataframe[f'dc_lower{val}'] = dataframe['low'].rolling(window=val).min().shift(1)


        for val in self.ema_period.range: 
            dataframe[f"ema{val}"] = ta.EMA(dataframe, timeperiod=val)
        
        dataframe['chop'] = pta.chop(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            length=14
        )
        
        # ADX
        dataframe["adx"] = ta.ADX(dataframe)
        
        # ATR
        dataframe["atr"] = ta.ATR(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
                (
                    (qtpylib.crossed_above(dataframe["close"], dataframe[f'dc_upper{self.donchian_period.value}'])) &
                    (dataframe['close'] > dataframe[f"ema{self.ema_period.value}"]) &
                    (dataframe['adx'] > self.adx_threshold.value) &
                    (dataframe['chop'] < self.chop_threshold.value) &
                    (dataframe['volume'] > 0)
                ),
                'enter_long'] = 1
        
        dataframe.loc[
                (
                    (qtpylib.crossed_below(dataframe["close"], dataframe[f'dc_lower{self.donchian_period.value}'])) &
                    (dataframe['close'] < dataframe[f"ema{self.ema_period.value}"]) &
                    (dataframe['adx'] > self.adx_threshold.value) &
                    (dataframe['chop'] < self.chop_threshold.value) &
                    (dataframe['volume'] > 0)
                ),
                'enter_short'] = 1

        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

        return dataframe
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, after_fill: bool,
                        **kwargs) -> float | None:
        
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        candle = dataframe.iloc[-1].squeeze()
        side = 1 if trade.is_short else -1
        
        return stoploss_from_absolute(current_rate + (side * candle["atr"] * self.atr_mult.value), 
                                      current_rate=current_rate, 
                                      is_short=trade.is_short,
                                      leverage=trade.leverage)
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value