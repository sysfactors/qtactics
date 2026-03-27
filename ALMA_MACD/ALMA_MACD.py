# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Dict, Optional, Union, Tuple, List
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal
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
# ALMA + MACD Strategy: Why This Works Better (Real Backtest Results)
# https://youtu.be/x34p8VNitkY
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
    -c user_data/binance_futures_ALMA_MACD.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy ALMA_MACD \
    --config user_data/binance_futures_ALMA_MACD.json \
    --timeframe 1h \
    --timerange 20241101-20250701 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy \
    -e 100 \
    --job-workers 10 \
    --random-state 9319 \
    --min-trades 10 \
    --max-open-trades 1 \
    -p BCH/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy ALMA_MACD \
    --timeframe 1h \
    --timerange 20241101-20251101 \
    --breakdown month \
    -c user_data/binance_futures_ALMA_MACD.json \
    --max-open-trades 1 \
    --cache none \
    --timeframe-detail 5m \
    -p BCH/USDT:USDT
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_futures_ALMA_MACD.json
"""


class ALMA_MACD(IStrategy):
            
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1h"

    # Can this strategy go short?
    can_short: bool = False

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
    startup_candle_count: int = 250

    alma_window = CategoricalParameter([100, 150, 200], default=200, space="buy", optimize=True)
    alma_sigma = CategoricalParameter([10, 15, 20], default=10, space="buy", optimize=True)
    alma_offset = CategoricalParameter([0.7, 0.75, 0.8, 0.85], default=0.8, space="buy", optimize=True)

    alma_cross_window = CategoricalParameter([5, 10, 15], default=10, space="buy")
    macd_cross_window = CategoricalParameter([5, 10, 15], default=10, space="buy")
   
    risk_ratio = CategoricalParameter([1.5, 2, 2.5, 3], default=2, space="buy")    
    atr_mult = CategoricalParameter([1.5, 2, 2.5, 3], default=2, space="buy")

    @property
    def plot_config(self):

        plot_config = {
            "main_plot": {
                f"alma{self.alma_window.value}_{self.alma_sigma.value}_{self.alma_offset.value}": {
                    "color": "#FF9800",
                    "type": "line"
                    }
            },
            "subplots": {
                "MACD": {
                    "macd": {"color": "#2962ff", "fill_to": "macdhist"},
                    "macdsignal": {"color": "#ff6d00"},
                    "macdhist": {"type": "bar", "plotly": {"opacity": 0.9}}
                    }
            }
        }
        
        return plot_config
    
    
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
        
        for window_val in self.alma_window.range:
            for sigma_val in self.alma_sigma.range:
                for offset_val in self.alma_offset.range:
                    dataframe[f"alma{window_val}_{sigma_val}_{offset_val}"] = pta.alma(dataframe['close'], length=window_val, sigma=sigma_val, distribution_offset=offset_val, offset=0)
        
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        
        dataframe["atr"] = ta.ATR(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        
        
        
        long_conditions = []

        long_conditions.append(

                (dataframe["close"] > dataframe[f"alma{self.alma_window.value}_{self.alma_sigma.value}_{self.alma_offset.value}"]) &

                (dataframe["close"].rolling(window=self.alma_cross_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe[f"alma{self.alma_window.value}_{self.alma_sigma.value}_{self.alma_offset.value}"].iloc[x.index[0]:x.index[-1]+1])) 
                )) & 
                
                (dataframe["macd"] > dataframe["macdsignal"]) &
                
                (dataframe["macd"].rolling(window=self.macd_cross_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe["macdsignal"].iloc[x.index[0]:x.index[-1]+1])) 
                )) & 

                (dataframe["volume"] > 0)
            )

        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, long_conditions),
                "enter_long"] = 1
            
        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[:, "exit_long"] = 0

        return dataframe
    
    
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs):
        
        side = -1 if trade.is_short else 1

        # Retrieve TP and SL from custom data
        take_profit = trade.get_custom_data('take_profit')
        stop_loss = trade.get_custom_data('stop_loss')
        
        # If TP or SL is not set, initialize them
        if take_profit is None or stop_loss is None:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

            # Get the date just before trade opened
            trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)

            # Filter dataframe to candles before the trade opened
            signal_data = dataframe.loc[dataframe["date"] < trade_date]

            if signal_data.empty:
                logger.warning(f"[{pair}] No signal candle found. Skip setting TP/SL.")
                return None

            signal_candle = signal_data.iloc[-1]
            
            # Calculate TP and SL
            atr = signal_candle["atr"]
            close = signal_candle["close"]

            take_profit = close + side * self.atr_mult.value * atr * self.risk_ratio.value
            stop_loss = close - side * self.atr_mult.value * atr
            
            # Save to trade's custom data
            trade.set_custom_data('take_profit', take_profit)
            trade.set_custom_data('stop_loss', stop_loss)

            # logger.info(f"[{pair}] TP/SL set. TP: {take_profit:.2f}, SL: {stop_loss:.2f}")
            
            
        # Get the current close price
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_close = dataframe.iloc[-1]['close']

        # Check exit conditions
        if (trade.is_short and current_close <= take_profit) or \
        (not trade.is_short and current_close >= take_profit):
            # logger.info(f"[{pair}] Take Profit hit! Close: {current_close:.2f}, TP: {take_profit:.2f}")
            return "take_profit_achieved"

        if (trade.is_short and current_close >= stop_loss) or \
        (not trade.is_short and current_close <= stop_loss):
            # logger.info(f"[{pair}] Stop Loss hit! Close: {current_close:.2f}, SL: {stop_loss:.2f}")
            return "stop_loss_achieved"

        return None