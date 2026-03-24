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
# Money Flow Index indicator made 300% Profit! (Full Tutorial)
# https://youtu.be/Uxfhuf2KdsI
# ==========================================

# ================================
# Freqtrade Version
# ================================

"""
freqtrade -V

Operating System:       Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.36
Python Version:         Python 3.13.7
CCXT Version:           4.5.7

Freqtrade Version:      freqtrade 2025.9.1
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_MFI_CMF.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy MFI_CMF \
    --config user_data/binance_futures_MFI_CMF.json \
    --timeframe 1h \
    --timerange 20240801-20250401 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 30 \
    --max-open-trades 1 \
    -p ARB/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy MFI_CMF \
    --timeframe 1h \
    --timerange 20240801-20250801 \
    --breakdown month \
    -c user_data/binance_futures_MFI_CMF.json \
    --max-open-trades 1 \
    --cache none \
    --timeframe-detail 5m \
    -p ARB/USDT:USDT
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_futures_MFI_CMF.json
"""

class MFI_CMF(IStrategy):

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
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200
    
    # Strategy parameters
    cmf_length = CategoricalParameter([20, 30, 40 ,50], default=50, space="buy", optimize=True)
    cmf_threshold = CategoricalParameter([0.03, 0.05, 0.07], default=0.05, space="buy", optimize=True) 
    mfi_threshold = CategoricalParameter([50, 55, 60, 65], default=55, space="buy")
    
    risk_ratio = CategoricalParameter([2, 2.5], default=2, space="buy")    
    atr_mult = CategoricalParameter([2, 2.5, 3, 3.5], default=2.5, space="buy")

    leverage_level = IntParameter(1, 10, default=1, space="buy", optimize=False, load=False)
    
    @property
    def plot_config(self):
        return {
            "main_plot": {
                "ema200": {"color": "#2962FF"}
            },
            "subplots": {
                "MFI": {
                    "mfi": {"color": "#7E57C2"}
                },
                "CMF": {
                    f"cmf{self.cmf_length.value}": {"color": "#43A047"}
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
        
        # Money Flow Index (MFI)
        dataframe["mfi"] = ta.MFI(dataframe)

        # Chaikin Money Flow (CMF)
        for val in self.cmf_length.range:
            dataframe[f"cmf{val}"] = pta.cmf(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                volume=dataframe['volume'],
                length=val)

        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
            
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[
            (
                (dataframe['mfi'] > self.mfi_threshold.value) &
                (qtpylib.crossed_above(dataframe[f"cmf{self.cmf_length.value}"], self.cmf_threshold.value)) &
                (dataframe["close"] > dataframe["ema200"]) &
                (dataframe["volume"] > 0) 
            ),
            "enter_long"] = 1
        
        dataframe.loc[
            (
                (dataframe['mfi'] < (100 - self.mfi_threshold.value)) &
                (qtpylib.crossed_below(dataframe[f"cmf{self.cmf_length.value}"], - self.cmf_threshold.value)) &
                (dataframe["close"] < dataframe["ema200"]) &
                (dataframe["volume"] > 0) 
            ),
            "enter_short"] = 1
        
        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0

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

        # Check exit conditions
        if (trade.is_short and current_rate <= take_profit) or \
        (not trade.is_short and current_rate >= take_profit):
            # logger.info(f"[{pair}] Take Profit hit! Close: {current_close:.2f}, TP: {take_profit:.2f}")
            return "take_profit_achieved"

        if (trade.is_short and current_rate >= stop_loss) or \
        (not trade.is_short and current_rate <= stop_loss):
            # logger.info(f"[{pair}] Stop Loss hit! Close: {current_close:.2f}, SL: {stop_loss:.2f}")
            return "stop_loss_achieved"

        return None
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value
    
    