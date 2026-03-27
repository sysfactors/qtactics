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
# Freqtrade: ADX Indicator Strategy in Python makes 329%
# https://youtu.be/1kcFlxs7ufQ
# ==========================================


# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_TemaAdxCmo.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 2h 4h 1d
"""


# ================================
# Lookahead Analysis 
# ================================

"""
freqtrade lookahead-analysis \
    --strategy TemaAdxCmo \
    --timeframe 1h \
    --timerange 20240601-20250201 \
    -c user_data/binance_futures_TemaAdxCmo.json \
    --max-open-trades 1 \
    -p APT/USDT:USDT
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy TemaAdxCmo \
    --config user_data/binance_futures_TemaAdxCmo.json \
    --timeframe 1h \
    --timerange 20240601-20250201 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 30 \
    --max-open-trades 1 \
    -p APT/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy TemaAdxCmo \
    --timeframe 1h \
    --timerange 20240601-20250601 \
    --breakdown month \
    -c user_data/binance_futures_TemaAdxCmo.json \
    --max-open-trades 1 \
    --cache none \
    --timeframe-detail 5m \
    -p APT/USDT:USDT
"""

# ================================
# Start FreqUI Web Interface
# ================================

"""
freqtrade webserver \
    --config user_data/binance_futures_TemaAdxCmo.json
"""


class TemaAdxCmo(IStrategy):
            
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

    # Dictionary defining the exit points for take profit and stop loss levels.
    exit_loss_profit = {}
    
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
    
    tema_rolling_window = CategoricalParameter([4, 8], default=4, space="buy")
    adx_threshold = CategoricalParameter([25, 30, 35], default=30, space="buy")
    atr_mult = CategoricalParameter([1.5, 2, 2.5], default=2, space="buy")
    risk_ratio = CategoricalParameter([2, 2.5, 3], default=2, space="buy")    
    
    leverage_level = IntParameter(1, 10, default=1, space='buy', optimize=False, load=False)

    @property
    def plot_config(self):

        plot_config = {
            "main_plot": {
                "tema8": {"color": "red"},
                "tema13": {"color": "blue"},
                "tema21": {"color": "green"},
            },
            "subplots": {
                "ADX": {
                        "adx": {
                            "color": "#f50057",
                            "type": "line"
                        }
                    },
                "CMO": {
                    "cmo": {
                        "color": "#2962ff",
                        "type": "line"
                    }
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
        
        # get access to all pairs available in whitelist.
        # pairs = self.dp.current_whitelist()

        # # Assign tf to each pair so they can be downloaded and cached for strategy.
        # informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        
        return []
    
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe["tema8"] = ta.TEMA(dataframe, timeperiod=8)
        dataframe["tema13"] = ta.TEMA(dataframe, timeperiod=13)
        dataframe["tema21"] = ta.TEMA(dataframe, timeperiod=21)

        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["cmo"] = ta.CMO(dataframe, timeperiod=14)

        dataframe["atr"] = ta.ATR(dataframe)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:        

        dataframe.loc[
            (   
                (dataframe["tema8"] > dataframe["tema13"]) &
                (dataframe["tema13"] > dataframe["tema21"]) &

                # Check if TEMA(8) crossed above TEMA(21) in the rolling window
                (dataframe["tema8"].rolling(window=self.tema_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe["tema21"].iloc[x.index[0]:x.index[-1]+1])))) &
                
                # Check if TEMA(13) crossed above TEMA(21) in the rolling window
                (dataframe["tema13"].rolling(window=self.tema_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe["tema21"].iloc[x.index[0]:x.index[-1]+1])))) &

                (dataframe["adx"] > self.adx_threshold.value) &
                (dataframe["cmo"] > 0) &
                (dataframe["volume"] > 0)

            ),
            'enter_long'] = 1

        dataframe.loc[
            (
                (dataframe["tema8"] < dataframe["tema13"]) &
                (dataframe["tema13"] < dataframe["tema21"]) &

                # Check if TEMA(8) crossed below TEMA(13) in the rolling window
                (dataframe["tema8"].rolling(window=self.tema_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe["tema21"].iloc[x.index[0]:x.index[-1]+1])))) &
                
                # Check if TEMA(13) crossed below TEMA(21) in the rolling window
                (dataframe["tema13"].rolling(window=self.tema_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe["tema21"].iloc[x.index[0]:x.index[-1]+1])))) &

                (dataframe["adx"] > self.adx_threshold.value) &
                (dataframe["cmo"] < 0) &
                (dataframe["volume"] > 0)
            ),
            'enter_short'] = 1

        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

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
        
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_level.value