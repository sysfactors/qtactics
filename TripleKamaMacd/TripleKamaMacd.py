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
# Triple KAMA & MACD Strategy Made 214% Profit! (Full Tutorial)
# YouTube Link: https://youtu.be/g8k3CGlMtYM
# ==========================================

# ================================
# Freqtrade Version
# ================================

"""
freqtrade -V

Operating System:       Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.36
Python Version:         Python 3.12.10
CCXT Version:           4.4.82

Freqtrade Version:      freqtrade 2025.5
"""

# ================================
# Download Historical Data
# ================================

"""
freqtrade download-data \
    -c user_data/binance_futures_TripleKamaMacd.json \
    --timerange 20230101- \
    -t 1m 5m 15m 30m 1h 4h 1d
"""

# ================================
# Lookahead Analysis 
# ================================

"""
freqtrade lookahead-analysis \
    --strategy TripleKamaMacd \
    --timeframe 1h \
    --timerange 20240501-20250101 \
    -c user_data/binance_futures_TripleKamaMacd.json \
    --max-open-trades 1 \
    -p BCH/USDT:USDT
"""

# ================================
# Hyperopt Optimization
# ================================

"""
freqtrade hyperopt \
    --strategy TripleKamaMacd \
    --config user_data/binance_futures_TripleKamaMacd.json \
    --timeframe 1h \
    --timerange 20240501-20250101 \
    --hyperopt-loss MultiMetricHyperOptLoss \
    --spaces buy\
    -e 50 \
    --j -2 \
    --random-state 9319 \
    --min-trades 30 \
    --max-open-trades 1 \
    -p BCH/USDT:USDT
"""

# ================================
# Backtesting
# ================================

"""
freqtrade backtesting \
    --strategy TripleKamaMacd \
    --timeframe 1h \
    --timerange 20240501-20250501 \
    --breakdown month \
    -c user_data/binance_futures_TripleKamaMacd.json \
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
    --config user_data/binance_futures_TripleKamaMacd.json
"""


class TripleKamaMacd(IStrategy):

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
    kama_rolling_window = CategoricalParameter([4, 6], default=4, space="buy")
    macd_rolling_window = CategoricalParameter([8, 10], default=8, space="buy")

    risk_ratio = CategoricalParameter([2, 2.5], default=2, space="buy")    
    atr_mult = CategoricalParameter([2.5, 3], default=2.5, space="buy")
    
    leverage_level = IntParameter(1, 10, default=1, space="buy", optimize=False, load=False)
        
    @property
    def plot_config(self):
        
        return {
            "main_plot": {
                "kama8": {"color": "red"},
                "kama13": {"color": "blue"},
                "kama21": {"color": "green"},
            },
            "subplots": {
                
                "MACD": {
                    "macd": {"color": "#2962ff", "fill_to": "macdhist"},
                    "macdsignal": {"color": "#ff6d00"},
                    "macdhist": {"type": "bar", "plotly": {"opacity": 0.9}}
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
        
        # KAMA
        dataframe["kama8"] = ta.KAMA(dataframe, timeperiod=8)
        dataframe["kama13"] = ta.KAMA(dataframe, timeperiod=13)
        dataframe["kama21"] = ta.KAMA(dataframe, timeperiod=21)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]
        
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (

                (dataframe["kama8"] > dataframe["kama13"]) &
                (dataframe["kama13"] > dataframe["kama21"]) &

                # Check if KAMA(8) crossed above KAMA(21) in the rolling window
                (dataframe["kama8"].rolling(window=self.kama_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe["kama21"].iloc[x.index[0]:x.index[-1]+1])))) &
                
                # Check if KAMA(13) crossed above KAMA(21) in the rolling window
                (dataframe["kama13"].rolling(window=self.kama_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe["kama21"].iloc[x.index[0]:x.index[-1]+1])))) &

               # MACD crossed above MACD signal
                (dataframe["macd"] > dataframe["macdsignal"]) &
                (dataframe["macd"].rolling(window=self.macd_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_above(x, dataframe["macdsignal"].iloc[x.index[0]:x.index[-1]+1])) 
                )) &

                (dataframe["volume"] > 0)
            ),
            "enter_long"] = 1

        dataframe.loc[
            (

                (dataframe["kama8"] < dataframe["kama13"]) &
                (dataframe["kama13"] < dataframe["kama21"]) &

                # Check if KAMA(8) crossed below KAMA(13) in the rolling window
                (dataframe["kama8"].rolling(window=self.kama_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe["kama21"].iloc[x.index[0]:x.index[-1]+1])))) &
                # Check if KAMA(13) crossed below KAMA(21) in the rolling window
                (dataframe["kama13"].rolling(window=self.kama_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe["kama21"].iloc[x.index[0]:x.index[-1]+1])))) &

                # MACD crossed below MACD signal
                (dataframe['macd'] < dataframe['macdsignal']) &
                (dataframe['macd'].rolling(window=self.macd_rolling_window.value).apply(
                    lambda x: any(qtpylib.crossed_below(x, dataframe['macdsignal'].iloc[x.index[0]:x.index[-1]+1])) 
                )) &
                
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